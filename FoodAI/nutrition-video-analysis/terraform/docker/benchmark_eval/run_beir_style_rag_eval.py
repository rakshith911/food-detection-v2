#!/usr/bin/env python3
"""
Evaluate the current NutritionRAG retrieval stack on a BEIR-format dataset
without modifying any production pipeline files.

Expected dataset layout:
  <data-folder>/
    corpus.jsonl
    queries.jsonl
    qrels/
      test.tsv

Examples:
  python run_beir_style_rag_eval.py --data-folder /path/to/scifact --split test
  python run_beir_style_rag_eval.py --data-folder /path/to/scifact --split test --limit-queries 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

import sys


DOCKER_DIR = Path(__file__).resolve().parents[1]
APP_DIR = DOCKER_DIR / "app"
if str(DOCKER_DIR) not in sys.path:
    sys.path.insert(0, str(DOCKER_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from nutrition_rag_system import NutritionRAG  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, required=True, help="BEIR dataset folder")
    parser.add_argument("--split", default="test", help="Qrels split name, e.g. test/dev/train")
    parser.add_argument("--top-k", type=int, default=100, help="Max results to keep per query")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size for corpus encoding")
    parser.add_argument("--limit-queries", type=int, default=0, help="Optional cap on number of queries to score")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DOCKER_DIR / "benchmark_eval" / "results",
        help="Where to write runfile and metrics JSON",
    )
    return parser.parse_args()


def load_corpus(path: Path) -> tuple[list[dict], list[str], list[str]]:
    docs: list[dict] = []
    doc_ids: list[str] = []
    names: list[str] = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            doc_id = str(row["_id"])
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            display = title if title else text[:200]
            combined = f"{title}. {text}".strip(". ").strip()
            if not combined:
                combined = display or doc_id
            docs.append(
                {
                    "id": doc_id,
                    "source": "benchmark",
                    "source_id": doc_id,
                    "source_table": "beir",
                    "description": display or combined,
                    "normalized_description": NutritionRAG._normalize_food_name(display or combined),
                    "ingredients_text": None,
                    "calories_per_100g": None,
                    "protein_g": None,
                    "carbs_g": None,
                    "fat_g": None,
                    "density_g_ml": None,
                    "density_method": None,
                    "density_source": None,
                    "cup_g": None,
                    "tbsp_g": None,
                    "tsp_g": None,
                    "fl_oz_g": None,
                    "benchmark_text": combined,
                }
            )
            doc_ids.append(doc_id)
            names.append(combined)
    return docs, doc_ids, names


def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            queries[str(row["_id"])] = str(row["text"])
    return queries


def load_qrels(path: Path, valid_query_ids: set[str]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = str(row["query-id"])
            if query_id not in valid_query_ids:
                continue
            corpus_id = str(row["corpus-id"])
            score = int(float(row["score"]))
            qrels.setdefault(query_id, {})[corpus_id] = score
    return qrels


def build_rag_for_corpus(docs: list[dict], names: list[str], *, batch_size: int) -> NutritionRAG:
    rag = NutritionRAG()
    rag.load()
    rag._use_unified = True
    rag._unified_foods = docs
    rag._unified_names = names

    vectors = rag._embedder.encode(  # type: ignore[union-attr]
        names,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    rag._unified_index = index
    rag._clip_index = None
    rag._clip_name_embeddings = None
    return rag


def retrieve_results(rag: NutritionRAG, queries: dict[str, str], *, top_k: int) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for query_id, text in queries.items():
        candidates = rag._retrieve_candidates(text, crop_image=None, top_k=top_k)
        ranked: dict[str, float] = {}
        for candidate in candidates[:top_k]:
            entry = candidate["entry"]
            doc_id = str(entry["id"])
            ranked[doc_id] = float(candidate["selection_score"])
        results[query_id] = ranked
    return results


def _dcg(relevances: Iterable[int]) -> float:
    total = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel <= 0:
            continue
        total += (2**rel - 1) / math.log2(idx + 1)
    return total


def evaluate_results(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10, 100),
) -> dict:
    metrics: dict[str, dict[str, float]] = {
        "ndcg": {},
        "recall": {},
        "precision": {},
        "mrr": {},
    }

    query_ids = [query_id for query_id in qrels.keys() if query_id in results]
    if not query_ids:
        raise ValueError("No overlapping query ids between qrels and results")

    for k in ks:
        ndcg_total = 0.0
        recall_total = 0.0
        precision_total = 0.0
        mrr_total = 0.0

        for query_id in query_ids:
            gold = qrels[query_id]
            ranked = sorted(results.get(query_id, {}).items(), key=lambda item: item[1], reverse=True)[:k]
            ranked_ids = [doc_id for doc_id, _score in ranked]
            ranked_rels = [gold.get(doc_id, 0) for doc_id in ranked_ids]

            ideal_rels = sorted(gold.values(), reverse=True)[:k]
            dcg = _dcg(ranked_rels)
            idcg = _dcg(ideal_rels)
            ndcg_total += (dcg / idcg) if idcg > 0 else 0.0

            relevant_total = sum(1 for rel in gold.values() if rel > 0)
            hits = sum(1 for rel in ranked_rels if rel > 0)
            recall_total += (hits / relevant_total) if relevant_total else 0.0
            precision_total += (hits / k) if k else 0.0

            reciprocal_rank = 0.0
            for idx, doc_id in enumerate(ranked_ids, start=1):
                if gold.get(doc_id, 0) > 0:
                    reciprocal_rank = 1.0 / idx
                    break
            mrr_total += reciprocal_rank

        denom = float(len(query_ids))
        metrics["ndcg"][f"@{k}"] = ndcg_total / denom
        metrics["recall"][f"@{k}"] = recall_total / denom
        metrics["precision"][f"@{k}"] = precision_total / denom
        metrics["mrr"][f"@{k}"] = mrr_total / denom

    metrics["query_count"] = len(query_ids)
    return metrics


def save_runfile(path: Path, results: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for query_id, ranked in results.items():
            sorted_docs = sorted(ranked.items(), key=lambda item: item[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                handle.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} nutrition-rag\n")


def main() -> int:
    args = parse_args()
    data_folder = args.data_folder
    corpus_path = data_folder / "corpus.jsonl"
    queries_path = data_folder / "queries.jsonl"
    qrels_path = data_folder / "qrels" / f"{args.split}.tsv"

    docs, _doc_ids, names = load_corpus(corpus_path)
    queries_all = load_queries(queries_path)
    qrels = load_qrels(qrels_path, set(queries_all.keys()))
    qrel_query_ids = [query_id for query_id in queries_all.keys() if query_id in qrels]
    if args.limit_queries:
        qrel_query_ids = qrel_query_ids[: args.limit_queries]
    queries = {query_id: queries_all[query_id] for query_id in qrel_query_ids}
    qrels = {query_id: qrels[query_id] for query_id in qrel_query_ids}

    rag = build_rag_for_corpus(docs, names, batch_size=args.batch_size)
    results = retrieve_results(rag, queries, top_k=args.top_k)
    metrics = evaluate_results(qrels, results)

    dataset_name = data_folder.name
    args.results_dir.mkdir(parents=True, exist_ok=True)
    runfile_path = args.results_dir / f"{dataset_name}.{args.split}.run.trec"
    metrics_path = args.results_dir / f"{dataset_name}.{args.split}.metrics.json"
    save_runfile(runfile_path, results)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Saved runfile: {runfile_path}")
    print(f"Saved metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
