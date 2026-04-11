# BEIR-Style RAG Eval

This folder contains a standalone benchmark harness for evaluating the current
`NutritionRAG` retrieval stack on BEIR-format datasets without modifying the
existing production/local pipeline files.

## Expected dataset layout

```text
<data-folder>/
  corpus.jsonl
  queries.jsonl
  qrels/
    test.tsv
```

## What it does

- loads the current `NutritionRAG` model stack
- builds an in-memory FAISS index from the benchmark corpus
- runs the current retrieval pipeline over the benchmark queries
- computes:
  - NDCG@k
  - Recall@k
  - Precision@k
  - MRR@k
- saves:
  - a TREC-style runfile
  - a metrics JSON

## Example

```bash
cd /Users/rakshith911/Documents/food_detection/FoodAI/nutrition-video-analysis/terraform/docker
OMP_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE ./venv311/bin/python benchmark_eval/run_beir_style_rag_eval.py \
  --data-folder /path/to/beir_dataset \
  --split test
```

## Download public BEIR datasets

```bash
cd /Users/rakshith911/Documents/food_detection/FoodAI/nutrition-video-analysis/terraform/docker
./venv311/bin/python benchmark_eval/download_beir_datasets.py
```

This downloads a practical default set:
- `scifact`
- `nfcorpus`
- `arguana`
- `fiqa`

## Notes

- This is a broad retrieval sanity check, not a nutrition-domain correctness benchmark.
- It does not change any existing RAG scripts or database files.
- For large BEIR corpora, evaluation can be expensive because it embeds the full corpus locally.
