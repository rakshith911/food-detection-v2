#!/usr/bin/env python3
"""
Unit tests for NutritionRAG retrieval fusion logic.
These tests avoid loading real ML dependencies by stubbing module imports.
"""
import sys
import types
from pathlib import Path


def _install_stub_modules() -> None:
    if "faiss" not in sys.modules:
        faiss = types.ModuleType("faiss")
        faiss.IndexFlatIP = object
        faiss.read_index = lambda *_args, **_kwargs: None
        sys.modules["faiss"] = faiss

    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")

        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        torch.no_grad = lambda: _NoGrad()
        sys.modules["torch"] = torch

    if "sentence_transformers" not in sys.modules:
        sentence_transformers = types.ModuleType("sentence_transformers")

        class SentenceTransformer:  # pragma: no cover - simple import stub
            pass

        sentence_transformers.SentenceTransformer = SentenceTransformer
        sys.modules["sentence_transformers"] = sentence_transformers

    if "sentence_transformers.cross_encoder" not in sys.modules:
        cross_encoder_mod = types.ModuleType("sentence_transformers.cross_encoder")

        class CrossEncoder:  # pragma: no cover - simple import stub
            pass

        cross_encoder_mod.CrossEncoder = CrossEncoder
        sys.modules["sentence_transformers.cross_encoder"] = cross_encoder_mod

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")

        class CLIPModel:  # pragma: no cover - simple import stub
            pass

        class CLIPProcessor:  # pragma: no cover - simple import stub
            pass

        transformers.CLIPModel = CLIPModel
        transformers.CLIPProcessor = CLIPProcessor
        sys.modules["transformers"] = transformers


def _build_rag():
    _install_stub_modules()
    sys.path.insert(0, str(Path(__file__).parent))
    from nutrition_rag_system import NutritionRAG

    rag = NutritionRAG()
    rag._use_unified = True
    rag._unified_index = object()
    rag._unified_foods = [
        {"source": "usda", "density_g_ml": 1.0, "calories_per_100g": 100.0, "density_method": "cup(test)"},
        {"source": "usda", "density_g_ml": 1.1, "calories_per_100g": 200.0, "density_method": "cup(test)"},
    ]
    rag._unified_names = ["apple pie", "caramel sauce"]
    return rag


class _DummyCrossEncoder:
    def predict(self, pairs):
        scores = []
        for _query, matched in pairs:
            scores.append(9.0 if matched == "apple pie" else 8.0)
        return scores


class _NaNCrossEncoder:
    def predict(self, pairs):
        return [float("nan") for _query, _matched in pairs]


def test_retrieve_candidates_merges_clip_and_text_hits():
    rag = _build_rag()
    rag._cross_encoder = _DummyCrossEncoder()
    rag._clip_fused_search = lambda _variant, _crop, _top_k: [(1, 0.62)]
    rag._faiss_search = lambda _index, _labels, _top_k: [(0, 0.88)]

    candidates = rag._retrieve_candidates("apple pie", crop_image=None, top_k=5)

    assert [item["matched"] for item in candidates] == ["apple pie", "caramel sauce"]
    assert candidates[0]["retrieval_source"] == "text"
    assert abs(candidates[0]["retrieval_sim"] - 0.88) < 1e-6
    assert candidates[1]["retrieval_source"] == "clip"
    assert abs(candidates[1]["retrieval_sim"] - 0.62) < 1e-6


def test_retrieve_candidates_keeps_best_hybrid_similarity():
    rag = _build_rag()
    rag._cross_encoder = _DummyCrossEncoder()
    rag._clip_fused_search = lambda _variant, _crop, _top_k: [(0, 0.57)]
    rag._faiss_search = lambda _index, _labels, _top_k: [(0, 0.91)]

    candidates = rag._retrieve_candidates("apple pie", crop_image=None, top_k=5)

    assert len(candidates) == 1
    assert candidates[0]["matched"] == "apple pie"
    assert candidates[0]["retrieval_source"] == "hybrid"
    assert abs(candidates[0]["clip_sim"] - 0.57) < 1e-6
    assert abs(candidates[0]["text_sim"] - 0.91) < 1e-6
    assert abs(candidates[0]["retrieval_sim"] - 0.91) < 1e-6


def test_non_finite_cross_encoder_scores_fallback_cleanly():
    rag = _build_rag()
    rag._cross_encoder = _NaNCrossEncoder()
    rag._clip_fused_search = lambda _variant, _crop, _top_k: []
    rag._faiss_search = lambda _index, _labels, _top_k: [(0, 0.91)]

    candidates = rag._retrieve_candidates("apple pie", crop_image=None, top_k=5)

    assert len(candidates) == 1
    assert candidates[0]["matched"] == "apple pie"
    assert candidates[0]["cross_score"] == float("-inf")


if __name__ == "__main__":
    test_retrieve_candidates_merges_clip_and_text_hits()
    test_retrieve_candidates_keeps_best_hybrid_similarity()
    test_non_finite_cross_encoder_scores_fallback_cleanly()
    print("PASS: RAG retrieval fusion unit tests passed.")
