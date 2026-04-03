"""
Production image pipeline model helpers.
Integrates SAM3 and ZoeDepth assets from the local PRODUCTION folder.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import transformers

logger = logging.getLogger(__name__)

_PRODUCTION_MODEL_CACHE: dict[str, Any] = {}


def _ensure_local_zoedepth_on_path() -> None:
    candidates = []
    env_repo = os.environ.get("ZOEDEPTH_REPO_DIR")
    if env_repo:
        candidates.append(Path(env_repo))
    candidates.extend([
        Path("/app/vendor/ZoeDepth"),
        Path(__file__).resolve().parent.parent / "vendor" / "ZoeDepth",
    ])

    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            logger.info(f"[production_models] Added ZoeDepth repo to sys.path: {candidate}")
            break


def load_sam3(model_dir: str | Path, device: str = "cpu") -> tuple[Any, Any]:
    cache_key = f"sam3::{model_dir}::{device}"
    cached = _PRODUCTION_MODEL_CACHE.get(cache_key)
    if cached:
        logger.info(f"[production_models] Using cached SAM3 model from {model_dir} on {device}")
        return cached

    model_dir = str(model_dir)
    logger.info(
        f"[production_models] Loading SAM3 from {model_dir} on {device} "
        f"(transformers={transformers.__version__}, torch={torch.__version__})"
    )
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    result = (model, processor)
    _PRODUCTION_MODEL_CACHE[cache_key] = result
    return result


def _make_sam3_inputs(sam3_processor: Any, pil_image: Image.Image, prompt: str) -> dict[str, Any]:
    for kwargs in (
        {"images": pil_image, "text": prompt, "return_tensors": "pt"},
        {"images": pil_image, "text": [prompt], "return_tensors": "pt"},
        {"images": pil_image, "input_text": [prompt], "return_tensors": "pt"},
    ):
        try:
            return sam3_processor(**kwargs)
        except TypeError:
            continue

    w, h = pil_image.size
    cx, cy = w // 2, h // 2
    try:
        return sam3_processor(
            images=pil_image,
            input_points=[[[cx, cy]]],
            input_labels=[[1]],
            return_tensors="pt",
        )
    except Exception:
        return sam3_processor(images=pil_image, return_tensors="pt")


def run_sam3_image(
    sam3_model: Any,
    sam3_processor: Any,
    pil_image: Image.Image,
    prompts: list[str],
    device: str = "cpu",
    min_coverage: float = 0.002,
    max_coverage: float = 0.60,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        try:
            inputs = _make_sam3_inputs(sam3_processor, pil_image, prompt)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.no_grad():
                pred = sam3_model(**inputs)

            masks = getattr(pred, "pred_masks", None)
            if masks is None:
                out[prompt] = {"mask": None, "score": 0.0}
                continue

            if hasattr(pred, "iou_scores") and pred.iou_scores is not None:
                scores = pred.iou_scores.squeeze(0)
            elif hasattr(pred, "pred_iou_scores") and pred.pred_iou_scores is not None:
                scores = pred.pred_iou_scores.squeeze(0)
            else:
                scores = masks.sigmoid().mean(dim=(-2, -1)).squeeze(0)

            best_mask = None
            best_score = -1.0
            for idx in range(scores.shape[0]):
                score = float(scores[idx].item())
                if score <= best_score:
                    continue
                mask_np = masks[0, idx].sigmoid().cpu().numpy()
                binary = mask_np > 0.5
                coverage = float(binary.mean())
                if coverage < min_coverage or coverage > max_coverage:
                    continue
                best_mask = binary
                best_score = score

            out[prompt] = {
                "mask": best_mask.astype(bool) if best_mask is not None else None,
                "score": max(best_score, 0.0),
            }
        except Exception:
            out[prompt] = {"mask": None, "score": 0.0}
    return out


def _patch_torch_hub_for_local_midas(midas_repo_dir: str | Path) -> None:
    local_midas_repo = Path(midas_repo_dir)
    if not local_midas_repo.exists():
        return

    original_load = torch.hub.load
    if getattr(torch.hub.load, "_production_midas_patched", False):
        return

    def patched_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "intel-isl/MiDaS":
            return original_load(str(local_midas_repo), model, *args, source="local", **kwargs)
        return original_load(repo_or_dir, model, *args, **kwargs)

    patched_load._production_midas_patched = True
    torch.hub.load = patched_load


def zoedepth_available() -> bool:
    _ensure_local_zoedepth_on_path()
    spec = importlib.util.find_spec("zoedepth")
    available = spec is not None
    logger.info(
        f"[production_models] ZoeDepth availability check: available={available} "
        f"module_origin={getattr(spec, 'origin', None)}"
    )
    return available


def load_zoedepth(checkpoint_path: str | Path, midas_repo_dir: str | Path, device: str = "cpu") -> Any:
    if not zoedepth_available():
        raise RuntimeError("zoedepth package is not installed in the current Python environment")

    cache_key = f"zoedepth::{checkpoint_path}::{device}"
    cached = _PRODUCTION_MODEL_CACHE.get(cache_key)
    if cached:
        logger.info(f"[production_models] Using cached ZoeDepth model from {checkpoint_path} on {device}")
        return cached

    _patch_torch_hub_for_local_midas(midas_repo_dir)
    logger.info(
        f"[production_models] Loading ZoeDepth from checkpoint={checkpoint_path} "
        f"midas_repo={midas_repo_dir} device={device} torch={torch.__version__}"
    )

    from zoedepth.models import model_io
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config

    original_load_state_dict = model_io.load_state_dict

    def _load_state_dict_compat(model, state_dict):
        state_dict = state_dict.get("model", state_dict)
        do_prefix = isinstance(
            model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        )
        state = {}
        for key, value in state_dict.items():
            if key.startswith("module.") and not do_prefix:
                key = key[7:]
            if not key.startswith("module.") and do_prefix:
                key = "module." + key
            state[key] = value

        incompatible = model.load_state_dict(state, strict=False)
        logger.info(
            "[production_models] ZoeDepth checkpoint loaded with strict=False; "
            f"missing_keys={len(getattr(incompatible, 'missing_keys', []))} "
            f"unexpected_keys={len(getattr(incompatible, 'unexpected_keys', []))}"
        )
        if getattr(incompatible, "unexpected_keys", None):
            logger.warning(
                "[production_models] Ignoring ZoeDepth unexpected checkpoint keys: "
                f"{list(incompatible.unexpected_keys)[:10]}"
            )
        return model

    model_io.load_state_dict = _load_state_dict_compat

    checkpoint_path = Path(checkpoint_path)
    try:
        if checkpoint_path.exists():
            conf = get_config("zoedepth", "infer", pretrained_resource=f"local::{checkpoint_path}")
        else:
            conf = get_config("zoedepth", "infer")
            logger.warning(f"[production_models] ZoeDepth checkpoint not found at {checkpoint_path}; falling back to default config")
        model = build_model(conf).to(device).eval()
    finally:
        model_io.load_state_dict = original_load_state_dict

    _PRODUCTION_MODEL_CACHE[cache_key] = model
    return model


def run_zoedepth(model: Any, pil_image: Image.Image) -> np.ndarray:
    depth = model.infer_pil(pil_image, output_type="numpy")
    return depth.astype(np.float32)
