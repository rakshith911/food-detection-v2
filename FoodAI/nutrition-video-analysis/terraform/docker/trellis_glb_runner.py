#!/usr/bin/env python3
"""
Minimal TRELLIS GPU runner for Option A bring-up.

Goal:
- run on a dedicated GPU task
- load one or more dish images
- generate GLB outputs
- keep this separate from the main nutrition worker
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import cv2
import imageio
import torch
from PIL import Image


DEFAULT_TRELLIS_SRC_DIR = Path(os.environ.get("TRELLIS_SRC_DIR", "/opt/TRELLIS.2"))
DEFAULT_HF_MODEL_ID = os.environ.get("TRELLIS_MODEL_ID", "microsoft/TRELLIS.2-4B")
DEFAULT_ENV_MAP = os.environ.get(
    "TRELLIS_ENV_MAP",
    str(DEFAULT_TRELLIS_SRC_DIR / "assets" / "hdri" / "forest.exr"),
)
DEFAULT_PREVIEW_SECONDS = float(os.environ.get("TRELLIS_PREVIEW_SECONDS", "8"))
DEFAULT_PREVIEW_FPS = int(os.environ.get("TRELLIS_PREVIEW_FPS", "15"))


def _bootstrap_trellis_imports(trellis_src_dir: Path) -> None:
    if not trellis_src_dir.exists():
        raise FileNotFoundError(
            f"TRELLIS source directory not found: {trellis_src_dir}. "
            "Set TRELLIS_SRC_DIR to the installed TRELLIS.2 location."
        )
    sys.path.insert(0, str(trellis_src_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GLB files from dish images with TRELLIS.")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="One or more input image paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="/app/data/trellis_outputs",
        help="Directory to write GLB and optional MP4 outputs.",
    )
    parser.add_argument(
        "--trellis-src-dir",
        default=str(DEFAULT_TRELLIS_SRC_DIR),
        help="Path to the TRELLIS.2 source tree.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_HF_MODEL_ID,
        help="Hugging Face model id for TRELLIS.",
    )
    parser.add_argument(
        "--envmap",
        default=DEFAULT_ENV_MAP,
        help="Environment map EXR used for preview video rendering.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip MP4 preview rendering and only export GLB.",
    )
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=DEFAULT_PREVIEW_SECONDS,
        help="Duration of the preview MP4 in seconds.",
    )
    parser.add_argument(
        "--preview-fps",
        type=int,
        default=DEFAULT_PREVIEW_FPS,
        help="FPS for the preview MP4.",
    )
    return parser.parse_args()


def _load_envmap(envmap_path: Path):
    from trellis2.renderers import EnvMap

    if not envmap_path.exists():
        raise FileNotFoundError(f"Environment map not found: {envmap_path}")
    env_image = cv2.imread(str(envmap_path), cv2.IMREAD_UNCHANGED)
    if env_image is None:
        raise RuntimeError(f"Failed to read envmap: {envmap_path}")
    rgb = cv2.cvtColor(env_image, cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(rgb, dtype=torch.float32, device="cuda"))


def _load_pipeline(model_id: str):
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    print(f"Loading TRELLIS pipeline from {model_id} ...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()
    return pipeline


def _export_glb(mesh, glb_path: Path) -> None:
    import o_voxel

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1_000_000,
        texture_size=4096,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    # Use baseline GLB export for bring-up. WebP texture export currently
    # trips a Pillow/WebP plugin mismatch inside the container.
    glb.export(str(glb_path))


def _render_video(mesh, mp4_path: Path, preview_seconds: float, preview_fps: int) -> None:
    import numpy as np
    from trellis2.renderers import MeshRenderer
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    num_frames = max(1, int(round(preview_seconds * preview_fps)))

    # Bypass render_utils.get_renderer which picks PbrMeshRenderer for
    # MeshWithPbrMaterial (requires nvdiffrec_render, not compiled in this image).
    # MeshRenderer only needs nvdiffrast which is installed.
    renderer = MeshRenderer()
    renderer.rendering_options.resolution = 512
    renderer.rendering_options.near = 1
    renderer.rendering_options.far = 100
    renderer.rendering_options.ssaa = 2

    yaws = (-torch.linspace(0, 2 * 3.1415, num_frames) + np.pi / 2).tolist()
    pitchs = (0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))).tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, 2, 40)

    frames = []
    for extr, intr in zip(extrinsics, intrinsics):
        # "attr" for MeshWithPbrMaterial splits into base_color/metallic/roughness/alpha keys
        res = renderer.render(mesh, extr, intr, return_types=["attr"])
        color = res["base_color"]  # [3, H, W], actual texture colors in [0, 1]
        alpha = res["alpha"]       # [1, H, W]
        frame = np.clip(color.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        frame[alpha.squeeze(0).detach().cpu().numpy() < 0.5] = 0
        frames.append(frame)

    imageio.mimsave(str(mp4_path), frames, fps=preview_fps)


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    trellis_src_dir = Path(args.trellis_src_dir)
    envmap_path = Path(args.envmap)
    image_paths = [Path(p).expanduser().resolve() for p in args.images]

    _bootstrap_trellis_imports(trellis_src_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This runner expects a GPU task.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"TRELLIS source: {trellis_src_dir}")
    print(f"Model: {args.model_id}")
    print(f"Images: {len(image_paths)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _load_pipeline(args.model_id)

    manifest: list[dict[str, object]] = []
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        stem = image_path.stem
        glb_path = output_dir / f"{stem}.glb"
        mp4_path = output_dir / f"{stem}.mp4"
        print(f"\n=== Processing {image_path.name} ===")
        image = Image.open(image_path).convert("RGB")

        started = time.time()
        mesh = pipeline.run(image)[0]
        mesh.simplify(16_777_216)
        duration_s = time.time() - started
        print(f"Generation took {duration_s:.1f}s")

        _export_glb(mesh, glb_path)
        print(f"Saved {glb_path}")

        if not args.skip_video:
            _render_video(mesh, mp4_path, args.preview_seconds, args.preview_fps)
            print(f"Saved {mp4_path}")

        manifest.append(
            {
                "input_image": str(image_path),
                "glb_path": str(glb_path),
                "preview_mp4_path": str(mp4_path) if not args.skip_video else None,
                "duration_seconds": round(duration_s, 2),
                "voxel_count": int(len(mesh.coords)),
                "voxel_size": float(mesh.voxel_size),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path}")
    print("TRELLIS GLB run finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
