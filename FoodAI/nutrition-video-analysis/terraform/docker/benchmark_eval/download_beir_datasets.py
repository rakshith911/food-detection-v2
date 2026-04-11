#!/usr/bin/env python3
"""
Download a practical subset of public BEIR datasets for local benchmarking.

Defaults to a few small/medium public datasets that are reasonable to run
locally first:
  - scifact
  - nfcorpus
  - arguana
  - fiqa
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_DATASETS = ["scifact", "nfcorpus", "arguana", "fiqa"]
DATASET_MD5 = {
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
    "arguana": "8ad3e3c2a5867cdced806d6503f29b99",
    "fiqa": "17918ed23cd04fb15047f73e6c3bd9d9",
}
BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=DEFAULT_DATASETS, help="Dataset names to download")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets",
        help="Directory to place extracted datasets",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if dataset folder already exists")
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        dataset_dir = out_dir / dataset
        zip_path = out_dir / f"{dataset}.zip"
        url = f"{BASE_URL}/{dataset}.zip"

        if dataset_dir.exists() and not args.force:
            print(f"Skip {dataset}: already extracted at {dataset_dir}")
            continue

        if zip_path.exists() and args.force:
            zip_path.unlink()
        if dataset_dir.exists() and args.force:
            shutil.rmtree(dataset_dir)

        print(f"Downloading {dataset} from {url}")
        download_file(url, zip_path)

        expected_md5 = DATASET_MD5.get(dataset)
        if expected_md5:
            actual_md5 = md5sum(zip_path)
            if actual_md5 != expected_md5:
                print(
                    f"MD5 mismatch for {dataset}: expected {expected_md5}, got {actual_md5}",
                    file=sys.stderr,
                )
                return 1
            print(f"Verified md5 for {dataset}")

        print(f"Extracting {dataset} to {dataset_dir}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(out_dir)
        print(f"Ready: {dataset_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
