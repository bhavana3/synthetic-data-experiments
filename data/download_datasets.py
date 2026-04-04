"""
Download all datasets needed for the synthetic fraud benchmark.

Usage:
    python data/download_datasets.py --dataset ieee_cis
    python data/download_datasets.py --dataset amazon_fdb
    python data/download_datasets.py --dataset all

Requires: kaggle.json configured at ~/.kaggle/kaggle.json
          (chmod 600 ~/.kaggle/kaggle.json)
"""

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"


def download_ieee_cis():
    out_dir = DATA_ROOT / "ieee_cis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading IEEE-CIS Fraud Detection dataset...")
    print("NOTE: You must have accepted the competition rules on Kaggle first.")
    print("  → https://www.kaggle.com/competitions/ieee-fraud-detection/rules\n")

    result = subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", "ieee-fraud-detection",
            "-p", str(out_dir),
        ],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print("ERROR:", result.stderr)
        print("\nIf you see '403 Forbidden', go accept the competition rules:")
        print("  https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
        return False

    print(result.stdout)

    # Unzip
    zip_path = out_dir / "ieee-fraud-detection.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        zip_path.unlink()
        print(f"Extracted to {out_dir}")

    # List what we got
    files = list(out_dir.glob("*"))
    print("\nFiles available:")
    for f in sorted(files):
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"  {f.name:40s}  {size_mb:.1f} MB")

    return True


def download_amazon_fdb():
    """
    Amazon Fraud Detector Benchmark (FDB).
    Public S3 bucket — no Kaggle auth needed.

    Dataset paper: https://arxiv.org/abs/2208.13797
    Data source:   https://github.com/amazon-science/fraud-dataset-benchmark
    """
    out_dir = DATA_ROOT / "amazon_fdb"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Amazon FDB dataset (fraudecom split)...")

    # The public S3 URLs for the FDB benchmark datasets
    # fraudecom: e-commerce fraud with user/device/IP signals
    base_url = "https://fraud-dataset-benchmark.s3.amazonaws.com"
    files_to_download = [
        "fraudecom/train.csv",
        "fraudecom/test.csv",
    ]

    success = True
    for fname in files_to_download:
        url = f"{base_url}/{fname}"
        local_path = out_dir / Path(fname).name
        if local_path.exists():
            print(f"  Already exists: {local_path.name}")
            continue

        print(f"  Downloading {fname} ...")
        result = subprocess.run(
            ["curl", "-L", "-o", str(local_path), url],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ERROR downloading {fname}: {result.stderr}")
            success = False
        else:
            size_mb = local_path.stat().st_size / (1024 ** 2)
            print(f"  OK: {local_path.name} ({size_mb:.1f} MB)")

    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["ieee_cis", "amazon_fdb", "all"],
        default="all", help="Which dataset to download"
    )
    args = parser.parse_args()

    if args.dataset in ("ieee_cis", "all"):
        download_ieee_cis()

    if args.dataset in ("amazon_fdb", "all"):
        download_amazon_fdb()


if __name__ == "__main__":
    main()
