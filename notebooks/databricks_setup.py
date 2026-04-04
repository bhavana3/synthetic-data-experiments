# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Synthetic Fraud Benchmark — Databricks Setup
# MAGIC Run this once before any other notebook. Installs packages, configures paths, downloads data.

# COMMAND ----------

# Install packages
%pip install sdv==1.14.0 xgboost==2.0.3 shap==0.45.1 POT==0.9.3 networkx==3.3 seaborn==0.13.2 kaggle==1.6.17 tqdm==4.66.4 realtabformer

# COMMAND ----------

# Install TabDDPM from source
%pip install git+https://github.com/rotot0/tab-ddpm.git

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Kaggle credentials

# COMMAND ----------

import os, json, pathlib

# ── Paste your values here ──────────────────────────────────────
KAGGLE_USERNAME = "bhavana37"
KAGGLE_TOKEN    = "KGAT_your_token_here"   # ← replace with your actual token
# ────────────────────────────────────────────────────────────────

# Write kaggle.json to home dir — this is what the CLI actually reads
kaggle_dir  = pathlib.Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)
kaggle_json = kaggle_dir / "kaggle.json"
kaggle_json.write_text(json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_TOKEN}))
kaggle_json.chmod(0o600)

# Also set env vars as backup
os.environ["KAGGLE_USERNAME"]  = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"]       = KAGGLE_TOKEN

print(f"kaggle.json written to {kaggle_json}")
print("Verifying...")

import subprocess
result = subprocess.run(["kaggle", "competitions", "list", "--page", "1"],
                        capture_output=True, text=True)
if "401" in result.stderr or "Unauthorized" in result.stderr:
    print("ERROR: Still unauthorized — check your token value")
else:
    print("✓ Kaggle auth working")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure DBFS paths

# COMMAND ----------

DBFS_ROOT  = "/dbfs/FileStore/synthetic_fraud_benchmark"
DATA_DIR   = f"{DBFS_ROOT}/data"
RESULTS_DIR = f"{DBFS_ROOT}/results"
SYN_DIR    = f"{RESULTS_DIR}/synthetic"
EVAL_DIR   = f"{RESULTS_DIR}/behavioral_fidelity"

for d in [DATA_DIR, RESULTS_DIR, SYN_DIR, EVAL_DIR,
          f"{DATA_DIR}/ieee_cis", f"{DATA_DIR}/amazon_fdb"]:
    os.makedirs(d, exist_ok=True)

print("Paths ready:")
for name, path in [("Data", DATA_DIR), ("Results", RESULTS_DIR), ("Synthetic", SYN_DIR)]:
    print(f"  {name:12s}: {path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download IEEE-CIS dataset

# COMMAND ----------

import subprocess, zipfile

ieee_dir = f"{DATA_DIR}/ieee_cis"
if not os.path.exists(f"{ieee_dir}/train_transaction.csv"):
    print("Downloading IEEE-CIS (~450MB)...")
    result = subprocess.run(
        ["kaggle", "competitions", "download",
         "-c", "ieee-fraud-detection", "-p", ieee_dir],
        capture_output=True, text=True
    )
    print(result.stdout or result.stderr)

    zip_path = f"{ieee_dir}/ieee-fraud-detection.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ieee_dir)
        os.remove(zip_path)
        print("Extracted.")
else:
    print("Already downloaded.")

print("\nFiles:")
for f in sorted(os.listdir(ieee_dir)):
    size = os.path.getsize(f"{ieee_dir}/{f}") / 1e6
    print(f"  {f:40s}  {size:.0f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Amazon FDB dataset
# MAGIC Source: Kaggle dataset vbinh002/fraud-ecommerce
# MAGIC File: Fraud_Data.csv  (~150K rows, columns: user_id, purchase_time, purchase_value,
# MAGIC        device_id, source, browser, sex, age, ip_address, class)

# COMMAND ----------

fdb_dir = f"{DATA_DIR}/amazon_fdb"
os.makedirs(fdb_dir, exist_ok=True)

if not os.path.exists(f"{fdb_dir}/Fraud_Data.csv"):
    print("Downloading Amazon FDB from Kaggle (vbinh002/fraud-ecommerce)...")
    result = subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", "vbinh002/fraud-ecommerce",
         "-p", fdb_dir, "--unzip"],
        capture_output=True, text=True
    )
    print(result.stdout or result.stderr)
else:
    print("  Already exists: Fraud_Data.csv")

print("\nFiles in amazon_fdb dir:")
for f in sorted(os.listdir(fdb_dir)):
    size = os.path.getsize(f"{fdb_dir}/{f}") / 1e6
    print(f"  {f:40s}  {size:.1f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add project code to Python path

# COMMAND ----------

import sys

# After uploading project zip to DBFS or Repos, set path here:
# Option A — Databricks Repos:
# sys.path.insert(0, "/Workspace/Repos/bhavana37/synthetic-fraud-benchmark")

# Option B — Uploaded zip extracted to DBFS:
# sys.path.insert(0, "/dbfs/FileStore/synthetic_fraud_benchmark/code")

# Verify
# from evaluation.behavioral_fidelity import BehavioralFidelityEvaluator
# print("✓ Import OK")

# COMMAND ----------

print("""
Setup complete. Next: open databricks_eda_ieee_cis.py
""")
