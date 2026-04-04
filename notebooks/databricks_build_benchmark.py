# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Phase 2: Build Behavioral Benchmark
# MAGIC Runs feature engineering pipeline on IEEE-CIS and Amazon FDB.
# MAGIC Produces ground-truth entity behavioral feature tables.
# MAGIC
# MAGIC **Run after:** `databricks_setup.py`
# MAGIC **Runtime:** ~10 min IEEE-CIS, ~5 min Amazon FDB

# COMMAND ----------

import sys, os
sys.path.insert(0, "/Workspace/Repos/bhavana37/synthetic-fraud-benchmark")  # adjust to your repo path

from pathlib import Path
from features.behavioral_feature_engineering import build_benchmark, DATASET_CONFIG

DBFS_ROOT = "/dbfs/FileStore/synthetic_fraud_benchmark"

# COMMAND ----------

# MAGIC %md
# MAGIC ## IEEE-CIS

# COMMAND ----------

ieee_entity_df, ieee_txn_df = build_benchmark(
    cfg          = DATASET_CONFIG["ieee_cis"],
    data_dir     = Path(f"{DBFS_ROOT}/data/ieee_cis"),
    out_dir      = Path(f"{DBFS_ROOT}/results/benchmark"),
    dataset_name = "ieee_cis",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Amazon FDB

# COMMAND ----------

fdb_entity_df, fdb_txn_df = build_benchmark(
    cfg          = DATASET_CONFIG["amazon_fdb"],
    data_dir     = Path(f"{DBFS_ROOT}/data/amazon_fdb"),
    out_dir      = Path(f"{DBFS_ROOT}/results/benchmark"),
    dataset_name = "amazon_fdb",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify: are fraud/legit distributions actually separable?
# MAGIC This is the go/no-go check before generator training.

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for name, df in [("IEEE-CIS", ieee_entity_df), ("Amazon FDB", fdb_entity_df)]:
    fraud = df[df["label"] == 1]
    legit = df[df["label"] == 0]
    print(f"\n{'='*55}")
    print(f"{name}  |  fraud={len(fraud):,}  legit={len(legit):,}")
    print(f"{'='*55}")

    checks = [
        ("iet_median",           "Median IET (s)"),
        ("iet_autocorr_lag1",    "IET autocorrelation"),
        ("active_lifetime_days", "Active lifetime (days)"),
        ("burst_count_3600s",    "Burst count (1hr delta)"),
        ("burst_len_max_3600s",  "Max burst length"),
        ("ever_r1",              "R1 trigger rate (>3/hr)"),
        ("ever_r3",              "R3 trigger rate (>$1k/day)"),
        ("n_rules_triggered",    "Avg rules triggered"),
    ]
    print(f"  {'Feature':33s}  {'Fraud':>9s}  {'Legit':>9s}  {'F/L':>6s}")
    print(f"  {'-'*33}  {'-'*9}  {'-'*9}  {'-'*6}")
    for col, lbl in checks:
        if col not in df.columns:
            print(f"  {lbl:33s}  {'N/A':>9s}")
            continue
        fv = fraud[col].mean()
        lv = legit[col].mean()
        ratio = fv / lv if abs(lv) > 1e-9 else float("nan")
        flag  = "✓" if abs(ratio - 1.0) > 0.1 else "~"  # flag if >10% difference
        print(f"  {lbl:33s}  {fv:>9.3f}  {lv:>9.3f}  {ratio:>5.2f}x {flag}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribution plots — key P1 and P2 signals

# COMMAND ----------

import seaborn as sns

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

plot_features = [
    ("iet_median",           "Median IET (s)",          True),
    ("active_lifetime_days", "Active lifetime (days)",  False),
    ("burst_count_3600s",    "Burst count (1hr)",       False),
]

for row_i, (name, df) in enumerate([("IEEE-CIS", ieee_entity_df), ("Amazon FDB", fdb_entity_df)]):
    for col_i, (feat, label, log_scale) in enumerate(plot_features):
        ax = axes[row_i][col_i]
        if feat not in df.columns:
            ax.set_visible(False)
            continue
        fraud = df[df["label"]==1][feat].dropna()
        legit = df[df["label"]==0][feat].dropna()
        cap   = df[feat].quantile(0.99)
        fraud = fraud.clip(upper=cap)
        legit = legit.clip(upper=cap)
        ax.hist(legit.sample(min(2000, len(legit)), random_state=42),
                bins=40, alpha=0.6, color="steelblue", label="Non-fraud", density=True)
        ax.hist(fraud.sample(min(2000, len(fraud)), random_state=42),
                bins=40, alpha=0.6, color="tomato",    label="Fraud",     density=True)
        if log_scale and (fraud > 0).all():
            ax.set_xscale("log")
        ax.set_title(f"{name} — {label}")
        ax.set_xlabel(label)
        ax.legend(fontsize=8)

plt.suptitle("Benchmark Ground Truth: Fraud vs. Non-Fraud Distributions", y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig(f"{DBFS_ROOT}/results/benchmark/benchmark_signal_check.png", dpi=120, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Go / No-Go decision
# MAGIC
# MAGIC **Proceed to generator training if:**
# MAGIC - At least 4 of the 8 features above show F/L ratio > 1.2x or < 0.8x
# MAGIC - The distribution plots show visible separation between fraud and non-fraud
# MAGIC
# MAGIC **If no separation:** revisit entity key strategy (try composite key for IEEE-CIS)

# COMMAND ----------

print("""
Benchmark tables saved:
  entity_behavioral_ieee_cis.csv
  entity_behavioral_amazon_fdb.csv
  txn_behavioral_ieee_cis.csv
  txn_behavioral_amazon_fdb.csv

If signal check passed → proceed to databricks_train_sdv.py
""")
