# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Phase 3: Behavioral Fidelity Evaluation
# MAGIC
# MAGIC Evaluates each generator's synthetic data against real data using P1–P4.
# MAGIC
# MAGIC **Run after:** `databricks_train_sdv.py` and `databricks_train_tabularargn.py`
# MAGIC
# MAGIC **Scope:**
# MAGIC - P1 (IET) + P2 (Burst) + P4 (Velocity) → IEEE-CIS only
# MAGIC - P3 (Graph motifs)                      → Amazon FDB only
# MAGIC
# MAGIC **Output:** `results/evaluation/behavioral_results.csv`
# MAGIC — one row per (generator × dataset × pattern), plus composite degradation ratio.

# COMMAND ----------

import sys, os, json, warnings
warnings.filterwarnings("ignore")

REPO = "/Workspace/Repos/bhavana37/synthetic-fraud-benchmark"
sys.path.insert(0, REPO)

import pandas as pd
import numpy as np
from pathlib import Path
from evaluation.behavioral_fidelity import BehavioralFidelityEvaluator

DBFS_ROOT  = "/dbfs/FileStore/synthetic_fraud_benchmark"
DATA_DIR   = f"{DBFS_ROOT}/data"
SYN_DIR    = f"{DBFS_ROOT}/synthetic"
BENCH_DIR  = f"{DBFS_ROOT}/results/benchmark"
EVAL_DIR   = f"{DBFS_ROOT}/results/evaluation"

os.makedirs(EVAL_DIR, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load baseline scores
# MAGIC
# MAGIC These were computed from real-vs-real random 50/50 splits.
# MAGIC They define the noise floor — a perfect generator should score ≈ 1.0 on each.

# COMMAND ----------

with open(f"{BENCH_DIR}/baseline_scores_ieee_cis.json") as f:
    baseline_ieee = json.load(f)

with open(f"{BENCH_DIR}/baseline_scores_amazon_fdb.json") as f:
    baseline_fdb = json.load(f)

print("IEEE-CIS baseline scores:")
for k, v in baseline_ieee.items():
    print(f"  {k}: {v:.4f}")

print("\nAmazon FDB baseline scores:")
for k, v in baseline_fdb.items():
    print(f"  {k}: {v:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load real data

# COMMAND ----------

print("Loading IEEE-CIS real data...")
txn = pd.read_csv(f"{DATA_DIR}/ieee_cis/train_transaction.csv",
                  usecols=["TransactionID","TransactionDT","TransactionAmt","isFraud","card1","card4"])
ieee_real = txn.rename(columns={
    "TransactionDT": "timestamp", "TransactionAmt": "amount",
    "isFraud": "label", "card1": "entity", "card4": "card"
})
ieee_real["entity"] = ieee_real["entity"].astype(str)
print(f"  {len(ieee_real):,} rows  |  fraud={ieee_real['label'].mean():.3f}")

print("\nLoading Amazon FDB real data...")
fdb_raw = pd.read_csv(f"{DATA_DIR}/amazon_fdb/Fraud_Data.csv")
fdb_real = fdb_raw.rename(columns={"user_id":"entity","class":"label","purchase_value":"amount"})
fdb_real["entity"]    = fdb_real["entity"].astype(str)
fdb_real["label"]     = fdb_real["label"].astype(int)
fdb_real["timestamp"] = pd.to_datetime(fdb_real["purchase_time"]).astype(np.int64) // 1_000_000_000
print(f"  {len(fdb_real):,} rows  |  fraud={fdb_real['label'].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluators

# COMMAND ----------

ev_ieee = BehavioralFidelityEvaluator(
    entity_col  = "entity",
    time_col    = "timestamp",
    label_col   = "label",
    amount_col  = "amount",
    card_col    = "card",
    burst_deltas= [300, 3600, 21600],
)

ev_fdb = BehavioralFidelityEvaluator(
    entity_col  = "entity",
    time_col    = "timestamp",
    label_col   = "label",
    amount_col  = "amount",
    attr_cols   = ["device_id", "ip_address"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mock-baseline objects (needed to carry raw_scores into evaluate_all)

# COMMAND ----------

from evaluation.behavioral_fidelity import BehavioralFidelityReport

def make_baseline_report(raw_scores: dict, dataset_name: str) -> BehavioralFidelityReport:
    r = BehavioralFidelityReport(generator_name="BASELINE", dataset_name=dataset_name)
    r.raw_scores = raw_scores
    return r

base_ieee = make_baseline_report(baseline_ieee, "ieee_cis")
base_fdb  = make_baseline_report(baseline_fdb,  "amazon_fdb")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate all generators
# MAGIC
# MAGIC Expected synthetic files under `/dbfs/FileStore/synthetic_fraud_benchmark/synthetic/`:
# MAGIC ```
# MAGIC ctgan_ieee_cis_syn.csv
# MAGIC tvae_ieee_cis_syn.csv
# MAGIC gaussiancopula_ieee_cis_syn.csv
# MAGIC tabularargn_ieee_cis_syn.csv
# MAGIC ctgan_amazon_fdb_syn.csv
# MAGIC tvae_amazon_fdb_syn.csv
# MAGIC gaussiancopula_amazon_fdb_syn.csv
# MAGIC tabularargn_amazon_fdb_syn.csv
# MAGIC ```

# COMMAND ----------

GENERATORS = ["ctgan", "tvae", "gaussiancopula", "tabularargn"]

all_reports = []

def load_syn(generator: str, dataset: str) -> pd.DataFrame:
    path = f"{SYN_DIR}/{generator}_{dataset}_syn.csv"
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    # Normalise column names to match real data
    rename = {}
    if "TransactionDT"  in df.columns: rename["TransactionDT"]  = "timestamp"
    if "TransactionAmt" in df.columns: rename["TransactionAmt"] = "amount"
    if "isFraud"        in df.columns: rename["isFraud"]        = "label"
    if "card1"          in df.columns: rename["card1"]          = "entity"
    if "card4"          in df.columns: rename["card4"]          = "card"
    if "user_id"        in df.columns: rename["user_id"]        = "entity"
    if "class"          in df.columns: rename["class"]          = "label"
    if "purchase_value" in df.columns: rename["purchase_value"] = "amount"
    df = df.rename(columns=rename)
    if "entity" in df.columns:
        df["entity"] = df["entity"].astype(str)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    if "purchase_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["purchase_time"]).astype(np.int64) // 1_000_000_000
    return df

# ── Entity assignment for IEEE-CIS synthetic data ─────────────────────────────
# card1 (entity key) was excluded from generator training — 13K unique values
# causes CTGAN tensor blowup. But P1/P2/P4 require entity groupings to compute
# IET sequences and burst density. Solution: assign fake entity IDs to synthetic
# rows by sampling from the real entity-size distribution.
#
# Why this is valid:
#   - It preserves the real data's entity-transaction density (~36 txn/entity)
#   - Row-independent generators (CTGAN, TVAE, Copula) produce rows with no
#     inter-row dependencies, so any entity assignment is as valid as any other
#   - The expected finding: within-entity temporal sequences are incoherent
#     (timestamps weren't modeled jointly), so P1/P2 will show degradation
#   - TabularARGN (autoregressive) may preserve temporal dependencies if it
#     models sequences — assignment lets us test this on equal footing
#
# For Amazon FDB: user_id is already 1-txn-per-user, so no assignment needed.

def assign_synthetic_entities(syn_df: pd.DataFrame, real_df: pd.DataFrame,
                               entity_col: str = "entity",
                               label_col:  str = "label",
                               seed: int = 42) -> pd.DataFrame:
    """
    Assign entity IDs to synthetic rows using the real entity size distribution.
    Each class (fraud=1, legit=0) is assigned separately to preserve class-level
    entity structure — fraud entities in real data are typically smaller
    (fewer transactions) than legit entities.

    Returns syn_df with a new 'entity' column.
    """
    rng = np.random.default_rng(seed)
    syn = syn_df.copy()
    syn[entity_col] = "unassigned"

    for cls in [0, 1]:
        # Entity sizes for this class in real data
        real_cls      = real_df[real_df[label_col] == cls]
        entity_sizes  = real_cls.groupby(entity_col).size().values

        # Synthetic rows for this class
        syn_idx = syn[syn[label_col] == cls].index
        n_syn   = len(syn_idx)
        if n_syn == 0:
            continue

        # Build entity assignments: fill bins until we've covered n_syn rows
        assignments = []
        eid = 0
        while len(assignments) < n_syn:
            size = int(rng.choice(entity_sizes))
            assignments.extend([f"syn_{cls}_{eid}"] * size)
            eid += 1
        assignments = assignments[:n_syn]        # trim to exact count
        rng.shuffle(assignments)                  # randomise row-entity pairing

        syn.loc[syn_idx, entity_col] = assignments

    return syn


# Compute entity assignment once from real IEEE-CIS training data
print("Pre-computing entity size distribution from real IEEE-CIS data...")
entity_sizes_ieee = ieee_real.groupby("entity").size()
print(f"  {len(entity_sizes_ieee):,} unique entities  |  "
      f"median size={entity_sizes_ieee.median():.0f}  max={entity_sizes_ieee.max()}")

# ── IEEE-CIS: P1 + P2 + P4 ───────────────────────────────────────────────────
print("=" * 60)
print("IEEE-CIS — P1, P2, P4")
print("=" * 60)

for gen in GENERATORS:
    syn = load_syn(gen, "ieee_cis")
    if syn is None:
        continue

    # Assign entity IDs if card1 was not generated (i.e., excluded from training)
    if "entity" not in syn.columns:
        print(f"  [{gen.upper()}] No entity col found — assigning from real entity distribution...")
        syn = assign_synthetic_entities(syn, ieee_real, entity_col="entity", label_col="label")
        print(f"  [{gen.upper()}] Entity assignment done: {syn['entity'].nunique():,} unique entities")

    print(f"\n  Generator: {gen.upper()}  ({len(syn):,} rows, fraud={syn['label'].mean():.3f})")
    report = ev_ieee.evaluate_all(
        ieee_real, syn,
        generator_name = gen.upper(),
        dataset_name   = "ieee_cis",
        skip           = ["p3"],
        baseline       = base_ieee,
    )
    print(report.summary())
    all_reports.append(report)

# ── Amazon FDB: P3 only ───────────────────────────────────────────────────────
# CTGAN and TVAE were trained WITHOUT device_id and ip_address (high-cardinality
# categoricals cause one-hot OOM). We assign random values from the real pool so
# P3 can run and produce a score — the score will be worst-case (random assignment
# destroys all graph structure), which is the correct finding.
# GaussianCopula was trained WITH device_id/ip_address (frequency encoding).

def assign_random_graph_cols(syn_df: pd.DataFrame, real_df: pd.DataFrame,
                              graph_cols: list, seed: int = 42) -> pd.DataFrame:
    """
    For each graph column missing from syn_df, randomly sample values from
    real_df's marginal distribution and assign them to synthetic rows.
    Random assignment intentionally destroys all cross-row sharing structure,
    producing worst-case P3 scores — this is the expected result for
    row-independent generators that couldn't model these columns.
    """
    rng = np.random.default_rng(seed)
    syn = syn_df.copy()
    for col in graph_cols:
        if col not in syn.columns:
            pool = real_df[col].dropna().values
            syn[col] = rng.choice(pool, size=len(syn), replace=True)
            print(f"    Assigned random {col} from real pool ({len(pool):,} unique values)")
    return syn

print("\n" + "=" * 60)
print("Amazon FDB — P3 (Graph Motifs)")
print("=" * 60)

GRAPH_COLS_FDB = ["device_id", "ip_address"]

for gen in GENERATORS:
    syn = load_syn(gen, "amazon_fdb")
    if syn is None:
        continue

    # ── Entity column ──────────────────────────────────────────────────────────
    # user_id was dropped from all Amazon FDB training sets (high-cardinality).
    # Amazon FDB is 1-transaction-per-user, so every synthetic row is its own
    # unique entity. Assign unique per-row IDs so the graph builder can find
    # the 'entity' column.
    if "entity" not in syn.columns:
        syn["entity"] = [f"syn_u{i}" for i in range(len(syn))]
        print(f"  [{gen.upper()}] No entity col — assigning unique per-row entity IDs "
              f"(1-txn-per-user structure)")

    # ── Graph attribute columns ────────────────────────────────────────────────
    missing = [c for c in GRAPH_COLS_FDB if c not in syn.columns]
    if missing:
        print(f"  [{gen.upper()}] Graph cols {missing} absent (dropped during training) "
              f"— assigning random values for P3 evaluation")
        syn = assign_random_graph_cols(syn, fdb_real, GRAPH_COLS_FDB)

    print(f"\n  Generator: {gen.upper()}  ({len(syn):,} rows, fraud={syn['label'].mean():.3f}  "
          f"graph_cols={'random' if missing else 'model-generated'})")
    report = ev_fdb.evaluate_all(
        fdb_real, syn,
        generator_name = gen.upper(),
        dataset_name   = "amazon_fdb",
        skip           = ["p1","p2","p4"],
        baseline       = base_fdb,
    )
    print(report.summary())
    all_reports.append(report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compile results table

# COMMAND ----------

rows = []
for r in all_reports:
    base_row = {"generator": r.generator_name, "dataset": r.dataset_name,
                "composite": r.composite_score}
    base_row.update(r.raw_scores)
    rows.append(base_row)

results_df = pd.DataFrame(rows)
out_path   = f"{EVAL_DIR}/behavioral_results.csv"
results_df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(results_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Table — Degradation Ratios
# MAGIC
# MAGIC Read: each cell = generator score / baseline score.
# MAGIC - 1.0  = matches real-data noise floor (perfect generator)
# MAGIC - 5.0  = 5× worse than sampling real data
# MAGIC - 10.0 = catastrophic failure

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# Pivot for display
pivot_cols = [c for c in results_df.columns if c.startswith("p") or c == "composite"]
print("\n=== DEGRADATION RATIOS (generator / baseline) ===")
print(results_df[["generator","dataset"] + pivot_cols].to_string(index=False))
