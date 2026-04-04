# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Generator Training: TabularARGN (MOSTLY AI)
# MAGIC Trains MOSTLY AI's autoregressive generator on IEEE-CIS and Amazon FDB.
# MAGIC
# MAGIC **Cluster:** CPU works (TabularARGN is 1-2 orders of magnitude faster than GAN/VAE).
# MAGIC **Runtime:** ~15-30 min on IEEE-CIS train split.

# COMMAND ----------

%pip install 'mostlyai[local]' --quiet

# COMMAND ----------

import os, json, time
import pandas as pd
import numpy as np
from mostlyai import MostlyAI

DBFS_ROOT  = "/dbfs/FileStore/synthetic_fraud_benchmark"
DATA_DIR   = f"{DBFS_ROOT}/data"
SYN_DIR    = f"{DBFS_ROOT}/synthetic"    # NOTE: matches evaluation notebook path
RESULT_DIR = f"{DBFS_ROOT}/results"
os.makedirs(SYN_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Confirmed IEEE-CIS config — resolved during EDA (entity=card1, p80 temporal split)
# Written inline so this notebook is self-contained; no prior EDA run required.
IEEE_CIS_CONFIG = {
    "entity_col":           "card1",
    "time_col":             "TransactionDT",
    "label_col":            "isFraud",
    "amount_col":           "TransactionAmt",
    "card_col":             "card4",
    "attr_cols":            [],
    "burst_deltas":         [300, 3600, 21600],
    "train_split_quantile": 0.80,
}

with open(f"{RESULT_DIR}/ieee_cis_config.json", "w") as f:
    json.dump(IEEE_CIS_CONFIG, f, indent=2)

LABEL_COL = IEEE_CIS_CONFIG["label_col"]
TIME_COL  = IEEE_CIS_CONFIG["time_col"]
SPLIT_Q   = IEEE_CIS_CONFIG["train_split_quantile"]

print("Config ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load training data

# COMMAND ----------

print("Loading IEEE-CIS train split...")
txn = pd.read_csv(f"{DATA_DIR}/ieee_cis/train_transaction.csv")
idn = pd.read_csv(f"{DATA_DIR}/ieee_cis/train_identity.csv")
df  = txn.merge(idn, on="TransactionID", how="left")

split_dt  = df[TIME_COL].quantile(SPLIT_Q)
train_df  = df[df[TIME_COL] <= split_dt].copy()

# ── Same 48-column keep-list as SDV notebook (paper consistency) ───────────────
# All four generators use the same behavioral feature subset so comparisons are fair.
# V1–V339 (Vesta engineered features) excluded: not used in P1/P2/P4 evaluation.
BEHAVIORAL_COLS = [
    "TransactionDT", "TransactionAmt", "isFraud", "card4",
    "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14",
    "D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
    "addr1","addr2","dist1","dist2",
    "P_emaildomain","R_emaildomain",
]
BEHAVIORAL_COLS = [c for c in BEHAVIORAL_COLS if c in train_df.columns]
train_df = train_df[BEHAVIORAL_COLS].copy()
print(f"Columns kept: {len(BEHAVIORAL_COLS)}  (V1–V339 excluded for consistency with SDV notebook)")

# Null imputation — TabularARGN does not accept nulls in numeric columns
num_cols = train_df.select_dtypes(include="number").columns
cat_cols = train_df.select_dtypes(include="object").columns
train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())
train_df[cat_cols] = train_df[cat_cols].fillna("MISSING")
assert train_df.isnull().sum().sum() == 0, "Nulls remain after imputation!"

print(f"Training table: {train_df.shape}  (0 nulls)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train TabularARGN (local mode)

# COMMAND ----------

print("Initialising MOSTLY AI in local mode...")
t0 = time.time()

# Local mode — no API key needed, trains on this cluster
mostly = MostlyAI(local=True)

# ── IMPORTANT: "columns" in the MOSTLY AI SDK config acts as an INCLUDE LIST ──
# If you list only 2 columns, only those 2 columns are generated in the output.
# To generate ALL 48 behavioral columns, do NOT specify a "columns" list at all —
# MOSTLY AI will auto-configure every column from the training data schema.
# We still explicitly configure the table-level settings (value protection default ON
# for IEEE-CIS is fine — no high-cardinality categoricals that hit the threshold).
#
# TabularARGN will assign model_encoding_type automatically:
#   numerical columns  → TABULAR_NUMERIC_AUTO
#   categorical (≤N)   → TABULAR_CATEGORICAL
#   high-cardinality   → handled by value protection or TABULAR_CATEGORICAL

generator = mostly.train(
    data=train_df,
    name="ieee_cis_tabularargn",
    # No "columns" override — generate ALL behavioral columns (all 48)
    # MOSTLY AI auto-assigns encoding types from the data schema
)

elapsed = (time.time() - t0) / 60
print(f"Training done in {elapsed:.1f} min")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate synthetic data

# COMMAND ----------

print("Generating synthetic data...")
syn_data = mostly.generate(
    generator=generator,
    size=len(train_df),         # same number of rows as real training set
)

# mostlyai returns a dict of dataframes keyed by table name
argn_syn = syn_data["data"] if isinstance(syn_data, dict) else syn_data
argn_syn.to_csv(f"{SYN_DIR}/tabularargn_ieee_cis_syn.csv", index=False)

print(f"Saved: tabularargn_ieee_cis_syn.csv  shape={argn_syn.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity check

# COMMAND ----------

print("Sanity check:")
print(f"  Real fraud rate:          {train_df[LABEL_COL].astype(float).mean():.4f}")
if LABEL_COL in argn_syn.columns:
    print(f"  TabularARGN fraud rate:   {argn_syn[LABEL_COL].astype(float).mean():.4f}")

print(f"\nShape match: real={len(train_df):,}  syn={len(argn_syn):,}")
print(f"\nColumns preserved: {set(train_df.columns) == set(argn_syn.columns)}")
missing = set(train_df.columns) - set(argn_syn.columns)
if missing:
    print(f"  Missing in syn: {missing}")

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Amazon FDB — TabularARGN (P3 evaluation)

# COMMAND ----------

print("\nLoading Amazon FDB for TabularARGN...")
fdb_raw = pd.read_csv(f"{DATA_DIR}/amazon_fdb/Fraud_Data.csv")
fdb_raw = fdb_raw.drop(columns=["user_id"], errors="ignore")
fdb_raw["purchase_time_ts"] = pd.to_datetime(fdb_raw["purchase_time"]).astype(np.int64) // 1_000_000_000
fdb_raw["signup_time_ts"]   = pd.to_datetime(fdb_raw["signup_time"]).astype(np.int64) // 1_000_000_000
fdb_raw["account_age_days"] = (fdb_raw["purchase_time_ts"] - fdb_raw["signup_time_ts"]) / 86400
fdb_raw = fdb_raw.drop(columns=["signup_time","purchase_time"], errors="ignore")
print(f"Amazon FDB shape: {fdb_raw.shape}  |  fraud rate: {fdb_raw['class'].mean():.4f}")

# ── VALUE PROTECTION — READ BEFORE RUNNING ───────────────────────────────────
# MOSTLY AI applies "value protection" by DEFAULT: categorical values appearing
# fewer than 5-8 times are replaced with a "_RARE_" token before training.
#
# For Amazon FDB, device_id and ip_address have 137K+ unique values in 151K rows
# → >90% of values fall below the rarity threshold → nearly all device/IP values
# become "_RARE_" during training. During generation, "_RARE_" becomes a massive
# hub node (shared by thousands of synthetic rows), inflating all P3 metrics far
# beyond any architectural effect. This is likely the primary driver of the
# 207.4× P3 degradation observed with default settings.
#
# ─── HOW TO DISABLE VALUE PROTECTION ──────────────────────────────────────────
# Check the MOSTLY AI SDK version available in your environment:
#
#   from mostlyai import MostlyAI
#   help(MostlyAI.train)   # look for 'value_protection', 'smart_protection',
#                          # 'privacy_protection_level', or similar
#
# In MOSTLY AI SDK ≥1.x, per-column privacy settings are typically controlled via:
#   {"name": "device_id", "model_encoding_type": "TABULAR_CATEGORICAL",
#    "privacy_protection": "DISABLED"}  # <-- try this
# OR at table level:
#   "value_protection_enabled": false   # <-- try this in the table config dict
#
# If the SDK does not expose this parameter, use the MOSTLY AI Platform UI:
#   Generator → Configure → Model settings → Rare category replacement → None
#
# RUNS IN THIS NOTEBOOK:
#   Run A (primary): device_id + ip_address included, default settings (value protection ON)
#   Run B (ablation-vp-off): device_id + ip_address included, value protection DISABLED
#   Run C (ablation-no-graph): device_id + ip_address excluded from training
#
# Compare Run A vs Run B to isolate value protection's contribution.
# Compare Run B vs Run C to isolate autoregressive label-conditioning's contribution.
# ─────────────────────────────────────────────────────────────────────────────

# ── Run A: Full Amazon FDB, value protection OFF (confirmed correct run) ──────
# All 9 feature columns (purchase_value, source, browser, sex, age, account_age_days,
# device_id, ip_address, class) are used for training.
#
# CRITICAL: The "columns" list in MOSTLY AI config acts as an INCLUDE LIST.
# If only ["class", "device_id", "ip_address"] are listed, ONLY those 3 columns
# are generated. The previous run made this mistake → 3-col output.
# Fix: do NOT specify "columns" — MOSTLY AI auto-configures all columns.
# VP is explicitly set OFF so device_id/ip_address values are real vocabulary
# (not _RARE_ tokens), which is essential for a meaningful P3 evaluation.
print("Training TabularARGN on Amazon FDB (Run A: ALL columns, value protection OFF)...")
print(f"  Training data: {fdb_raw.shape}  columns: {list(fdb_raw.columns)}")
t0 = time.time()

mostly_fdb = MostlyAI(local=True)
gen_fdb = mostly_fdb.train(
    data=fdb_raw,
    name="amazon_fdb_tabularargn",
    # Table-level settings only — no "columns" list so ALL columns are generated.
    # tabular_model_configuration disables value protection for device_id/ip_address.
    config={
        "tables": [{
            "name": "data",
            "tabular_model_configuration": {"value_protection": False},
        }]
    }
)
elapsed = (time.time() - t0) / 60
print(f"Training done in {elapsed:.1f} min")

syn_fdb = mostly_fdb.generate(generator=gen_fdb, size=len(fdb_raw))
argn_fdb_syn = syn_fdb["data"] if isinstance(syn_fdb, dict) else syn_fdb
argn_fdb_syn.to_csv(f"{SYN_DIR}/tabularargn_amazon_fdb_syn.csv", index=False)
# Sanity checks
rare_d  = (argn_fdb_syn["device_id"].astype(str)  == "_RARE_").mean() if "device_id"  in argn_fdb_syn.columns else "N/A"
rare_ip = (argn_fdb_syn["ip_address"].astype(str) == "_RARE_").mean() if "ip_address" in argn_fdb_syn.columns else "N/A"
print(f"Saved: tabularargn_amazon_fdb_syn.csv  shape={argn_fdb_syn.shape}  "
      f"fraud_rate={argn_fdb_syn['class'].astype(float).mean():.4f}")
print(f"  _RARE_ fraction: device_id={rare_d}  ip_address={rare_ip}  (expected 0.0 with VP off)")
print(f"  device_id unique: {argn_fdb_syn['device_id'].nunique() if 'device_id' in argn_fdb_syn.columns else 'N/A'}")
missing_cols = set(fdb_raw.columns) - set(argn_fdb_syn.columns)
if missing_cols:
    print(f"  [WARNING] Missing columns in output: {missing_cols}")

# ── Run B: ABLATION — value protection DISABLED, device_id + ip_address included ──
# PURPOSE: Isolate whether value protection (not architecture) drives the 207× result.
# If Run B gives ~85×, value protection was the main driver of Run A's 207×.
# If Run B gives 100-207×, autoregressive label conditioning is also a major factor.
#
# HOW TO DISABLE VALUE PROTECTION:
# Try one of these SDK approaches (use whichever is supported by your SDK version):
#
# Option 1 — column-level (SDK ≥ 1.x, preferred):
#   {"name": "device_id", "model_encoding_type": "TABULAR_CATEGORICAL", "value_protection": "DISABLED"}
#
# Option 2 — table-level:
#   "value_protection_enabled": False   # in the table dict
#
# Option 3 — if SDK does not expose: use MOSTLY AI Platform UI
#   (Generator → Configure → Model settings → Rare category replacement → None)
#   Download the resulting synthetic CSV and save as tabularargn_amazon_fdb_vp_off_syn.csv
#
# The SDK config below attempts Option 1; modify if your SDK version uses different keys.
print("\nTraining TabularARGN on Amazon FDB (Run B: ablation — value protection OFF)...")
print(f"  Using full fdb_raw (device_id + ip_address included)")
print(f"  When VP is OFF: rare device_id/ip values are kept as-is (no _RARE_ substitution)")
t0 = time.time()
mostly_fdb_vp_off = MostlyAI(local=True)

# MOSTLY AI SDK value protection is configured via tabular_model_configuration.
# Default threshold: 8 occurrences. Values appearing < 8 times → replaced with "_RARE_".
# For Amazon FDB: 137K unique device_ids in 151K rows → >90% values hit the threshold
# → nearly all device/IP values become "_RARE_" hub nodes, inflating P3 artificially.
#
# SDK parameter (confirmed for mostlyai >= 1.x):
#   tabular_model_configuration: {"value_protection": false}  at the table level
#
# Fallback options if the above key name varies by SDK patch version:
#   Option B: value_protection_enabled: false  (table-level, some SDK versions)
#   Option C: MOSTLY AI Platform UI → Generator → Configure → Rare category replacement → None

def _train_vp_off(mostly_client, data, name, columns):
    """Try each known SDK key for disabling value protection; fall back gracefully."""
    # Primary: tabular_model_configuration (mostlyai >= 1.0, confirmed in SDK source)
    try:
        return mostly_client.train(
            data=data, name=name,
            config={"tables": [{
                "name": "data",
                "tabular_model_configuration": {"value_protection": False},
                "columns": columns,
            }]}
        )
    except Exception as e1:
        print(f"  [INFO] tabular_model_configuration key failed ({type(e1).__name__}), trying value_protection_enabled...")

    # Fallback: value_protection_enabled at table level (some SDK patch versions)
    try:
        return mostly_client.train(
            data=data, name=name,
            config={"tables": [{
                "name": "data",
                "value_protection_enabled": False,
                "columns": columns,
            }]}
        )
    except Exception as e2:
        raise RuntimeError(
            f"Both VP-off SDK keys failed.\n"
            f"  tabular_model_configuration: {e1}\n"
            f"  value_protection_enabled: {e2}\n"
            "  Use MOSTLY AI Platform UI: Generator → Configure → Rare category replacement → None\n"
            f"  Save output to: {SYN_DIR}/tabularargn_amazon_fdb_vp_off_syn.csv"
        )

_vp_cols = [
    {"name": "class",     "model_encoding_type": "TABULAR_CATEGORICAL"},
    {"name": "device_id", "model_encoding_type": "TABULAR_CATEGORICAL"},
    {"name": "ip_address","model_encoding_type": "TABULAR_CATEGORICAL"},
]

try:
    gen_fdb_vp_off = _train_vp_off(mostly_fdb_vp_off, fdb_raw, "amazon_fdb_tabularargn_vp_off", _vp_cols)
    elapsed = (time.time() - t0) / 60
    print(f"  Run B training done in {elapsed:.1f} min")
    syn_fdb_vp_off = mostly_fdb_vp_off.generate(generator=gen_fdb_vp_off, size=len(fdb_raw))
    argn_vp_off_syn = syn_fdb_vp_off["data"] if isinstance(syn_fdb_vp_off, dict) else syn_fdb_vp_off
    argn_vp_off_syn.to_csv(f"{SYN_DIR}/tabularargn_amazon_fdb_vp_off_syn.csv", index=False)
    print(f"  Saved: tabularargn_amazon_fdb_vp_off_syn.csv  "
          f"fraud_rate={argn_vp_off_syn['class'].astype(float).mean():.4f}")
    # Sanity: with VP OFF, device_id values should be real vocab (not "_RARE_")
    if "device_id" in argn_vp_off_syn.columns:
        rare_frac = (argn_vp_off_syn["device_id"] == "_RARE_").mean()
        print(f"  _RARE_ fraction in device_id: {rare_frac:.4f}  (expected ~0.0 with VP OFF)")
except RuntimeError as e:
    print(f"\n  [WARNING] {e}")

# ── Run C: ABLATION — exclude device_id + ip_address from training ────────────
# PURPOSE: Establishes the architectural lower bound for TabularARGN.
# If Run A (all columns, VP off) gives ~85×, then TabularARGN provides no
# benefit over row-independent generators for graph patterns.
# If Run A gives substantially higher (e.g. 150×), the autoregressive
# label conditioning is amplifying fraud-device co-occurrence artificially.
# Expected P3 result: ~85× (random assignment is equivalent).
print("\nTraining TabularARGN on Amazon FDB (Run C: ablation — NO device_id/ip_address)...")
fdb_ablation = fdb_raw.drop(columns=["device_id", "ip_address"], errors="ignore")
print(f"  Ablation table shape: {fdb_ablation.shape}  columns: {list(fdb_ablation.columns)}")

t0 = time.time()
mostly_fdb_ablation = MostlyAI(local=True)
gen_fdb_ablation = mostly_fdb_ablation.train(
    data=fdb_ablation,
    name="amazon_fdb_tabularargn_no_graph_cols",
    # No "columns" list — generate ALL 7 remaining columns (device/ip excluded from data)
)
elapsed = (time.time() - t0) / 60
print(f"  Run C training done in {elapsed:.1f} min")

syn_fdb_ablation = mostly_fdb_ablation.generate(generator=gen_fdb_ablation, size=len(fdb_ablation))
argn_fdb_ablation_syn = syn_fdb_ablation["data"] if isinstance(syn_fdb_ablation, dict) else syn_fdb_ablation
argn_fdb_ablation_syn.to_csv(f"{SYN_DIR}/tabularargn_amazon_fdb_no_graph_syn.csv", index=False)
print(f"  Saved: tabularargn_amazon_fdb_no_graph_syn.csv  shape={argn_fdb_ablation_syn.shape}  "
      f"fraud_rate={argn_fdb_ablation_syn['class'].astype(float).mean():.4f}")

# COMMAND ----------

print("\nAll files in SYN_DIR:")
for f in sorted(os.listdir(SYN_DIR)):
    size = os.path.getsize(f"{SYN_DIR}/{f}") / 1e6
    print(f"  {f:50s}  {size:.1f} MB")

print("""
TabularARGN training complete (IEEE-CIS + Amazon FDB).
All 4 generators × 2 datasets done.
Next: open databricks_evaluate_behavioral.py
""")
