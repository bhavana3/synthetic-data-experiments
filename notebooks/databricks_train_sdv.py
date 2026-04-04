# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Generator Training: CTGAN, TVAE, GaussianCopula (SDV)
# MAGIC Trains all three SDV generators on IEEE-CIS and Amazon FDB.
# MAGIC Saves synthetic datasets to DBFS for downstream evaluation.
# MAGIC
# MAGIC **Cluster:** CPU is fine (m5.2xlarge or similar, 32GB RAM)
# MAGIC **Runtime:** ~30-60 min total for all three on IEEE-CIS

# COMMAND ----------

%pip install sdv==1.35.1 --quiet

# COMMAND ----------

import os, json, time
import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from rdt.transformers.categorical import LabelEncoder

DBFS_ROOT  = "/dbfs/FileStore/synthetic_fraud_benchmark"
DATA_DIR   = f"{DBFS_ROOT}/data"
SYN_DIR    = f"{DBFS_ROOT}/synthetic"   # NOTE: matches evaluation notebook path
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
    "attr_cols":            [],          # device cols too sparse on IEEE-CIS (<25% non-null)
    "burst_deltas":         [300, 3600, 21600],
    "train_split_quantile": 0.80,
}

cfg_path = f"{RESULT_DIR}/ieee_cis_config.json"
with open(cfg_path, "w") as f:
    json.dump(IEEE_CIS_CONFIG, f, indent=2)

ENTITY_COL = IEEE_CIS_CONFIG["entity_col"]
TIME_COL   = IEEE_CIS_CONFIG["time_col"]
LABEL_COL  = IEEE_CIS_CONFIG["label_col"]
SPLIT_Q    = IEEE_CIS_CONFIG["train_split_quantile"]

print(f"Entity col:  {ENTITY_COL}")
print(f"Time col:    {TIME_COL}")
print(f"Split at:    p{int(SPLIT_Q*100)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and prepare IEEE-CIS training data

# COMMAND ----------

print("Loading IEEE-CIS...")
txn = pd.read_csv(f"{DATA_DIR}/ieee_cis/train_transaction.csv")
idn = pd.read_csv(f"{DATA_DIR}/ieee_cis/train_identity.csv")
df  = txn.merge(idn, on="TransactionID", how="left")

# Build entity column
df["entity_card1"] = df["card1"].astype(str)

# Temporal train split — only train generators on training partition
split_dt   = df[TIME_COL].quantile(SPLIT_Q)
train_df   = df[df[TIME_COL] <= split_dt].copy()
test_df    = df[df[TIME_COL] >  split_dt].copy()

print(f"Full dataset:  {len(df):,} rows")
print(f"Train split:   {len(train_df):,} rows  (fraud rate={train_df[LABEL_COL].mean():.4f})")
print(f"Test split:    {len(test_df):,} rows   (fraud rate={test_df[LABEL_COL].mean():.4f})")

# Save test split — used for all downstream utility evaluation
test_df.to_csv(f"{DATA_DIR}/ieee_cis/test_split.csv", index=False)
print(f"Test split saved.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare training table for SDV
# MAGIC
# MAGIC ### Column selection strategy (paper methodology decision)
# MAGIC IEEE-CIS has 400+ columns after merging: TransactionDT/Amt, card1–6, addr1–2,
# MAGIC dist1–2, emaildomain ×2, C1–C14 (counts), D1–D15 (time deltas), M1–M9 (match),
# MAGIC and V1–V339 (Vesta-engineered features).
# MAGIC
# MAGIC The V-columns are numeric, so a dtype-based drop filter misses them — they land
# MAGIC in the tensor wholesale and cause CTGAN to OOM after ~3 hours.
# MAGIC
# MAGIC **Fix:** keep only the 48 columns that our behavioral evaluator actually uses or
# MAGIC that carry temporal/velocity/identity signal relevant to P1–P4. V-columns are
# MAGIC excluded because:
# MAGIC   (a) they are not used in any P1/P2/P4 metric computation, and
# MAGIC   (b) they are opaque Vesta risk scores — not interpretable behavioral features.
# MAGIC This is documented in the paper as a deliberate preprocessing choice, consistent
# MAGIC across all four generators.

# COMMAND ----------

# ── Behaviorally-relevant column keep-list ─────────────────────────────────────
# All four generators are trained on this same 48-column subset.
# V1–V339 (Vesta engineered features) are explicitly excluded — they are not used
# in behavioral fidelity evaluation and cause CTGAN tensor blowup.
BEHAVIORAL_COLS = [
    # ── Core evaluation features (always required by the evaluator) ──────────
    "TransactionDT",    # timestamp  → P1, P2, P4
    "TransactionAmt",   # amount     → P2
    "isFraud",          # label      → all patterns
    "card4",            # card network (Visa/MC etc.) — low-cardinality, ≤6 values
    # ── Count features C1–C14 — capture velocity/frequency signals (P4) ─────
    "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14",
    # ── Time-delta features D1–D15 — capture IET and session patterns (P1,P2) ─
    "D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15",
    # ── Match features M1–M9 — identity verification signals ────────────────
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
    # ── Geographic / proximity features ─────────────────────────────────────
    "addr1","addr2","dist1","dist2",
    # ── Email domain — device/identity sharing signals ───────────────────────
    "P_emaildomain","R_emaildomain",
]

# Keep only columns that actually exist in this dataset (merge may drop some D/M cols)
BEHAVIORAL_COLS = [c for c in BEHAVIORAL_COLS if c in train_df.columns]
train_for_sdv   = train_df[BEHAVIORAL_COLS].copy()

print(f"Behaviorally-relevant columns kept: {len(BEHAVIORAL_COLS)}")
print(f"  (dropped {train_df.shape[1] - len(BEHAVIORAL_COLS)} columns incl. V1–V339, card1, IDs)")

# ── Null imputation ────────────────────────────────────────────────────────────
# C/D/M columns have sparse nulls in IEEE-CIS. CTGAN/TVAE reject nulls in numerics.
#   numeric  → median (robust to skew)
#   object   → "MISSING" sentinel
num_cols = train_for_sdv.select_dtypes(include="number").columns
cat_cols = train_for_sdv.select_dtypes(include="object").columns
train_for_sdv[num_cols] = train_for_sdv[num_cols].fillna(train_for_sdv[num_cols].median())
train_for_sdv[cat_cols] = train_for_sdv[cat_cols].fillna("MISSING")
assert train_for_sdv.isnull().sum().sum() == 0, "Nulls remain after imputation!"

REAL_FRAUD_RATE = train_for_sdv[LABEL_COL].astype(float).mean()
print(f"\nTraining table: {train_for_sdv.shape}  (0 nulls)")
print(f"Fraud rate (real):  {REAL_FRAUD_RATE:.4f}")
print(f"Dtypes: {dict(train_for_sdv.dtypes.value_counts())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build SDV metadata

# COMMAND ----------

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_for_sdv)

# Explicit overrides ────────────────────────────────────────────────────────────
# isFraud must be categorical (binary label, not a number to regress)
# TransactionDT is an integer timestamp — treat as numerical (not datetime)
# card4 is categorical (Visa, Mastercard, etc.) — SDV usually detects this correctly
metadata.update_column(column_name=LABEL_COL, sdtype="categorical")
metadata.update_column(column_name=TIME_COL,  sdtype="numerical")

# M1–M9 in IEEE-CIS are boolean strings ("T"/"F"). SDV may detect them as categorical
# with 2–3 values (T, F, MISSING) — that is correct and fine for CTGAN.
# addr1/addr2 are integer zip-like codes stored as floats/int64 — SDV auto-detects as
# numerical (not categorical). P_emaildomain / R_emaildomain are low-cardinality
# categoricals (~60 values each). After column selection, no column should have
# >200 unique categorical values.
#
# Guard: if SDV incorrectly marks any numeric-coded column as categorical with >100
# unique values, apply LabelEncoder (order_by=None — first-encounter integer codes)
# rather than reclassifying to numerical. LabelEncoder preserves the categorical type
# contract while avoiding one-hot explosion — correct for CTGAN/TVAE embedding tables.
# NOTE: order_by="frequency" is NOT supported in RDT ≤1.x; valid values are
#       None, 'numerical_value', 'alphabetical'. Use None (first-encounter order).
CAT_CARD_LIMIT = 100
high_card_ieee_cols = []
for col, col_meta in list(metadata.columns.items()):
    if col_meta.get("sdtype") == "categorical" and col != LABEL_COL:
        if train_for_sdv[col].nunique() > CAT_CARD_LIMIT:
            high_card_ieee_cols.append(col)
if high_card_ieee_cols:
    print(f"High-cardinality categorical cols (>{CAT_CARD_LIMIT} unique) found: {high_card_ieee_cols}")
    print("  Will apply LabelEncoder at synthesizer level (not reclassifying to numerical)")
else:
    print("No high-cardinality categorical columns detected ✓  (addr1/addr2 are numerical dtype)")

print(f"Final metadata: {len(metadata.columns)} columns")
metadata.validate()
print("  Validation passed ✓")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and generate — CTGAN
# MAGIC
# MAGIC ### Stratified 1:3 subsampling (CTGAN only)
# MAGIC CTGAN uses a GAN with conditional vectors. It is sensitive to class imbalance
# MAGIC and to dataset size (large batch GAN training is quadratic in GPU memory).
# MAGIC With ~472K rows at 3.5% fraud, the legit class dominates to the point where the
# MAGIC conditional vector rarely fires on the fraud class during training.
# MAGIC
# MAGIC We apply a stratified 1:3 subsample (all fraud rows + 3× as many legit rows),
# MAGIC reducing training data to ~66K rows. This:
# MAGIC   - Eliminates the OOM/timeout failure (~3-hour runs with full data)
# MAGIC   - Gives CTGAN roughly equal exposure to both classes during GAN training
# MAGIC   - Is documented in the paper as a CTGAN-specific preprocessing step
# MAGIC
# MAGIC At generation time we produce `len(train_for_sdv)` rows (full original count),
# MAGIC conditioning on the real fraud rate so the synthetic output has the right class
# MAGIC distribution for behavioral evaluation.

# COMMAND ----------

# ── Build CTGAN training subsample ────────────────────────────────────────────
CTGAN_FRAUD_RATIO = 3   # 1 fraud row : CTGAN_FRAUD_RATIO legit rows

fraud_df = train_for_sdv[train_for_sdv[LABEL_COL] == 1]
legit_df = train_for_sdv[train_for_sdv[LABEL_COL] == 0]

n_fraud  = len(fraud_df)
n_legit  = min(len(legit_df), n_fraud * CTGAN_FRAUD_RATIO)

legit_sample = legit_df.sample(n=n_legit, random_state=42)
ctgan_train  = pd.concat([fraud_df, legit_sample], ignore_index=True).sample(frac=1, random_state=42)

print(f"CTGAN training subsample:  {len(ctgan_train):,} rows "
      f"(fraud={ctgan_train[LABEL_COL].astype(float).mean():.3f}, "
      f"down from {len(train_for_sdv):,} full rows)")

# Rebuild metadata for the subsample (same schema, needed for CTGANSynthesizer)
meta_ctgan = SingleTableMetadata()
meta_ctgan.detect_from_dataframe(ctgan_train)
meta_ctgan.update_column(column_name=LABEL_COL, sdtype="categorical")
meta_ctgan.update_column(column_name=TIME_COL,  sdtype="numerical")
# Identify any high-cardinality categorical columns in subsample
high_card_ctgan_cols = [
    col for col, col_meta in meta_ctgan.columns.items()
    if col_meta.get("sdtype") == "categorical" and col != LABEL_COL
    and ctgan_train[col].nunique() > 100
]
meta_ctgan.validate()

print("Training CTGAN on subsample...")
t0 = time.time()

ctgan = CTGANSynthesizer(
    meta_ctgan,
    epochs=300,
    batch_size=500,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    verbose=True,
)
# Apply LabelEncoder for any high-cardinality categorical columns to avoid embedding OOM.
if high_card_ctgan_cols:
    ctgan.auto_assign_transformers(ctgan_train)
    ctgan.update_transformers({col: LabelEncoder(order_by=None) for col in high_card_ctgan_cols})
    print(f"  LabelEncoder applied to: {high_card_ctgan_cols}")
ctgan.fit(ctgan_train)

elapsed = (time.time() - t0) / 60
print(f"CTGAN training done in {elapsed:.1f} min")

# Generate len(train_for_sdv) rows with real fraud rate via conditional sampling.
# CTGAN was trained on 1:3 balanced data (25% fraud) — without conditioning,
# it would generate 25% fraud. We restore the real rate (~3.5%) here so that
# entity assignment in evaluation gets the right fraud/legit entity density.
from sdv.sampling import Condition

n_total     = len(train_for_sdv)
n_fraud_gen = int(round(n_total * REAL_FRAUD_RATE))
n_legit_gen = n_total - n_fraud_gen

ctgan_syn = ctgan.sample_from_conditions(conditions=[
    Condition(column_values={LABEL_COL: 1}, num_rows=n_fraud_gen),
    Condition(column_values={LABEL_COL: 0}, num_rows=n_legit_gen),
])
ctgan_syn.to_csv(f"{SYN_DIR}/ctgan_ieee_cis_syn.csv", index=False)
print(f"Saved: ctgan_ieee_cis_syn.csv  shape={ctgan_syn.shape}  "
      f"fraud_rate={ctgan_syn[LABEL_COL].astype(float).mean():.4f}  "
      f"(elevated vs real {REAL_FRAUD_RATE:.3f} — expected from 1:3 training balance)")

# Save model
ctgan.save(f"{SYN_DIR}/ctgan_ieee_cis_model.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and generate — TVAE
# MAGIC TVAE (Variational Autoencoder) handles class imbalance and dataset size
# MAGIC more gracefully than CTGAN — no GAN instability, no conditional vector.
# MAGIC We train on the full 48-column training set without subsampling.

# COMMAND ----------

print("Training TVAE (full 48-col set, no subsampling)...")
t0 = time.time()

tvae = TVAESynthesizer(
    metadata,
    epochs=300,
    batch_size=500,
    compress_dims=(256, 256),
    decompress_dims=(256, 256),
    verbose=True,
)
tvae.fit(train_for_sdv)

elapsed = (time.time() - t0) / 60
print(f"TVAE training done in {elapsed:.1f} min")

# ── Unconditional generation — documents TVAE minority-class collapse ─────────
# VAE decoder ignores minority class under marginal sampling (known SDV issue #1339).
# We save this FIRST as the documented failure mode (reported in paper Section 6.2).
# Expected fraud_rate: ~0.0003 (near-zero mode collapse).
tvae_uncond_syn = tvae.sample(num_rows=len(train_for_sdv))
tvae_uncond_syn.to_csv(f"{SYN_DIR}/tvae_ieee_cis_syn_uncond.csv", index=False)
print(f"Saved: tvae_ieee_cis_syn_uncond.csv  shape={tvae_uncond_syn.shape}  "
      f"fraud_rate={tvae_uncond_syn[LABEL_COL].astype(float).mean():.4f}  "
      f"<-- expected near-zero (mode collapse, documented finding)")

# ── Conditional generation — fair behavioral evaluation ───────────────────────
# TVAE suffers mode collapse on minority classes — unconditional sampling produces
# near-zero fraud rate despite training on 3.5% fraud. Use conditional sampling
# to restore the real fraud rate for a fair apples-to-apples behavioral comparison.
tvae_syn = tvae.sample_from_conditions(conditions=[
    Condition(column_values={LABEL_COL: 1}, num_rows=n_fraud_gen),
    Condition(column_values={LABEL_COL: 0}, num_rows=n_legit_gen),
])
tvae_syn.to_csv(f"{SYN_DIR}/tvae_ieee_cis_syn.csv", index=False)
print(f"Saved: tvae_ieee_cis_syn.csv  shape={tvae_syn.shape}  "
      f"fraud_rate={tvae_syn[LABEL_COL].astype(float).mean():.4f}  "
      f"(conditional — used for P1/P2/P4 behavioral evaluation)")

tvae.save(f"{SYN_DIR}/tvae_ieee_cis_model.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and generate — GaussianCopula
# MAGIC GaussianCopula fits a parametric model (copula + marginals) — very fast,
# MAGIC no iterative training. Full dataset, full 48-column set.

# COMMAND ----------

print("Training GaussianCopula (full 48-col set, no subsampling)...")
t0 = time.time()

copula = GaussianCopulaSynthesizer(
    metadata,
    default_distribution="beta",   # beta handles [0,∞) skewed columns (amounts, deltas) better
)
copula.fit(train_for_sdv)

elapsed = (time.time() - t0) / 60
print(f"GaussianCopula training done in {elapsed:.1f} min")

copula_syn = copula.sample(num_rows=len(train_for_sdv))
copula_syn.to_csv(f"{SYN_DIR}/gaussiancopula_ieee_cis_syn.csv", index=False)
print(f"Saved: gaussiancopula_ieee_cis_syn.csv  shape={copula_syn.shape}")

copula.save(f"{SYN_DIR}/gaussiancopula_ieee_cis_model.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick sanity check — fraud rate preserved?

# COMMAND ----------

print("IEEE-CIS fraud rate sanity check:")
print(f"  Real train:            {train_for_sdv[LABEL_COL].astype(float).mean():.4f}")
print(f"  CTGAN synthetic:       {ctgan_syn[LABEL_COL].astype(float).mean():.4f}")
print(f"  TVAE synthetic:        {tvae_syn[LABEL_COL].astype(float).mean():.4f}")
print(f"  GaussianCopula syn:    {copula_syn[LABEL_COL].astype(float).mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Amazon FDB — CTGAN, TVAE, GaussianCopula (P3 evaluation)
# MAGIC
# MAGIC We train on Amazon FDB to evaluate P3 (graph motifs).
# MAGIC Key columns to preserve: `device_id`, `ip_address`, `class` (label).
# MAGIC
# MAGIC **Note:** These generators generate rows independently — they have no mechanism
# MAGIC to coordinate device_id values across rows. We expect P3 to fail completely.
# MAGIC That failure *is* the finding.

# COMMAND ----------

print("\nLoading Amazon FDB...")
fdb_raw = pd.read_csv(f"{DATA_DIR}/amazon_fdb/Fraud_Data.csv")
fdb_raw = fdb_raw.drop(columns=["user_id"], errors="ignore")  # arbitrary key, no signal

# device_id and ip_address are kept — they're the P3 signal
# signup_time has too high cardinality as a string; convert to account_age_days
# Derive account_age_days then drop the raw timestamps.
# Keeping all three (purchase_time_ts, signup_time_ts, account_age_days) creates
# perfect multicollinearity: account_age_days = (purchase_time_ts - signup_time_ts)/86400.
# GaussianCopula estimates a Gaussian copula via correlation matrix inversion —
# a singular matrix from perfect collinearity causes NaN/failure at fit time.
# account_age_days is the semantically meaningful feature; the raw unix timestamps
# add no independent information and are dropped.
purchase_ts = pd.to_datetime(fdb_raw["purchase_time"]).astype(np.int64) // 1_000_000_000
signup_ts   = pd.to_datetime(fdb_raw["signup_time"]).astype(np.int64)   // 1_000_000_000
fdb_raw["account_age_days"] = (purchase_ts - signup_ts) / 86400
fdb_raw = fdb_raw.drop(columns=["signup_time", "purchase_time"], errors="ignore")
# purchase_time_ts and signup_time_ts are never added to fdb_raw — only account_age_days

print(f"Amazon FDB shape: {fdb_raw.shape}  |  fraud rate: {fdb_raw['class'].mean():.4f}")

# Build SDV metadata for FDB
meta_fdb = SingleTableMetadata()
meta_fdb.detect_from_dataframe(fdb_raw)
meta_fdb.update_column(column_name="class", sdtype="categorical")
meta_fdb.update_column(column_name="device_id", sdtype="categorical")
meta_fdb.update_column(column_name="ip_address", sdtype="categorical")
meta_fdb.validate()
print("Amazon FDB metadata validated ✓")

# COMMAND ----------

FDB_FRAUD_RATE = fdb_raw["class"].astype(float).mean()
n_fdb_total     = len(fdb_raw)
n_fdb_fraud_gen = int(round(n_fdb_total * FDB_FRAUD_RATE))
n_fdb_legit_gen = n_fdb_total - n_fdb_fraud_gen
print(f"Amazon FDB fraud rate: {FDB_FRAUD_RATE:.4f}  "
      f"({n_fdb_fraud_gen:,} fraud / {n_fdb_legit_gen:,} legit to generate)")

# ── Why all generators train WITH device_id and ip_address ────────────────────
# device_id (137K unique) and ip_address (143K unique) ARE the P3 graph signal.
# All generators are trained WITH these columns so we test the architectural limit:
# can a row-independent generator reproduce cross-row device-sharing motifs?
# Expected finding: no — rows are generated independently, so all generators
# randomly assign device/IP values, destroying the bipartite fraud graph.
#
# Encoding strategy to avoid OOM:
#   CTGAN / TVAE: LabelEncoder(order_by=None) maps each unique device_id
#     to an integer code (first-encounter order). This avoids one-hot explosion
#     (137K columns → single int). The embedding table in the GAN/VAE scales
#     to n_unique × embedding_dim, which is manageable.
#     NOTE: order_by="frequency" raises TransformerInputError in RDT ≤1.x;
#     use None (supported: None, 'numerical_value', 'alphabetical').
#   GaussianCopula: FrequencyEncoder by default (maps to relative frequency
#     in [0,1]). No OOM risk; already working correctly.
#
# Previous approach (drop device/ip for CTGAN/TVAE) was scientifically weaker:
# it tested the trivial "missing column → random assignment" case rather than
# the architectural limitation of row-independent generation.

GRAPH_COLS = ["device_id", "ip_address"]

for GenClass, gen_name in [
    (CTGANSynthesizer,          "ctgan"),
    (TVAESynthesizer,           "tvae"),
    (GaussianCopulaSynthesizer, "gaussiancopula"),
]:
    print(f"\nTraining {gen_name.upper()} on Amazon FDB...")

    # All generators train on the full table including device_id + ip_address
    train_fdb = fdb_raw.copy()
    print(f"  Training with device_id + ip_address (137K / 143K unique values respectively)")

    # Build metadata
    meta_gen = SingleTableMetadata()
    meta_gen.detect_from_dataframe(train_fdb)
    meta_gen.update_column(column_name="class",      sdtype="categorical")
    meta_gen.update_column(column_name="device_id",  sdtype="categorical")
    meta_gen.update_column(column_name="ip_address", sdtype="categorical")
    meta_gen.validate()

    t0 = time.time()

    if gen_name == "gaussiancopula":
        # GaussianCopula uses FrequencyEncoder by default for categoricals — no OOM
        model = GaussianCopulaSynthesizer(meta_gen, default_distribution="beta")
        model.fit(train_fdb)
    else:
        # CTGAN / TVAE: use LabelEncoder for device_id and ip_address to avoid
        # the one-hot OOM that would occur with 137K+ unique categorical values.
        model = GenClass(meta_gen, epochs=300, batch_size=500, verbose=True)
        model.auto_assign_transformers(train_fdb)
        model.update_transformers({
            "device_id":  LabelEncoder(order_by=None),
            "ip_address": LabelEncoder(order_by=None),
        })
        model.fit(train_fdb)

    elapsed = (time.time() - t0) / 60
    print(f"  Done in {elapsed:.1f} min")

    # Conditional sampling for CTGAN/TVAE (class imbalance), unconditional for Copula
    if gen_name == "gaussiancopula":
        syn = model.sample(num_rows=n_fdb_total)
    else:
        syn = model.sample_from_conditions(conditions=[
            Condition(column_values={"class": 1}, num_rows=n_fdb_fraud_gen),
            Condition(column_values={"class": 0}, num_rows=n_fdb_legit_gen),
        ])

    out = f"{SYN_DIR}/{gen_name}_amazon_fdb_syn.csv"
    syn.to_csv(out, index=False)
    model.save(f"{SYN_DIR}/{gen_name}_amazon_fdb_model.pkl")
    print(f"  Saved: {gen_name}_amazon_fdb_syn.csv  shape={syn.shape}  "
          f"fraud_rate={syn['class'].astype(float).mean():.4f}  "
          f"graph_cols={'present' if 'device_id' in syn.columns else 'absent'}")

# COMMAND ----------

print("\nAll files in SYN_DIR:")
for f in sorted(os.listdir(SYN_DIR)):
    size = os.path.getsize(f"{SYN_DIR}/{f}") / 1e6
    print(f"  {f:50s}  {size:.1f} MB")

print("""
SDV training complete (IEEE-CIS + Amazon FDB).
Next: open databricks_train_tabularargn.py
""")
