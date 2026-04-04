# Databricks notebook source
# MAGIC %md
# MAGIC ## Layer 1 (Statistical Fidelity) + Layer 2 (TSTR AUROC) Evaluation
# MAGIC
# MAGIC Computes the three missing columns in Table 1 of the paper:
# MAGIC   - Layer 1a: Mean Jensen-Shannon divergence across all 48 columns
# MAGIC   - Layer 1b: Mean absolute pairwise correlation matrix difference
# MAGIC   - Layer 2:  TSTR AUROC (XGBoost trained on synthetic, tested on real held-out)
# MAGIC
# MAGIC Run this notebook ONCE after all four synthetic CSVs are available in SYN_DIR.
# MAGIC Results print to stdout and can be pasted directly into Table 1 of main.tex.

# COMMAND ----------

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ─── Paths — edit to match your cluster ───────────────────────────────────────
DATA_DIR = "/dbfs/FileStore/synthetic_fraud/data"
SYN_DIR  = "/dbfs/FileStore/synthetic_fraud/synthetic"

GENERATORS = ["ctgan", "tvae", "gaussiancopula", "tabularargn"]

# File naming convention (must match what training notebooks saved)
SYN_FILES = {
    "ctgan":          f"{SYN_DIR}/ctgan_ieee_cis_syn.csv",
    "tvae":           f"{SYN_DIR}/tvae_ieee_cis_syn.csv",
    "gaussiancopula": f"{SYN_DIR}/gaussiancopula_ieee_cis_syn.csv",
    "tabularargn":    f"{SYN_DIR}/tabularargn_ieee_cis_syn.csv",
}

LABEL_COL = "isFraud"
TIME_COL  = "TransactionDT"
SEED = 42

# COMMAND ----------

# ─── Load real data, reproduce 80/20 temporal split ───────────────────────────
print("Loading real IEEE-CIS data...")
raw = pd.read_csv(f"{DATA_DIR}/ieee_cis/train_transaction.csv")

# Merge identity table if available
id_path = f"{DATA_DIR}/ieee_cis/train_identity.csv"
if os.path.exists(id_path):
    identity = pd.read_csv(id_path)
    raw = raw.merge(identity, on="TransactionID", how="left")

raw = raw.drop(columns=["TransactionID"], errors="ignore")

# Temporal 80/20 split (same as training)
split_dt = raw[TIME_COL].quantile(0.80)
real_train = raw[raw[TIME_COL] <= split_dt].copy()
real_test  = raw[raw[TIME_COL] >  split_dt].copy()
print(f"Real train: {len(real_train):,}  |  Real test: {len(real_test):,}")

# 48-column keep list (same as training)
V_COLS = [c for c in real_train.columns if c.startswith("V") and c[1:].isdigit()]
real_train_48 = real_train.drop(columns=V_COLS + ["card1"], errors="ignore")
real_test_48  = real_test.drop(columns=V_COLS + ["card1"], errors="ignore")
FEATURE_COLS = [c for c in real_train_48.columns if c != LABEL_COL]
print(f"Feature columns: {len(FEATURE_COLS)}")

# COMMAND ----------

# ─── Helper: Jensen-Shannon divergence per column ─────────────────────────────
def js_divergence_categorical(s_real, s_syn):
    """JS divergence between two categorical series."""
    cats = set(s_real.dropna().unique()) | set(s_syn.dropna().unique())
    p = s_real.value_counts(normalize=True).reindex(cats, fill_value=0).values
    q = s_syn.value_counts(normalize=True).reindex(cats, fill_value=0).values
    return float(jensenshannon(p, q))

def js_divergence_continuous(s_real, s_syn, n_bins=50):
    """JS divergence between two continuous series via equal-width binning."""
    combined = pd.concat([s_real.dropna(), s_syn.dropna()])
    bins = np.linspace(combined.min(), combined.max(), n_bins + 1)
    p, _ = np.histogram(s_real.dropna(), bins=bins, density=False)
    q, _ = np.histogram(s_syn.dropna(), bins=bins, density=False)
    p = (p + 1e-10) / (p.sum() + n_bins * 1e-10)   # Laplace smoothing
    q = (q + 1e-10) / (q.sum() + n_bins * 1e-10)
    return float(jensenshannon(p, q))

def mean_js_divergence(real_df, syn_df, feature_cols):
    """Layer 1a: mean JS divergence across all feature columns."""
    scores = []
    for col in feature_cols:
        if col not in syn_df.columns:
            continue
        r = real_df[col]
        s = syn_df[col]
        if r.dtype == object or r.nunique() < 20:
            scores.append(js_divergence_categorical(r, s))
        else:
            scores.append(js_divergence_continuous(r, s))
    return float(np.mean(scores))

# COMMAND ----------

# ─── Helper: pairwise correlation matrix difference ───────────────────────────
def corr_matrix_delta(real_df, syn_df, feature_cols, n_sample=5000):
    """Layer 1b: mean absolute difference in pairwise Spearman correlation matrix."""
    cols = [c for c in feature_cols if c in syn_df.columns]
    # Encode categoricals to numeric for correlation
    real_enc = real_df[cols].copy()
    syn_enc  = syn_df[cols].copy()
    for col in cols:
        if real_enc[col].dtype == object:
            codes = {v: i for i, v in enumerate(real_enc[col].fillna("__NaN__").unique())}
            real_enc[col] = real_enc[col].fillna("__NaN__").map(codes).fillna(-1)
            syn_enc[col]  = syn_enc[col].fillna("__NaN__").map(codes).fillna(-1)
        else:
            real_enc[col] = pd.to_numeric(real_enc[col], errors="coerce").fillna(0)
            syn_enc[col]  = pd.to_numeric(syn_enc[col], errors="coerce").fillna(0)
    # Sample for speed
    r_sample = real_enc.sample(min(n_sample, len(real_enc)), random_state=SEED)
    s_sample = syn_enc.sample(min(n_sample, len(syn_enc)), random_state=SEED)
    corr_r = r_sample.corr(method="spearman").fillna(0).values
    corr_s = s_sample.corr(method="spearman").fillna(0).values
    n = len(cols)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(np.mean(np.abs(corr_r[mask] - corr_s[mask])))

# COMMAND ----------

# ─── Helper: TSTR AUROC ───────────────────────────────────────────────────────
def tstr_auroc(syn_df, real_test_df, feature_cols, label_col):
    """Layer 2: train XGBoost on synthetic, test on real held-out."""
    cols = [c for c in feature_cols if c in syn_df.columns]

    # Encode and align
    X_train = syn_df[cols].copy()
    y_train = syn_df[label_col].astype(int)
    X_test  = real_test_df[cols].copy()
    y_test  = real_test_df[label_col].astype(int)

    for col in cols:
        if X_train[col].dtype == object:
            codes = {v: i for i, v in enumerate(X_train[col].fillna("__NaN__").unique())}
            X_train[col] = X_train[col].fillna("__NaN__").map(codes).fillna(-1)
            X_test[col]  = X_test[col].fillna("__NaN__").map(codes).fillna(-1)
        else:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0)
            X_test[col]  = pd.to_numeric(X_test[col], errors="coerce").fillna(0)

    # Skip if synthetic has no fraud (TVAE collapse)
    if y_train.sum() == 0:
        print("  [WARNING] Synthetic has zero fraud rows — AUROC undefined, returning 0.500")
        return 0.500

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, y_prob))

# COMMAND ----------

# ─── Real-vs-Real TRTR AUROC baseline ────────────────────────────────────────
print("\n=== TRTR baseline (train on real, test on real) ===")
X_tr = real_train_48[FEATURE_COLS].copy()
y_tr = real_train_48[LABEL_COL].astype(int)
X_te = real_test_48[FEATURE_COLS].copy()
y_te = real_test_48[LABEL_COL].astype(int)
for col in FEATURE_COLS:
    if X_tr[col].dtype == object:
        codes = {v: i for i, v in enumerate(X_tr[col].fillna("__NaN__").unique())}
        X_tr[col] = X_tr[col].fillna("__NaN__").map(codes).fillna(-1)
        X_te[col] = X_te[col].fillna("__NaN__").map(codes).fillna(-1)
    else:
        X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce").fillna(0)
        X_te[col] = pd.to_numeric(X_te[col], errors="coerce").fillna(0)

scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
trtr_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=scale_pos, use_label_encoder=False,
    eval_metric="logloss", random_state=SEED, n_jobs=-1,
)
trtr_model.fit(X_tr, y_tr)
trtr_auroc = float(roc_auc_score(y_te, trtr_model.predict_proba(X_te)[:, 1]))
print(f"TRTR AUROC: {trtr_auroc:.4f}")

# COMMAND ----------

# ─── Main evaluation loop ─────────────────────────────────────────────────────
results = {}

for gen in GENERATORS:
    syn_path = SYN_FILES.get(gen)
    if not syn_path or not os.path.exists(syn_path):
        print(f"\n[{gen.upper()}] Synthetic file not found: {syn_path} — skipping")
        continue

    print(f"\n=== {gen.upper()} ===")
    syn = pd.read_csv(syn_path)
    print(f"  Synthetic shape: {syn.shape}  |  fraud_rate: {syn[LABEL_COL].astype(float).mean():.4f}")

    # Align columns to 48-col subset
    syn_48 = syn[[c for c in real_train_48.columns if c in syn.columns]].copy()
    feat_cols_avail = [c for c in FEATURE_COLS if c in syn_48.columns]

    print(f"  Computing Layer 1a (JS divergence)...")
    js_div = mean_js_divergence(real_train_48, syn_48, feat_cols_avail)
    print(f"    JS Divergence (mean): {js_div:.4f}")

    print(f"  Computing Layer 1b (correlation delta)...")
    corr_delta = corr_matrix_delta(real_train_48, syn_48, feat_cols_avail)
    print(f"    Corr |Delta| (mean): {corr_delta:.4f}")

    print(f"  Computing Layer 2 (TSTR AUROC)...")
    auroc = tstr_auroc(syn_48, real_test_48, feat_cols_avail, LABEL_COL)
    print(f"    TSTR AUROC: {auroc:.4f}")

    results[gen] = {
        "fraud_rate": float(syn[LABEL_COL].astype(float).mean()),
        "js_div":     js_div,
        "corr_delta": corr_delta,
        "tstr_auroc": auroc,
    }

# COMMAND ----------

# ─── Print Table 1 LaTeX rows ─────────────────────────────────────────────────
print("\n\n" + "="*70)
print("PAPER TABLE 1 (copy-paste into main.tex Table tab:layer1_2)")
print("="*70)
print(f"Real (TRTR)    & 0.035 & ---   & ---   & {trtr_auroc:.3f} \\\\")
print("\\midrule")

gen_labels = {
    "ctgan":          r"CTGAN",
    "tvae":           r"TVAE$^\dagger$",
    "gaussiancopula": r"GaussianCopula",
    "tabularargn":    r"TabularARGN",
}
for gen, label in gen_labels.items():
    if gen not in results:
        print(f"{label:20s} & --- & --- & --- & --- \\\\  % not evaluated")
        continue
    r = results[gen]
    print(f"{label:20s} & {r['fraud_rate']:.3f} & {r['js_div']:.3f} & "
          f"{r['corr_delta']:.3f} & {r['tstr_auroc']:.3f} \\\\")

print("="*70)
print("\nDone. Paste the above rows into Table 1 in main.tex and recompile.")
