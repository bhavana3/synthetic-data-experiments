"""
Behavioral Feature Engineering Pipeline — Vectorized
======================================================
Phase 2: Enrich public datasets with formally-defined behavioral fraud signals.

Converts raw transaction data into an entity-level behavioral feature table.
This table IS the benchmark — ground-truth behavioral pattern values for real
data, against which synthetic data is compared.

All features use pandas vectorized operations (no Python loops over entities).

Outputs:
  results/benchmark/entity_behavioral_{dataset}.csv  — one row per entity
  results/benchmark/txn_behavioral_{dataset}.csv     — per-txn P4 features

Usage:
    python features/behavioral_feature_engineering.py --dataset ieee_cis
    python features/behavioral_feature_engineering.py --dataset amazon_fdb
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent

DATASET_CONFIG = {
    "ieee_cis": {
        "txn_file":    "train_transaction.csv",
        "id_file":     "train_identity.csv",
        "join_key":    "TransactionID",
        "entity_col":  "card1",
        "time_col":    "TransactionDT",
        "label_col":   "isFraud",
        "amount_col":  "TransactionAmt",
        "device_cols": [],         # too sparse per EDA
        "card_col":    "card4",
    },
    "amazon_fdb": {
        "txn_file":    "Fraud_Data.csv",
        "id_file":     None,
        "join_key":    None,
        "entity_col":  "user_id",
        "time_col":    "purchase_time",
        "label_col":   "class",
        "amount_col":  "purchase_value",
        "device_cols": ["device_id", "ip_address"],
        "card_col":    None,
    },
}

BURST_DELTAS = [300, 3600, 21600]  # 5min, 1hr, 6hr


# ── Load ──────────────────────────────────────────────────────────────────────

def load_dataset(cfg: dict, data_dir: Path) -> pd.DataFrame:
    print(f"Loading {cfg['txn_file']}...")
    df = pd.read_csv(data_dir / cfg["txn_file"])

    if cfg["id_file"]:
        idn = pd.read_csv(data_dir / cfg["id_file"])
        df = df.merge(idn, on=cfg["join_key"], how="left")

    df["entity"] = df[cfg["entity_col"]].astype(str)

    lbl = df[cfg["label_col"]]
    df["label"] = (lbl.astype(str).str.lower().str.strip() == "fraud").astype(int) \
        if lbl.dtype == object else lbl.astype(int)

    tc = cfg["time_col"]
    df["timestamp"] = pd.to_datetime(df[tc]).astype(np.int64) // 1_000_000_000 \
        if df[tc].dtype == object else df[tc].astype(float)

    if cfg["amount_col"] and cfg["amount_col"] in df.columns:
        df["amount"] = pd.to_numeric(df[cfg["amount_col"]], errors="coerce")
    else:
        df["amount"] = np.nan

    df = df.sort_values(["entity", "timestamp"]).reset_index(drop=True)

    print(f"  Rows: {len(df):,}  |  Entities: {df['entity'].nunique():,}  |  Fraud rate: {df['label'].mean():.4f}")
    return df


# ── P1: Inter-event time features (vectorized) ────────────────────────────────

def compute_p1_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  [P1] Inter-event time features...")

    # IET = diff within entity
    df["iet"] = df.groupby("entity")["timestamp"].diff()

    # Entity-level label
    entity_label = df.groupby("entity")["label"].max().rename("label")
    n_txn        = df.groupby("entity")["timestamp"].count().rename("n_transactions")

    iet = df.dropna(subset=["iet"])

    # Distribution stats
    p1 = iet.groupby("entity")["iet"].agg(
        iet_mean   = "mean",
        iet_std    = "std",
        iet_median = "median",
        iet_p10    = lambda x: x.quantile(0.10),
        iet_p90    = lambda x: x.quantile(0.90),
        iet_pct_under_5min  = lambda x: (x < 300).mean(),
        iet_pct_under_1hr   = lambda x: (x < 3600).mean(),
    )

    # Lag-1 autocorrelation — need at least 3 IETs
    def lag1_autocorr(x):
        if len(x) < 3:
            return np.nan
        return pd.Series(x.values).autocorr(lag=1)

    autocorr = iet.groupby("entity")["iet"].apply(lag1_autocorr).rename("iet_autocorr_lag1")

    result = (entity_label.to_frame()
              .join(n_txn)
              .join(p1)
              .join(autocorr)
              .reset_index())

    df.drop(columns=["iet"], inplace=True)  # cleanup
    print(f"    Done — {len(result):,} entities, {result.shape[1]} features")
    return result


# ── P2: Burst structure and active lifetime (vectorized) ──────────────────────

def compute_p2_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  [P2] Burst structure and active lifetime features...")

    # Active lifetime per entity
    span = df.groupby("entity")["timestamp"].agg(
        _first="min", _last="max"
    )
    span["active_lifetime_days"] = (span["_last"] - span["_first"]) / 86400
    al_df = span[["active_lifetime_days"]]

    # Burst features — for each delta, compute burst count and lengths
    # Strategy: IET-based — a new burst starts when gap > delta
    df["_iet"] = df.groupby("entity")["timestamp"].diff().fillna(0)

    burst_dfs = []
    for delta in BURST_DELTAS:
        suffix = f"{delta}s"
        # Mark burst starts: first txn of entity OR gap > delta
        is_first   = df.groupby("entity").cumcount() == 0
        new_burst  = is_first | (df["_iet"] > delta)
        df["_bid"] = new_burst.groupby(df["entity"]).cumsum()  # burst ID within entity

        burst_stats = (
            df.groupby(["entity", "_bid"])
            .size()
            .rename("_blen")
            .groupby(level="entity")
            .agg(
                **{f"burst_count_{suffix}":    "count"},
                **{f"burst_len_mean_{suffix}": "mean"},
                **{f"burst_len_max_{suffix}":  "max"},
            )
        )
        burst_dfs.append(burst_stats)

    df.drop(columns=["_iet", "_bid"], inplace=True)

    result = al_df.copy()
    for bdf in burst_dfs:
        result = result.join(bdf, how="left")

    # Burst density = burst_count / active_lifetime_days (bursts per day)
    # NaN for single-transaction entities (lifetime=0) — density undefined,
    # not a meaningful comparison point for fidelity scoring.
    for delta in BURST_DELTAS:
        suffix = f"{delta}s"
        al = result["active_lifetime_days"]
        result[f"burst_density_{suffix}"] = np.where(
            al > 0,
            result[f"burst_count_{suffix}"] / al,
            np.nan
        )

    result = result.reset_index()
    print(f"    Done — {len(result):,} entities, {result.shape[1]} features")
    return result


# ── P3: Shared-entity graph features (vectorized) ─────────────────────────────

def compute_p3_features(df: pd.DataFrame, device_cols: list) -> pd.DataFrame:
    entity_label = df.groupby("entity")["label"].max().rename("label").reset_index()

    if not device_cols:
        print("  [P3] Skipping — no device columns (IEEE-CIS limitation, see paper Section 4).")
        return entity_label

    print(f"  [P3] Graph features using: {device_cols}...")
    result = entity_label.copy()

    for col in device_cols:
        if col not in df.columns:
            continue
        sub = df[["entity", col, "label"]].dropna(subset=[col])

        # Fanout: how many distinct entities share each attribute value
        fanout = sub.groupby(col)["entity"].nunique().rename("_fanout")
        sub    = sub.merge(fanout.reset_index(), on=col, how="left")

        # Per-entity: n distinct values, max fanout encountered, high-fanout flag
        p3_ent = sub.groupby("entity").agg(
            **{f"n_distinct_{col}":        (col, "nunique")},
            **{f"max_fanout_{col}":        ("_fanout", "max")},
        ).reset_index()
        p3_ent[f"is_in_high_fanout_{col}"] = (p3_ent[f"max_fanout_{col}"] >= 5).astype(int)

        result = result.merge(p3_ent, on="entity", how="left")

    print(f"    Done — {len(result):,} entities, {result.shape[1]} features")
    return result


# ── P4: Velocity-rule trigger features (vectorized) ───────────────────────────

def compute_p4_features(df: pd.DataFrame, card_col: str = None) -> tuple:
    """
    Returns (entity_df, txn_df).
    Vectorized using pandas groupby + rolling where possible.
    Per-entity rolling windows are computed entity-by-entity but without
    Python-level transaction loops.
    """
    print("  [P4] Velocity rule features...")

    # We need per-transaction rolling features — do it per entity with pandas rolling
    # This is O(n) in transactions, not O(n*m) loops
    results = []

    for entity, grp in df.groupby("entity"):
        grp  = grp.reset_index(drop=True).copy()
        t    = grp["timestamp"].values
        amt  = grp["amount"].values if "amount" in grp else np.full(len(grp), np.nan)
        n    = len(grp)
        lbl  = int(grp["label"].max())
        t0   = t[0]  # entity first timestamp

        # Use pandas with timestamp as index for rolling
        tmp = pd.DataFrame({"t": t, "amt": amt})
        tmp.index = pd.to_datetime(t, unit="s")

        # R1: txn count in 1-hr rolling window
        cnt_1h = tmp["amt"].rolling("3600s", min_periods=1).count().values

        # R3: sum(amount) in 24-hr rolling window
        sum_24h = tmp["amt"].rolling("86400s", min_periods=1).sum().values

        # R4: ratio of current amount to 30-day rolling median
        # Rolling median isn't native in pandas; use expanding on 30-day window
        roll_med = tmp["amt"].rolling("2592000s", min_periods=1).median().shift(1).values

        # R5: distinct card values in 7-day rolling window (approximate)
        if card_col and card_col in grp.columns:
            # Approximate: count distinct in a sliding window using groupby trick
            card_vals = grp[card_col].values
            tmp["card"] = card_vals
            # For speed: use 7-day rolling nunique approximation
            tmp.index = pd.to_datetime(t, unit="s")
            # rolling nunique requires a custom approach
            n_pm_7d = np.ones(n, dtype=int)  # default
            for i in range(n):
                mask = (t >= t[i] - 604800) & (t <= t[i])
                n_pm_7d[i] = len(set(card_vals[mask]))
        else:
            n_pm_7d = np.full(n, np.nan)

        # Account age at each transaction
        age_days = (t - t0) / 86400

        txn_rows = {
            "entity":          [entity] * n,
            "txn_idx":         list(range(n)),
            "timestamp":       t,
            "label":           [lbl] * n,
            "r1_cnt_1hr":      cnt_1h,
            "r1_triggered":    (cnt_1h > 3).astype(int),
            "r2_new_acct":     ((age_days < 7) & (np.arange(n) > 0)).astype(int),
            "r3_sum_24hr":     sum_24h,
            "r3_triggered":    (sum_24h > 1000).astype(int),
            "r4_amt_ratio":    amt / np.where(roll_med > 0, roll_med, np.nan),
            "r5_n_pm_7d":      n_pm_7d,
            "r5_triggered":    (n_pm_7d > 2).astype(int) if not np.all(np.isnan(n_pm_7d)) else np.full(n, np.nan),
        }
        results.append(pd.DataFrame(txn_rows))

    txn_df = pd.concat(results, ignore_index=True)
    txn_df["r4_triggered"] = (txn_df["r4_amt_ratio"] > 3).astype("Int64")

    # Entity-level: ever triggered each rule?
    entity_df = txn_df.groupby("entity").agg(
        label    = ("label",       "max"),
        ever_r1  = ("r1_triggered","max"),
        ever_r2  = ("r2_new_acct", "max"),
        ever_r3  = ("r3_triggered","max"),
        ever_r4  = ("r4_triggered","max"),
        ever_r5  = ("r5_triggered","max"),
    ).reset_index()
    entity_df["n_rules_triggered"] = (
        entity_df[["ever_r1","ever_r2","ever_r3","ever_r4","ever_r5"]]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1)
    )

    print(f"    Done — {len(entity_df):,} entities, {len(txn_df):,} txn-level rows")
    return entity_df, txn_df


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    fraud = df[df["label"] == 1]
    legit = df[df["label"] == 0]
    print(f"\n{'─'*65}")
    print(f"BENCHMARK GROUND TRUTH  (fraud n={len(fraud):,} | legit n={len(legit):,})")
    print(f"{'─'*65}")

    checks = [
        ("iet_median",           "Median IET (s)"),
        ("iet_autocorr_lag1",    "IET lag-1 autocorr"),
        ("iet_pct_under_1hr",    "Frac IETs < 1hr"),
        ("active_lifetime_days", "Active lifetime (days)"),
        ("burst_count_3600s",    "Burst count (1hr delta)"),
        ("burst_len_max_3600s",  "Max burst len (1hr)"),
        ("ever_r1",              "R1 trigger rate"),
        ("ever_r3",              "R3 trigger rate"),
        ("n_rules_triggered",    "Avg rules triggered"),
    ]
    print(f"  {'Feature':33s}  {'Fraud':>10s}  {'Legit':>10s}  {'F/L ratio':>9s}")
    print(f"  {'-'*33}  {'-'*10}  {'-'*10}  {'-'*9}")
    for col, lbl in checks:
        if col not in df.columns:
            continue
        fv = fraud[col].mean()
        lv = legit[col].mean()
        ratio = fv / lv if lv and abs(lv) > 1e-9 else np.nan
        print(f"  {lbl:33s}  {fv:>10.3f}  {lv:>10.3f}  {ratio:>9.2f}x")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_benchmark(cfg: dict, data_dir: Path, out_dir: Path, dataset_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Building benchmark: {dataset_name}")
    print(f"{'='*60}")

    df = load_dataset(cfg, data_dir)

    p1         = compute_p1_features(df)
    p2         = compute_p2_features(df)
    p3         = compute_p3_features(df, cfg["device_cols"])
    p4_ent, p4_txn = compute_p4_features(df, cfg.get("card_col"))

    # Merge all entity features
    entity_df = (
        p1
        .merge(p2, on="entity", how="left")
        .merge(p3.drop(columns=["label"], errors="ignore"), on="entity", how="left")
        .merge(p4_ent.drop(columns=["label"], errors="ignore"), on="entity", how="left")
    )

    entity_path = out_dir / f"entity_behavioral_{dataset_name}.csv"
    txn_path    = out_dir / f"txn_behavioral_{dataset_name}.csv"
    entity_df.to_csv(entity_path, index=False)
    p4_txn.to_csv(txn_path, index=False)

    print(f"\nSaved:")
    print(f"  {entity_path.name}  ({len(entity_df):,} rows × {entity_df.shape[1]} cols)")
    print(f"  {txn_path.name}     ({len(p4_txn):,} rows)")

    print_summary(entity_df)
    return entity_df, p4_txn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  choices=list(DATASET_CONFIG.keys()), required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--out_dir",  type=Path, default=ROOT / "results" / "benchmark")
    args = parser.parse_args()

    cfg      = DATASET_CONFIG[args.dataset]
    data_dir = args.data_dir or ROOT / "data" / args.dataset
    build_benchmark(cfg, data_dir, args.out_dir, args.dataset)


if __name__ == "__main__":
    main()
