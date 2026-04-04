"""
Run behavioral fidelity evaluation for all generators on a given dataset.

Usage:
    python experiments/run_behavioral_eval.py \
        --dataset ieee_cis \
        --generators ctgan tvae copula \
        --syn_dir results/synthetic/

Expects synthetic CSVs named: {generator}_{dataset}_syn.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.behavioral_fidelity import BehavioralFidelityEvaluator

# ── Dataset-specific config ───────────────────────────────────────────────────

DATASET_CONFIG = {
    "ieee_cis": {
        "path": ROOT / "data" / "ieee_cis",
        "train_file": "train_transaction.csv",
        "id_file": "train_identity.csv",
        "entity_col": "entity_card1",    # set after EDA confirms this
        "time_col": "TransactionDT",
        "label_col": "isFraud",
        "amount_col": "TransactionAmt",
        "merchant_col": None,            # IEEE-CIS doesn't have clean merchant ID
        "card_col": "card4",             # card type (payment method proxy)
        "attr_cols": [],                 # filled after EDA confirms device columns
    },
    "amazon_fdb": {
        "path": ROOT / "data" / "amazon_fdb",
        "train_file": "train.csv",
        "id_file": None,
        "entity_col": "user_id",
        "time_col": "EVENT_TIMESTAMP",
        "label_col": "EVENT_LABEL",
        "amount_col": "order_price",
        "merchant_col": None,
        "card_col": "payment_type",
        "attr_cols": ["ip_address", "device_fingerprint"],
    },
}

ALL_GENERATORS = ["ctgan", "tvae", "copula", "tabddpm", "realtabformer"]


def load_dataset(cfg: dict) -> pd.DataFrame:
    data_path = cfg["path"]
    txn = pd.read_csv(data_path / cfg["train_file"])

    if cfg.get("id_file"):
        idn = pd.read_csv(data_path / cfg["id_file"])
        txn = txn.merge(idn, on="TransactionID", how="left")

    # Build entity column for IEEE-CIS
    if "entity_card1" in cfg.get("entity_col", ""):
        txn["entity_card1"] = txn["card1"].astype(str)

    return txn


def load_synthetic(syn_path: Path, generator: str, dataset: str) -> pd.DataFrame:
    candidates = [
        syn_path / f"{generator}_{dataset}_syn.csv",
        syn_path / f"{generator}_{dataset}.csv",
        syn_path / f"{dataset}_{generator}_syn.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Synthetic file not found. Expected one of:\n" +
        "\n".join(f"  {p}" for p in candidates)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()), required=True)
    parser.add_argument("--generators", nargs="+", default=ALL_GENERATORS)
    parser.add_argument("--syn_dir", type=Path, default=ROOT / "results" / "synthetic")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results" / "behavioral_fidelity")
    parser.add_argument("--skip", nargs="*", default=[], help="Patterns to skip, e.g. p3 p4")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_CONFIG[args.dataset]
    print(f"\nLoading real dataset: {args.dataset}")
    real_df = load_dataset(cfg)
    print(f"  Shape: {real_df.shape}  |  Fraud rate: {real_df[cfg['label_col']].mean():.4f}")

    evaluator = BehavioralFidelityEvaluator(
        entity_col=cfg["entity_col"],
        time_col=cfg["time_col"],
        label_col=cfg["label_col"],
        amount_col=cfg.get("amount_col"),
        merchant_col=cfg.get("merchant_col"),
        card_col=cfg.get("card_col"),
        attr_cols=cfg.get("attr_cols", []),
    )

    all_results = {}
    for gen in args.generators:
        print(f"\n{'='*60}")
        print(f"Generator: {gen.upper()}")
        try:
            syn_df = load_synthetic(args.syn_dir, gen, args.dataset)
            print(f"  Synthetic shape: {syn_df.shape}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        report = evaluator.evaluate_all(
            real_df, syn_df,
            generator_name=gen.upper(),
            dataset_name=args.dataset,
            skip=args.skip,
        )
        print(report.summary())

        # Save per-generator results
        result_dict = {
            "generator": gen,
            "dataset": args.dataset,
            "composite_score": report.composite_score,
        }
        if report.p1:
            result_dict["p1"] = {
                "w1_fraud": report.p1.wasserstein_fraud,
                "w1_legit": report.p1.wasserstein_legit,
                "ks_fraud": report.p1.ks_stat_fraud,
                "autocorr_gap": report.p1.autocorr_gap_fraud,
            }
        if report.p2:
            result_dict["p2"] = {
                "al_fraud": report.p2.wasserstein_active_lifetime_fraud,
                "al_legit": report.p2.wasserstein_active_lifetime_legit,
                "burst_len": report.p2.wasserstein_burst_len,
                "burst_count": report.p2.wasserstein_burst_count,
            }
        if report.p3:
            result_dict["p3"] = {
                "fanout": report.p3.fanout_wasserstein,
                "cc_delta": report.p3.clustering_coeff_delta,
                "tri_logratio": report.p3.triangle_log_ratio,
                "ccs_w1": report.p3.component_size_wasserstein,
            }
        if report.p4:
            result_dict["p4"] = {
                "mean_delta": report.p4.mean_absolute_delta,
                "per_rule": report.p4.per_rule_delta,
                "direction": report.p4.per_rule_direction,
            }
        if report.p5:
            result_dict["p5"] = {
                "shap_rank_corr": report.p5.shap_rank_correlation,
                "shap_w1_mean": report.p5.shap_wasserstein_mean,
                "top10_preserved": report.p5.top_features_preserved,
            }

        out_file = args.out_dir / f"{gen}_{args.dataset}_behavioral.json"
        with open(out_file, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"\n  Saved: {out_file}")
        all_results[gen] = result_dict

    # Summary table
    if all_results:
        print(f"\n{'='*60}")
        print("COMPOSITE BEHAVIORAL FIDELITY SCORES (lower = better)")
        print(f"{'='*60}")
        for gen, res in sorted(all_results.items(), key=lambda x: x[1].get("composite_score", 999)):
            score = res.get("composite_score", float("nan"))
            print(f"  {gen.upper():20s}  {score:.4f}")

        # Save combined results
        combined_path = args.out_dir / f"all_generators_{args.dataset}_behavioral.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results: {combined_path}")


if __name__ == "__main__":
    main()
