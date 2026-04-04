"""
Behavioral Fidelity Evaluation Module
=======================================
Implements behavioral fraud patterns P1–P4 from BEHAVIORAL_PATTERNS_FORMAL.md.

P5 (SHAP cross-feature interactions) is implemented but excluded from the
composite score and paper main results. It is retained as a callable function
for future work.

Composite score design
----------------------
Each sub-metric is expressed as a **degradation ratio** relative to a baseline:
    score = raw_metric / baseline_metric

A perfect generator (identical to real data) scores ≈ 1.0 on each sub-metric.
A generator that is 5× worse than the real-holdout baseline scores ≈ 5.0.
The composite is the equal-weighted mean of these ratios across P1–P4.

Baseline must be computed first by calling evaluate_all() on a random 50/50
split of the real data (set generator_name="BASELINE"). Pass the resulting
BehavioralFidelityReport as `baseline` to subsequent evaluate_all() calls.

Usage:
    from evaluation.behavioral_fidelity import BehavioralFidelityEvaluator

    ev = BehavioralFidelityEvaluator(
        entity_col="entity", time_col="timestamp",
        label_col="label", amount_col="amount",
    )
    # Step 1: establish baseline
    baseline = ev.evaluate_all(real_A, real_B, generator_name="BASELINE",
                               dataset_name="ieee_cis")
    # Step 2: evaluate generator
    report = ev.evaluate_all(real_df, syn_df, generator_name="CTGAN",
                             dataset_name="ieee_cis", baseline=baseline)
    print(report.summary())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Data classes for results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class P1Result:
    """Inter-Event Time Distribution results."""
    wasserstein_fraud: float       # W1(real IETD_F, syn IETD_F)
    wasserstein_legit: float       # W1(real IETD_N, syn IETD_N)
    ks_stat_fraud: float           # KS statistic for fraud class
    autocorr_gap_fraud: float      # |mean(rho_u^1) real - syn| for fraud entities
    n_fraud_iets_real: int
    n_fraud_iets_syn: int


@dataclass
class P2Result:
    """Burst Structure and Active Lifetime results."""
    # Keyed by delta_seconds e.g. 300, 3600, 21600
    wasserstein_burst_len: Dict[int, float] = field(default_factory=dict)
    wasserstein_burst_count: Dict[int, float] = field(default_factory=dict)
    wasserstein_active_lifetime_fraud: float = 0.0
    wasserstein_active_lifetime_legit: float = 0.0
    ks_active_lifetime_fraud: float = 0.0


@dataclass
class P3Result:
    """Shared-Entity Graph Motif results."""
    fanout_wasserstein: Dict[str, float] = field(default_factory=dict)  # per attr col
    clustering_coeff_real: float = 0.0
    clustering_coeff_syn: float = 0.0
    clustering_coeff_delta: float = 0.0
    triangle_log_ratio: float = 0.0
    component_size_wasserstein: float = 0.0
    graph_built: bool = False
    note: str = ""


@dataclass
class P4Result:
    """Velocity-Rule Trigger Rate results."""
    # per rule: abs difference in trigger rate (fraud class)
    per_rule_delta: Dict[str, float] = field(default_factory=dict)
    mean_absolute_delta: float = 0.0
    # direction: + means real triggers more, - means syn triggers more
    per_rule_direction: Dict[str, float] = field(default_factory=dict)


@dataclass
class P5Result:
    """Cross-Feature Interaction results."""
    shap_rank_correlation: float        # Spearman rho of feature importance rankings
    shap_wasserstein_mean: float        # mean W1 of SHAP value distributions per feature
    top_features_preserved: float       # fraction of top-10 features in common
    method: str = "shap"


@dataclass
class BehavioralFidelityReport:
    generator_name: str
    dataset_name: str
    p1: Optional[P1Result] = None
    p2: Optional[P2Result] = None
    p3: Optional[P3Result] = None
    p4: Optional[P4Result] = None
    # P5 excluded from main results; retained for future work
    composite_score: float = 0.0   # degradation ratio vs baseline (1.0 = perfect)
    # Raw sub-metric scores (before normalization)
    raw_scores: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Generator: {self.generator_name}  |  Dataset: {self.dataset_name}",
            f"{'='*60}",
        ]
        if self.p1:
            lines.append(f"P1 IETD   — W1_fraud: {self.p1.wasserstein_fraud:>12.1f}s  "
                         f"W1_legit: {self.p1.wasserstein_legit:>12.1f}s  "
                         f"autocorr_gap: {self.p1.autocorr_gap_fraud:.4f}")
        if self.p2:
            delta_keys = sorted(self.p2.wasserstein_burst_len.keys())
            bl_str = "  ".join(f"δ={d//60}min:{self.p2.wasserstein_burst_len[d]:.3f}" for d in delta_keys)
            lines.append(f"P2 Burst  — AL_fraud_W1: {self.p2.wasserstein_active_lifetime_fraud:>10.1f}s  "
                         f"BurstLen_W1({bl_str})")
        if self.p3:
            if self.p3.graph_built:
                fo_str = "  ".join(f"{k}:{v:.3f}" for k, v in self.p3.fanout_wasserstein.items())
                lines.append(f"P3 Graph  — fanout_W1({fo_str})  CC_delta: {self.p3.clustering_coeff_delta:.4f}  "
                             f"tri_logratio: {self.p3.triangle_log_ratio:.4f}")
            else:
                lines.append(f"P3 Graph  — {self.p3.note}")
        if self.p4:
            lines.append(f"P4 VR-TR  — mean |Δtrigger_rate|: {self.p4.mean_absolute_delta:.4f}")
            for r, d in self.p4.per_rule_delta.items():
                dir_str = "↑real" if self.p4.per_rule_direction.get(r, 0) > 0 else "↑syn"
                lines.append(f"            {r}: Δ={d:.4f} ({dir_str})")
        lines.append(f"{'─'*60}")
        if self.raw_scores:
            lines.append("Raw sub-scores (unnormalized):")
            for k, v in self.raw_scores.items():
                lines.append(f"  {k}: {v:.4f}")
        lines.append(f"Composite Score (degradation ratio, 1.0=perfect): {self.composite_score:.4f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core utilities
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_sample(arr: np.ndarray, target_n: int, seed: int = 42) -> np.ndarray:
    """Sample target_n items from arr with replacement if needed."""
    rng = np.random.default_rng(seed)
    if len(arr) >= target_n:
        return rng.choice(arr, size=target_n, replace=False)
    return rng.choice(arr, size=target_n, replace=True)


def _safe_wasserstein(a: np.ndarray, b: np.ndarray, max_n: int = 20_000) -> float:
    """Compute W1 with sample cap for speed."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a = _bootstrap_sample(a, min(max_n, len(a)))
    b = _bootstrap_sample(b, min(max_n, len(b)))
    return float(wasserstein_distance(a, b))


def _get_class_entities(df: pd.DataFrame, entity_col: str, label_col: str,
                         target_class: int) -> List:
    return df[df[label_col] == target_class][entity_col].unique().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1: Inter-Event Time Distribution
# ─────────────────────────────────────────────────────────────────────────────

def _compute_iet(df: pd.DataFrame, entity_col: str, time_col: str,
                 label_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (iet_fraud, iet_legit, autocorr_lag1_fraud).
    iet values are in the same units as time_col (seconds for TransactionDT).
    """
    df_s = df.sort_values([entity_col, time_col])
    df_s = df_s.dropna(subset=[entity_col, time_col])

    df_s["_iet"] = df_s.groupby(entity_col)[time_col].diff()
    iet_df = df_s.dropna(subset=["_iet"])

    fraud_mask = iet_df[label_col] == 1
    iet_fraud = iet_df.loc[fraud_mask, "_iet"].values
    iet_legit = iet_df.loc[~fraud_mask, "_iet"].values

    # Within-entity lag-1 autocorrelation for fraud entities
    autocorrs = []
    fraud_entities = iet_df.loc[fraud_mask, entity_col].unique()
    for ent in fraud_entities:
        seq = iet_df.loc[iet_df[entity_col] == ent, "_iet"].values
        if len(seq) >= 3:
            r, _ = stats.pearsonr(seq[:-1], seq[1:])
            if not np.isnan(r):
                autocorrs.append(r)

    autocorr_fraud = np.array(autocorrs) if autocorrs else np.array([0.0])
    return iet_fraud, iet_legit, autocorr_fraud


def compute_p1(real_df: pd.DataFrame, syn_df: pd.DataFrame,
               entity_col: str, time_col: str, label_col: str) -> P1Result:
    """Pattern 1: Inter-Event Time Distribution."""
    r_fraud, r_legit, r_ac = _compute_iet(real_df, entity_col, time_col, label_col)
    s_fraud, s_legit, s_ac = _compute_iet(syn_df, entity_col, time_col, label_col)

    w1_fraud = _safe_wasserstein(r_fraud, s_fraud)
    w1_legit = _safe_wasserstein(r_legit, s_legit)

    ks_stat, _ = ks_2samp(
        _bootstrap_sample(r_fraud, min(5000, len(r_fraud))),
        _bootstrap_sample(s_fraud, min(5000, len(s_fraud)))
    ) if len(r_fraud) > 0 and len(s_fraud) > 0 else (float("nan"), 1.0)

    autocorr_gap = abs(np.mean(r_ac) - np.mean(s_ac))

    return P1Result(
        wasserstein_fraud=w1_fraud,
        wasserstein_legit=w1_legit,
        ks_stat_fraud=ks_stat,
        autocorr_gap_fraud=autocorr_gap,
        n_fraud_iets_real=len(r_fraud),
        n_fraud_iets_syn=len(s_fraud),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2: Burst Structure and Active Lifetime
# ─────────────────────────────────────────────────────────────────────────────

def _compute_burst_stats(df: pd.DataFrame, entity_col: str, time_col: str,
                          label_col: str, delta: int
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (burst_lengths_fraud, burst_counts_fraud,
             active_lifetime_fraud, active_lifetime_legit).
    delta is in the same unit as time_col.
    """
    df_s = df.sort_values([entity_col, time_col]).dropna(subset=[entity_col, time_col])

    all_bl_fraud, all_bc_fraud = [], []
    al_fraud, al_legit = [], []

    for entity, grp in df_s.groupby(entity_col):
        times = grp[time_col].values
        is_fraud = grp[label_col].max()
        n = len(times)

        # Active lifetime
        al = float(times[-1] - times[0]) if n > 1 else 0.0
        if is_fraud:
            al_fraud.append(al)
        else:
            al_legit.append(al)

        if not is_fraud:
            continue  # Only compute burst stats for fraud entities

        # Compute bursts using gap threshold
        if n == 1:
            all_bc_fraud.append(1)
            all_bl_fraud.append(1)
            continue

        gaps = np.diff(times)
        burst_lengths = []
        current_len = 1
        for gap in gaps:
            if gap <= delta:
                current_len += 1
            else:
                burst_lengths.append(current_len)
                current_len = 1
        burst_lengths.append(current_len)

        all_bc_fraud.append(len(burst_lengths))
        all_bl_fraud.extend(burst_lengths)

    return (np.array(all_bl_fraud), np.array(all_bc_fraud),
            np.array(al_fraud), np.array(al_legit))


def compute_p2(real_df: pd.DataFrame, syn_df: pd.DataFrame,
               entity_col: str, time_col: str, label_col: str,
               deltas: List[int] = (300, 3600, 21600)) -> P2Result:
    """Pattern 2: Burst Structure and Active Lifetime."""
    result = P2Result()

    for delta in deltas:
        r_bl, r_bc, r_al_f, r_al_l = _compute_burst_stats(real_df, entity_col, time_col, label_col, delta)
        s_bl, s_bc, s_al_f, s_al_l = _compute_burst_stats(syn_df, entity_col, time_col, label_col, delta)

        result.wasserstein_burst_len[delta] = _safe_wasserstein(r_bl, s_bl)
        result.wasserstein_burst_count[delta] = _safe_wasserstein(r_bc, s_bc)

        if delta == deltas[0]:  # Compute AL once (delta-independent)
            result.wasserstein_active_lifetime_fraud = _safe_wasserstein(r_al_f, s_al_f)
            result.wasserstein_active_lifetime_legit = _safe_wasserstein(r_al_l, s_al_l)
            if len(r_al_f) > 0 and len(s_al_f) > 0:
                ks, _ = ks_2samp(
                    _bootstrap_sample(r_al_f, min(3000, len(r_al_f))),
                    _bootstrap_sample(s_al_f, min(3000, len(s_al_f)))
                )
                result.ks_active_lifetime_fraud = ks

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 3: Shared-Entity Graph Motifs
# ─────────────────────────────────────────────────────────────────────────────

def _build_bipartite_graph(df: pd.DataFrame, entity_col: str,
                            attr_col: str) -> nx.Graph:
    """Build entity-attribute bipartite graph."""
    G = nx.Graph()
    sub = df[[entity_col, attr_col]].dropna()
    for _, row in sub.iterrows():
        ent = f"E:{row[entity_col]}"
        att = f"A:{row[attr_col]}"
        G.add_node(ent, bipartite=0)
        G.add_node(att, bipartite=1)
        G.add_edge(ent, att)
    return G


def _fanout_distribution(G: nx.Graph) -> np.ndarray:
    """Fan-out = degree of attribute nodes (type A) in bipartite graph."""
    attr_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
    return np.array([G.degree(n) for n in attr_nodes])


def _project_entity_graph(G: nx.Graph) -> nx.Graph:
    """Project bipartite graph to entity-entity graph (co-occurrence via shared attr)."""
    entity_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}
    return nx.bipartite.projected_graph(G, entity_nodes)


def compute_p3(real_df: pd.DataFrame, syn_df: pd.DataFrame,
               entity_col: str, attr_cols: List[str],
               max_rows: int = 100_000,
               max_attr_cardinality: int = 200_000) -> P3Result:
    """Pattern 3: Shared-Entity Graph Motifs."""
    result = P3Result()

    if not attr_cols:
        result.note = "No usable attribute columns — skipped."
        return result

    # Sample for tractability
    r_sub = real_df.sample(min(max_rows, len(real_df)), random_state=42)
    s_sub = syn_df.sample(min(max_rows, len(syn_df)), random_state=42)

    for attr_col in attr_cols[:2]:  # max 2 attr cols
        if attr_col not in real_df.columns or attr_col not in syn_df.columns:
            continue
        # Skip extremely high-cardinality columns (likely free-form strings / UUIDs)
        if real_df[attr_col].nunique() > max_attr_cardinality:
            continue

        try:
            G_r = _build_bipartite_graph(r_sub, entity_col, attr_col)
            G_s = _build_bipartite_graph(s_sub, entity_col, attr_col)

            fo_r = _fanout_distribution(G_r)
            fo_s = _fanout_distribution(G_s)
            result.fanout_wasserstein[attr_col] = _safe_wasserstein(fo_r, fo_s)

            # Project to entity-entity graph
            GE_r = _project_entity_graph(G_r)
            GE_s = _project_entity_graph(G_s)

            # Clustering coefficient (sampled for speed)
            nodes_r = list(GE_r.nodes())[:2000]
            nodes_s = list(GE_s.nodes())[:2000]
            cc_r = nx.average_clustering(GE_r.subgraph(nodes_r)) if nodes_r else 0.0
            cc_s = nx.average_clustering(GE_s.subgraph(nodes_s)) if nodes_s else 0.0
            result.clustering_coeff_real = cc_r
            result.clustering_coeff_syn = cc_s
            result.clustering_coeff_delta = abs(cc_r - cc_s)

            # Triangle counts
            tri_r = sum(nx.triangles(GE_r).values()) // 3
            tri_s = sum(nx.triangles(GE_s).values()) // 3
            result.triangle_log_ratio = abs(np.log((tri_r + 1) / (tri_s + 1)))

            # Component size distribution
            ccs_r = np.array(sorted([len(c) for c in nx.connected_components(GE_r)], reverse=True))
            ccs_s = np.array(sorted([len(c) for c in nx.connected_components(GE_s)], reverse=True))
            result.component_size_wasserstein = _safe_wasserstein(ccs_r, ccs_s)

            result.graph_built = True
            break  # One attr col is enough for P3; add more if needed

        except Exception as e:
            result.note = f"Graph construction failed for {attr_col}: {e}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 4: Velocity-Rule Trigger Rates
# ─────────────────────────────────────────────────────────────────────────────

# Canonical rule set (8 rules from BEHAVIORAL_PATTERNS_FORMAL.md)
DEFAULT_RULES = {
    "R1_cnt_1hr":    {"feature": "count",     "threshold": 3,    "window_sec": 3600,   "op": ">"},
    "R2_merch_24hr": {"feature": "merchants",  "threshold": 5,    "window_sec": 86400,  "op": ">"},
    "R3_amt_24hr":   {"feature": "sum_amount", "threshold": 1000, "window_sec": 86400,  "op": ">"},
    "R5_pm_7d":      {"feature": "pm_changes", "threshold": 2,    "window_sec": 604800, "op": ">"},
    "R6_amt_spike":  {"feature": "amt_spike",  "threshold": 3.0,  "window_sec": None,   "op": ">"},
}


def _compute_velocity(df: pd.DataFrame, entity_col: str, time_col: str,
                       amount_col: Optional[str], merchant_col: Optional[str],
                       card_col: Optional[str]) -> pd.DataFrame:
    """Pre-compute velocity features for each transaction."""
    df_s = df.sort_values([entity_col, time_col]).copy()

    results = []
    for entity, grp in df_s.groupby(entity_col):
        grp = grp.reset_index(drop=True)
        times = grp[time_col].values
        n = len(grp)

        for i in range(n):
            row = {"entity": entity, "txn_idx": i}
            t = times[i]

            # 1-hour window transactions
            mask_1h = (times >= t - 3600) & (times <= t)
            row["cnt_1hr"] = mask_1h.sum()

            # 24-hour sum of amount
            mask_24h = (times >= t - 86400) & (times <= t)
            if amount_col and amount_col in grp.columns:
                row["sum_amt_24hr"] = grp.loc[mask_24h, amount_col].sum()
                # Amount spike: current / median of last 30 days
                mask_30d = (times >= t - 2592000) & (times < t)
                hist_amt = grp.loc[mask_30d, amount_col].values
                median_hist = np.median(hist_amt) if len(hist_amt) > 0 else float("nan")
                curr_amt = float(grp.loc[i, amount_col]) if amount_col in grp.columns else float("nan")
                row["amt_spike_ratio"] = (curr_amt / median_hist) if median_hist > 0 else float("nan")
            else:
                row["sum_amt_24hr"] = float("nan")
                row["amt_spike_ratio"] = float("nan")

            # 24-hour distinct merchants
            mask_24h_arr = np.where(mask_24h)[0]
            if merchant_col and merchant_col in grp.columns:
                row["distinct_merchants_24hr"] = grp.loc[mask_24h_arr, merchant_col].nunique()
            else:
                row["distinct_merchants_24hr"] = float("nan")

            # 7-day payment method changes (using card4 as proxy)
            mask_7d = (times >= t - 604800) & (times <= t)
            mask_7d_arr = np.where(mask_7d)[0]
            if card_col and card_col in grp.columns:
                row["distinct_pm_7d"] = grp.loc[mask_7d_arr, card_col].nunique()
            else:
                row["distinct_pm_7d"] = float("nan")

            results.append(row)

    return pd.DataFrame(results)


def _trigger_rate_fraud(velocity_df: pd.DataFrame, rule_key: str,
                         original_df: pd.DataFrame, entity_col: str,
                         label_col: str) -> float:
    """Fraction of fraud entities that trigger the rule at least once."""
    # Map entity -> is_fraud
    entity_fraud = original_df.groupby(entity_col)[label_col].max()

    if rule_key == "R1_cnt_1hr":
        triggered = velocity_df[velocity_df["cnt_1hr"] > 3]["entity"].unique()
    elif rule_key == "R2_merch_24hr":
        triggered = velocity_df[velocity_df["distinct_merchants_24hr"] > 5]["entity"].unique()
    elif rule_key == "R3_amt_24hr":
        triggered = velocity_df[velocity_df["sum_amt_24hr"] > 1000]["entity"].unique()
    elif rule_key == "R5_pm_7d":
        triggered = velocity_df[velocity_df["distinct_pm_7d"] > 2]["entity"].unique()
    elif rule_key == "R6_amt_spike":
        triggered = velocity_df[velocity_df["amt_spike_ratio"] > 3.0]["entity"].unique()
    else:
        return float("nan")

    fraud_entities = entity_fraud[entity_fraud == 1].index
    n_fraud_triggered = len(set(triggered) & set(fraud_entities))
    n_fraud_total = len(fraud_entities)
    return n_fraud_triggered / n_fraud_total if n_fraud_total > 0 else float("nan")


def compute_p4(real_df: pd.DataFrame, syn_df: pd.DataFrame,
               entity_col: str, time_col: str, label_col: str,
               amount_col: Optional[str] = None,
               merchant_col: Optional[str] = None,
               card_col: Optional[str] = None,
               rules: Optional[Dict] = None,
               max_entities: int = 5000) -> P4Result:
    """Pattern 4: Velocity-Rule Trigger Rates."""
    if rules is None:
        rules = DEFAULT_RULES

    result = P4Result()
    deltas = []

    # Sample entities for speed
    def sample_df(df):
        entities = df[entity_col].unique()
        if len(entities) > max_entities:
            sampled = np.random.default_rng(42).choice(entities, max_entities, replace=False)
            df = df[df[entity_col].isin(sampled)]
        return df

    r_sub = sample_df(real_df)
    s_sub = sample_df(syn_df)

    print("  Computing velocity features for real data...")
    vel_r = _compute_velocity(r_sub, entity_col, time_col, amount_col, merchant_col, card_col)
    print("  Computing velocity features for synthetic data...")
    vel_s = _compute_velocity(s_sub, entity_col, time_col, amount_col, merchant_col, card_col)

    for rule_key in rules:
        tr_real = _trigger_rate_fraud(vel_r, rule_key, r_sub, entity_col, label_col)
        tr_syn = _trigger_rate_fraud(vel_s, rule_key, s_sub, entity_col, label_col)
        delta = abs(tr_real - tr_syn) if not (np.isnan(tr_real) or np.isnan(tr_syn)) else float("nan")
        direction = (tr_real - tr_syn) if not np.isnan(delta) else 0.0
        result.per_rule_delta[rule_key] = delta
        result.per_rule_direction[rule_key] = direction
        if not np.isnan(delta):
            deltas.append(delta)

    result.mean_absolute_delta = float(np.mean(deltas)) if deltas else float("nan")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 5: Cross-Feature Interactions (SHAP-based)
# ─────────────────────────────────────────────────────────────────────────────

def compute_p5(real_df: pd.DataFrame, syn_df: pd.DataFrame,
               feature_cols: List[str], label_col: str,
               n_estimators: int = 100, max_sample: int = 30_000) -> P5Result:
    """Pattern 5: Cross-Feature Interactions via SHAP."""
    try:
        import shap
        import xgboost as xgb
    except ImportError:
        return P5Result(shap_rank_correlation=float("nan"),
                        shap_wasserstein_mean=float("nan"),
                        top_features_preserved=float("nan"),
                        method="shap_unavailable")

    # Keep numeric features only
    num_cols = [c for c in feature_cols
                if c in real_df.columns and real_df[c].dtype in [np.float64, np.float32,
                                                                    np.int64, np.int32, np.int16, np.int8]]
    if not num_cols:
        return P5Result(shap_rank_correlation=float("nan"),
                        shap_wasserstein_mean=float("nan"),
                        top_features_preserved=float("nan"),
                        method="no_numeric_features")

    def prep(df, cols):
        X = df[cols].fillna(-999).values
        y = df[label_col].values
        if len(X) > max_sample:
            idx = np.random.default_rng(42).choice(len(X), max_sample, replace=False)
            X, y = X[idx], y[idx]
        return X, y

    X_r, y_r = prep(real_df, num_cols)
    X_s, y_s = prep(syn_df, num_cols)

    model_params = dict(n_estimators=n_estimators, max_depth=6,
                        learning_rate=0.1, n_jobs=-1, random_state=42,
                        verbosity=0, eval_metric="logloss")

    m_r = xgb.XGBClassifier(**model_params)
    m_r.fit(X_r, y_r)

    m_s = xgb.XGBClassifier(**model_params)
    m_s.fit(X_s, y_s)

    # SHAP values
    explainer_r = shap.TreeExplainer(m_r)
    explainer_s = shap.TreeExplainer(m_s)
    shap_r = np.abs(explainer_r.shap_values(X_r))
    shap_s = np.abs(explainer_s.shap_values(X_s))

    # Mean absolute SHAP per feature
    mean_shap_r = shap_r.mean(axis=0)
    mean_shap_s = shap_s.mean(axis=0)

    # Rank correlation
    rank_r = stats.rankdata(-mean_shap_r)
    rank_s = stats.rankdata(-mean_shap_s)
    rho, _ = stats.spearmanr(rank_r, rank_s)

    # Per-feature Wasserstein
    w1_per_feat = [wasserstein_distance(shap_r[:, j], shap_s[:, j]) for j in range(len(num_cols))]
    mean_w1 = float(np.mean(w1_per_feat))

    # Top-10 feature overlap
    top10_r = set(np.argsort(-mean_shap_r)[:10])
    top10_s = set(np.argsort(-mean_shap_s)[:10])
    top10_preserved = len(top10_r & top10_s) / 10.0

    return P5Result(
        shap_rank_correlation=float(rho),
        shap_wasserstein_mean=mean_w1,
        top_features_preserved=top10_preserved,
        method="shap",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Composite Score
# ─────────────────────────────────────────────────────────────────────────────

def _extract_raw_scores(p1: P1Result, p2: P2Result,
                        p3: P3Result, p4: P4Result) -> Dict[str, float]:
    """
    Extract the primary scalar from each pattern result.
    All values are "higher = worse fidelity".
    Returns a dict keyed by pattern label.
    """
    scores: Dict[str, float] = {}
    if p1:
        scores["p1_w1_fraud"]  = p1.wasserstein_fraud
        scores["p1_autocorr"]  = p1.autocorr_gap_fraud
    if p2:
        scores["p2_al_fraud"]  = p2.wasserstein_active_lifetime_fraud
        bl_vals = list(p2.wasserstein_burst_len.values())
        scores["p2_burstlen"]  = float(np.mean(bl_vals)) if bl_vals else float("nan")
    if p3 and p3.graph_built:
        fo_vals = list(p3.fanout_wasserstein.values())
        scores["p3_fanout"]    = float(np.mean(fo_vals)) if fo_vals else float("nan")
    if p4:
        scores["p4_vrtrigger"] = p4.mean_absolute_delta
    return scores


def _composite(p1: P1Result, p2: P2Result, p3: P3Result, p4: P4Result,
               baseline_scores: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
    """
    Composite behavioral fidelity score.

    Returns (composite, raw_scores).

    If baseline_scores is provided (from a real-vs-real holdout run),
    each sub-metric is expressed as a degradation ratio:
        ratio = generator_score / baseline_score
    A ratio of 1.0 means the generator is as good as sampling real data.
    A ratio of 5.0 means 5× worse than expected from real-data sampling noise.

    If no baseline is provided, returns the unweighted mean of raw sub-metrics
    (not comparable across patterns — use only for the baseline run itself).
    """
    raw = _extract_raw_scores(p1, p2, p3, p4)

    if baseline_scores is None:
        # Baseline run: just return the raw mean (used to compute baseline_scores)
        valid = [v for v in raw.values() if not np.isnan(v)]
        composite = float(np.mean(valid)) if valid else float("nan")
        return composite, raw

    # Generator run: compute degradation ratio per sub-metric
    ratios = {}
    for key, val in raw.items():
        base = baseline_scores.get(key, float("nan"))
        if np.isnan(val) or np.isnan(base) or base == 0:
            ratios[key] = float("nan")
        else:
            ratios[key] = val / base

    valid_ratios = [v for v in ratios.values() if not np.isnan(v)]
    composite = float(np.mean(valid_ratios)) if valid_ratios else float("nan")
    return composite, ratios


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator class
# ─────────────────────────────────────────────────────────────────────────────

class BehavioralFidelityEvaluator:
    """
    Orchestrates all 5 pattern evaluations.

    Parameters
    ----------
    entity_col   : column identifying an entity (card, user, account)
    time_col     : column with transaction timestamp (numeric seconds preferred)
    label_col    : binary fraud label (1=fraud)
    amount_col   : transaction amount (optional, for P4 R3/R6)
    merchant_col : merchant ID (optional, for P4 R2)
    card_col     : payment method / card type (optional, for P4 R5)
    attr_cols    : columns usable for P3 graph (device, IP, etc.)
    feature_cols : all feature columns for P5 (defaults to all numeric)
    burst_deltas : list of burst gap thresholds in time_col units
    """

    def __init__(
        self,
        entity_col: str,
        time_col: str,
        label_col: str,
        amount_col: Optional[str] = None,
        merchant_col: Optional[str] = None,
        card_col: Optional[str] = None,
        attr_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
        burst_deltas: List[int] = (300, 3600, 21600),
    ):
        self.entity_col = entity_col
        self.time_col = time_col
        self.label_col = label_col
        self.amount_col = amount_col
        self.merchant_col = merchant_col
        self.card_col = card_col
        self.attr_cols = attr_cols or []
        self.feature_cols = feature_cols
        self.burst_deltas = list(burst_deltas)

    def evaluate_all(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        generator_name: str = "Unknown",
        dataset_name: str = "Unknown",
        skip: Optional[List[str]] = None,
        baseline: Optional["BehavioralFidelityReport"] = None,
    ) -> "BehavioralFidelityReport":
        """
        Run P1–P4 pattern evaluations and return a report.

        Parameters
        ----------
        real_df       : real transaction data
        syn_df        : synthetic (or held-out real) data to compare against
        generator_name: label for the generator being evaluated
        dataset_name  : label for the dataset
        skip          : list of pattern keys to skip, e.g. ["p3"]
        baseline      : BehavioralFidelityReport from a real-vs-real holdout run.
                        When provided, composite_score is a degradation ratio
                        (1.0 = generator matches real-data noise floor).
                        When None (baseline run itself), composite_score is the
                        raw mean — save this report as your baseline.
        """
        skip = skip or []
        report = BehavioralFidelityReport(
            generator_name=generator_name,
            dataset_name=dataset_name,
        )

        if "p1" not in skip:
            print(f"  [P1] Inter-event time distribution...")
            report.p1 = compute_p1(real_df, syn_df, self.entity_col, self.time_col, self.label_col)

        if "p2" not in skip:
            print(f"  [P2] Burst structure and active lifetime...")
            report.p2 = compute_p2(real_df, syn_df, self.entity_col, self.time_col,
                                    self.label_col, self.burst_deltas)

        if "p3" not in skip:
            print(f"  [P3] Shared-entity graph motifs...")
            report.p3 = compute_p3(real_df, syn_df, self.entity_col, self.attr_cols)

        if "p4" not in skip:
            print(f"  [P4] Velocity-rule trigger rates...")
            report.p4 = compute_p4(
                real_df, syn_df, self.entity_col, self.time_col, self.label_col,
                amount_col=self.amount_col,
                merchant_col=self.merchant_col,
                card_col=self.card_col,
            )

        baseline_scores = baseline.raw_scores if baseline is not None else None
        report.composite_score, report.raw_scores = _composite(
            report.p1, report.p2, report.p3, report.p4, baseline_scores
        )
        return report
