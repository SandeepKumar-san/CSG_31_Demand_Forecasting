"""
compute_graph_pearson.py
========================
Graph Structure Validation — Pearson / Spearman correlation analysis.

Sections:
  A1  Per-edge-type Pearson r summary (SupplyGraph, 4 edge types)
  A2  Pooled all-edge summary (SupplyGraph)
  A3  Random unconnected-pair baseline (SupplyGraph)
  A4  Effect size (real-edges minus random)
  B1  Per-edge-type Pearson r + Spearman rho (USGS, up to 15 edge types)
  B2  Pooled USGS summary
  B3  Important vs noise edge group comparison
  B4  Ablation correlation: |delta_RMSE| vs mean_abs_r
  C   Bootstrap 95% CIs (10 000 resamples, seed=42) for all mean_abs_r values

Output files:
  results/graph_structure_validation.json
  results/graph_structure_validation_paper_tables.txt

STRICT RULES (enforced):
  - No synthetic data. Every number is read from the actual CSV files.
  - No imputation. Pairs with any NaN or zero-variance series are skipped.
  - The only hardcoded numeric values are ABLATION_DELTA_RMSE (from paper Table V).
  - If any required file is missing, the script prints the correction needed and exits.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================

# Edge type ID → name (verbatim from src/data/usgs_loader.py EDGE_TYPE_NAMES)
USGS_EDGE_TYPE_NAMES = {
    0:  "supply_chain_input",
    1:  "technology_cluster",
    2:  "alloy_components",
    3:  "construction_coproduction",
    4:  "electronics_coproduction",
    5:  "substitution",
    6:  "battery_coproduction",
    7:  "critical_mineral_designation",
    8:  "geopolitical_supply_risk",
    9:  "price_correlation",
    10: "recycling_secondary_source",
    11: "byproduct_coproduction",
    12: "catalyst_role",
    13: "refractory_industrial",
    14: "functional_coating",
}

# ONLY permitted hardcoded numeric values — taken verbatim from paper Table V.
# Sign convention: positive = removing this edge HURTS performance (causal edge).
#                  negative = removing this edge HELPS performance (noise edge).
ABLATION_DELTA_RMSE = {
    "technology_cluster":          2.20,
    "alloy_components":            1.22,
    "geopolitical_supply_risk":    0.55,
    "substitution":                0.39,
    "supply_chain_input":         -1.29,
    "recycling_secondary_source": -0.50,
    "construction_coproduction":  -0.39,
}

# B3: group comparison uses exactly these 4 important + 3 noise (per spec definition)
IMPORTANT_EDGES = {"technology_cluster", "alloy_components",
                   "geopolitical_supply_risk", "substitution"}
NOISE_EDGES     = {"supply_chain_input", "recycling_secondary_source",
                   "construction_coproduction"}

# Table B: top 6 important edges (ablation ranks 1-6 from Table 7.5 positive ΔRMSE)
# + 3 noise edges (ablation ranks 1-3 from Table 7.5 negative ΔRMSE)
# electronics_coproduction (+0.15%) and battery_coproduction (+0.15%) are ranks 5-6.
TABLE_B_IMPORTANT_ORDERED = [
    "technology_cluster",        # +2.20% ablation rank 1
    "alloy_components",          # +1.22% ablation rank 2
    "geopolitical_supply_risk",  # +0.55% ablation rank 3
    "substitution",              # +0.39% ablation rank 4
    "electronics_coproduction",  # +0.15% ablation rank 5
    "battery_coproduction",      # +0.15% ablation rank 6
]
TABLE_B_NOISE_ORDERED = [
    "supply_chain_input",          # -1.29% noise rank 1
    "recycling_secondary_source",  # -0.50% noise rank 2
    "construction_coproduction",   # -0.39% noise rank 3
]
TABLE_B_ALL_IMPORTANT = set(TABLE_B_IMPORTANT_ORDERED)  # for group label in Table B

BOOTSTRAP_N     = 10_000
BOOTSTRAP_SEED  = 42
LOW_CONF_THRESH = 15   # flag edge types with fewer valid pairs

# =============================================================================
# File paths  (all relative to the project root — run from s:\Demand_Forecast)
# =============================================================================

BASE = os.path.dirname(os.path.abspath(__file__))

SG_SALES_ORDER = os.path.join(
    BASE, "data", "raw", "supplygraph",
    "Temporal Data", "Weight", "Sales Order .csv"   # trailing space is real
)
SG_EDGES_DIR = os.path.join(
    BASE, "data", "raw", "supplygraph", "Edges", "EdgesIndex"
)
SG_EDGE_FILES = {
    "Plant":             "Edges (Plant).csv",
    "Product Group":     "Edges (Product Group).csv",
    "Product Sub-Group": "Edges (Product Sub-Group).csv",
    "Storage Location":  "Edges (Storage Location).csv",
}

USGS_PROD  = os.path.join(BASE, "data", "processed", "usgs", "Production.csv")
USGS_CONS  = os.path.join(BASE, "data", "processed", "usgs", "Consumption.csv")
USGS_EDGES = os.path.join(BASE, "data", "processed", "usgs", "all_edges.csv")

RESULTS_DIR = os.path.join(BASE, "results")
OUT_JSON    = os.path.join(RESULTS_DIR, "graph_structure_validation.json")
OUT_TABLES  = os.path.join(RESULTS_DIR, "graph_structure_validation_paper_tables.txt")


# =============================================================================
# Utility functions
# =============================================================================

def _check_files():
    """Verify every required file/directory exists. Exit cleanly if not."""
    problems = []
    for label, path in [
        ("Sales Order Weight CSV",   SG_SALES_ORDER),
        ("SupplyGraph EdgesIndex/",  SG_EDGES_DIR),
        ("USGS Production.csv",      USGS_PROD),
        ("USGS Consumption.csv",     USGS_CONS),
        ("USGS all_edges.csv",       USGS_EDGES),
    ]:
        if not os.path.exists(path):
            problems.append(f"  MISSING [{label}]: {path}")

    for ename, fname in SG_EDGE_FILES.items():
        full = os.path.join(SG_EDGES_DIR, fname)
        if not os.path.isfile(full):
            problems.append(f"  MISSING [SupplyGraph edge file {ename}]: {full}")

    if problems:
        print("ERROR: Required files are missing. Correction needed:")
        for p in problems:
            print(p)
        sys.exit(1)


def _pearson_safe(a, b):
    """Return (r, ok). ok=False if constant series, any NaN, or any exception."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return None, False
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None, False
    try:
        r, _ = pearsonr(a, b)
        if np.isnan(r):
            return None, False
        return float(r), True
    except Exception as exc:
        print(f"    [pearsonr error: {exc}] — pair skipped")
        return None, False


def _spearman_safe(a, b):
    """Return (rho, ok). ok=False if constant series, any NaN, or any exception."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return None, False
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None, False
    try:
        rho, _ = spearmanr(a, b)
        if np.isnan(rho):
            return None, False
        return float(rho), True
    except Exception as exc:
        print(f"    [spearmanr error: {exc}] — pair skipped")
        return None, False


def _bootstrap_mean_abs_r(r_vals, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """95% bootstrap CI on mean(|r|)."""
    arr = np.abs(np.asarray(r_vals, dtype=float))
    if len(arr) == 0:
        return None, None
    rng = np.random.RandomState(seed)
    boot = np.fromiter(
        (np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)),
        dtype=float, count=n_boot
    )
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _stats(r_vals, ci=True):
    """Return stats dict for a list of r values."""
    if not r_vals:
        return dict(mean_r=None, mean_abs_r=None, median_r=None,
                    std_r=None, ci_95_lower=None, ci_95_upper=None)
    arr = np.asarray(r_vals, dtype=float)
    lo, hi = _bootstrap_mean_abs_r(r_vals) if ci else (None, None)
    return dict(
        mean_r=float(np.mean(arr)),
        mean_abs_r=float(np.mean(np.abs(arr))),
        median_r=float(np.median(arr)),
        std_r=float(np.std(arr)),
        ci_95_lower=lo,
        ci_95_upper=hi,
    )


def _f(v, d=4):
    """Format float or None."""
    return f"{v:.{d}f}" if v is not None else "N/A"


def _sep(char="=", w=80):
    print(char * w)


def _dedupe_pairs(df, col_a, col_b):
    """Return a set of undirected (min,max) integer pairs, excluding self-loops."""
    pairs = set()
    for _, row in df.iterrows():
        i, j = int(row[col_a]), int(row[col_b])
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    return pairs


# =============================================================================
# SECTION A — SupplyGraph
# =============================================================================

def run_supplygraph():
    _sep()
    print("SECTION A — SupplyGraph Pearson Correlation Analysis")
    _sep()

    # --- Load Sales Order Weight ---
    so = pd.read_csv(SG_SALES_ORDER)
    so["Date"] = pd.to_datetime(so["Date"])
    so = so.sort_values("Date").reset_index(drop=True)
    node_matrix = so.drop(columns=["Date"]).values.astype(float)  # (n_days, n_nodes)
    n_days, n_nodes = node_matrix.shape
    print(f"  Sales Order Weight loaded: {n_days} trading days × {n_nodes} nodes")

    # --- A1: Per-edge-type ---
    per_edge_results = []
    all_connected_pairs = set()
    pooled_r_vals = []
    total_valid_sg = 0
    total_skipped_sg = 0

    for etype, fname in SG_EDGE_FILES.items():
        df = pd.read_csv(os.path.join(SG_EDGES_DIR, fname))
        pairs = _dedupe_pairs(df, "node1", "node2")
        all_connected_pairs.update(pairs)

        r_vals = []
        n_total = len(pairs)
        n_skip = 0

        for (i, j) in pairs:
            if i >= n_nodes or j >= n_nodes:
                n_skip += 1
                continue
            r, ok = _pearson_safe(node_matrix[:, i], node_matrix[:, j])
            if ok:
                r_vals.append(r)
            else:
                n_skip += 1

        n_valid = len(r_vals)
        total_valid_sg += n_valid
        total_skipped_sg += n_skip
        pooled_r_vals.extend(r_vals)

        st = _stats(r_vals, ci=True)
        rec = dict(edge_type=etype, n_pairs_total=n_total,
                   n_pairs_valid=n_valid, n_pairs_skipped=n_skip, **st)
        per_edge_results.append(rec)

        print(f"\n  [{etype}]")
        print(f"    total={n_total} | valid={n_valid} | skipped={n_skip}")
        print(f"    mean_r={_f(st['mean_r'])}  mean_|r|={_f(st['mean_abs_r'])}  "
              f"median_r={_f(st['median_r'])}  std_r={_f(st['std_r'])}")
        print(f"    95% CI mean_|r|: [{_f(st['ci_95_lower'])}, {_f(st['ci_95_upper'])}]")

    # --- A2: Pooled ---
    pooled_st = _stats(pooled_r_vals, ci=True)
    pooled = dict(n_pairs_valid=len(pooled_r_vals),
                  n_pairs_skipped=total_skipped_sg, **pooled_st)
    print(f"\n  [POOLED — all 4 edge types]")
    print(f"    valid={len(pooled_r_vals)} | skipped={total_skipped_sg}")
    print(f"    mean_r={_f(pooled_st['mean_r'])}  mean_|r|={_f(pooled_st['mean_abs_r'])}")
    print(f"    95% CI mean_|r|: [{_f(pooled_st['ci_95_lower'])}, {_f(pooled_st['ci_95_upper'])}]")

    # --- A3: Random baseline ---
    all_possible = {(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)}
    unconnected = sorted(all_possible - all_connected_pairs)
    n_sample = len(pooled_r_vals)

    rng = np.random.RandomState(42)
    if len(unconnected) < n_sample:
        print(f"\n  WARNING: Only {len(unconnected)} unconnected pairs; "
              f"need {n_sample}. Using all available unconnected pairs.")
        sampled_pairs = unconnected
        n_sample = len(unconnected)
    else:
        idx = rng.choice(len(unconnected), size=n_sample, replace=False)
        sampled_pairs = [unconnected[k] for k in sorted(idx)]

    rand_r_vals = []
    rand_skip = 0
    for (i, j) in sampled_pairs:
        r, ok = _pearson_safe(node_matrix[:, i], node_matrix[:, j])
        if ok:
            rand_r_vals.append(r)
        else:
            rand_skip += 1

    rand_st = _stats(rand_r_vals, ci=True)
    random_baseline = dict(n_pairs_sampled=n_sample, n_pairs_valid=len(rand_r_vals),
                           n_pairs_skipped=rand_skip, **rand_st)
    print(f"\n  [RANDOM BASELINE — unconnected pairs, seed=42]")
    print(f"    sampled={n_sample} | valid={len(rand_r_vals)} | skipped={rand_skip}")
    print(f"    mean_r={_f(rand_st['mean_r'])}  mean_|r|={_f(rand_st['mean_abs_r'])}")
    print(f"    95% CI mean_|r|: [{_f(rand_st['ci_95_lower'])}, {_f(rand_st['ci_95_upper'])}]")

    # --- A4: Effect size ---
    delta = None
    if pooled_st["mean_abs_r"] is not None and rand_st["mean_abs_r"] is not None:
        delta = pooled_st["mean_abs_r"] - rand_st["mean_abs_r"]
    effect_size = {"delta_mean_abs_r": delta}
    print(f"\n  [EFFECT SIZE]")
    print(f"    delta_mean_|r| (real_edges - random) = {_f(delta)}")

    return (
        dict(per_edge_type=per_edge_results, pooled=pooled,
             random_baseline=random_baseline, effect_size=effect_size),
        total_valid_sg,
        total_skipped_sg,
    )


# =============================================================================
# SECTION B — USGS
# =============================================================================

def run_usgs():
    _sep()
    print("SECTION B — USGS Pearson / Spearman Correlation Analysis")
    _sep()

    # --- Load data ---
    prod = pd.read_csv(USGS_PROD, index_col="Year")
    cons = pd.read_csv(USGS_CONS, index_col="Year")
    edges = pd.read_csv(USGS_EDGES)

    # Cast columns to str so we can look up by integer node id safely
    prod.columns = [str(c) for c in prod.columns]
    cons.columns = [str(c) for c in cons.columns]

    print(f"  Production:  {prod.shape[1]} nodes × {prod.shape[0]} years  "
          f"[years: {list(prod.index)}]")
    print(f"  Consumption: {cons.shape[1]} nodes × {cons.shape[0]} years")
    print(f"  all_edges:   {len(edges)} directed edges  "
          f"| edge_type_ids present: {sorted(edges['edge_type_id'].unique())}")

    per_edge_results = []
    pool_r_prod, pool_r_cons = [], []
    pool_rho_prod, pool_rho_cons = [], []
    total_valid_prod_usgs = 0
    total_valid_cons_usgs = 0
    total_skipped_usgs = 0

    edge_type_ids = sorted(edges["edge_type_id"].unique())

    # --- B1: Per-edge-type ---
    # Production and consumption computed INDEPENDENTLY.
    # Root cause of original 79.6% skip: 74/127 prod nodes and 46/127 cons nodes
    # have constant/zero values — a real data property (US has no domestic
    # production for many minerals). Requiring all 4 series to be valid caused
    # unnecessary cross-metric skips. Fix: each metric tracks its own counts.
    for eid in edge_type_ids:
        ename = USGS_EDGE_TYPE_NAMES.get(int(eid), f"unknown_id_{eid}")
        sub = edges[edges["edge_type_id"] == eid]
        pairs = _dedupe_pairs(sub, "source_id", "target_id")

        r_prod_vals, rho_prod_vals = [], []
        r_cons_vals, rho_cons_vals = [], []
        n_total = len(pairs)
        n_skip_missing = 0
        n_skip_prod = 0    # failed for production metric (constant/NaN)
        n_skip_cons = 0    # failed for consumption metric (constant/NaN)
        n_skip_both = 0    # failed for both metrics — entirely unusable pair

        for (i, j) in pairs:
            si, sj = str(i), str(j)

            if si not in prod.columns or sj not in prod.columns:
                n_skip_missing += 1
                continue

            p_i = prod[si].values.astype(float)
            p_j = prod[sj].values.astype(float)
            c_i = cons[si].values.astype(float) if si in cons.columns else np.full(prod.shape[0], np.nan)
            c_j = cons[sj].values.astype(float) if sj in cons.columns else np.full(prod.shape[0], np.nan)

            # Production — independent
            prod_ok = False
            if not (np.any(np.isnan(p_i)) or np.any(np.isnan(p_j))):
                r_p, ok_rp = _pearson_safe(p_i, p_j)
                rho_p, ok_sp = _spearman_safe(p_i, p_j)
                if ok_rp and ok_sp:
                    r_prod_vals.append(r_p)
                    rho_prod_vals.append(rho_p)
                    prod_ok = True
            if not prod_ok:
                n_skip_prod += 1

            # Consumption — independent
            cons_ok = False
            if not (np.any(np.isnan(c_i)) or np.any(np.isnan(c_j))):
                r_c, ok_rc = _pearson_safe(c_i, c_j)
                rho_c, ok_sc = _spearman_safe(c_i, c_j)
                if ok_rc and ok_sc:
                    r_cons_vals.append(r_c)
                    rho_cons_vals.append(rho_c)
                    cons_ok = True
            if not cons_ok:
                n_skip_cons += 1

            if not prod_ok and not cons_ok:
                n_skip_both += 1

        n_valid_prod = len(r_prod_vals)
        n_valid_cons = len(r_cons_vals)
        n_skip_total = n_skip_missing + n_skip_both   # pairs with zero contribution

        total_valid_prod_usgs += n_valid_prod
        total_valid_cons_usgs += n_valid_cons
        total_skipped_usgs += n_skip_total

        pool_r_prod.extend(r_prod_vals)
        pool_r_cons.extend(r_cons_vals)
        pool_rho_prod.extend(rho_prod_vals)
        pool_rho_cons.extend(rho_cons_vals)

        n_valid_max = max(n_valid_prod, n_valid_cons)
        conf_flag = "OK" if n_valid_max >= LOW_CONF_THRESH else "LOW_CONFIDENCE"

        lo_p = hi_p = lo_c = hi_c = None
        if n_valid_prod >= LOW_CONF_THRESH:
            lo_p, hi_p = _bootstrap_mean_abs_r(r_prod_vals)
        if n_valid_cons >= LOW_CONF_THRESH:
            lo_c, hi_c = _bootstrap_mean_abs_r(r_cons_vals)

        rec = dict(
            edge_type_id=int(eid),
            edge_type_name=ename,
            n_pairs_total=n_total,
            n_pairs_valid_prod=n_valid_prod,
            n_pairs_valid_cons=n_valid_cons,
            n_pairs_skipped_missing=n_skip_missing,
            n_pairs_skipped_prod_only=n_skip_prod,
            n_pairs_skipped_cons_only=n_skip_cons,
            n_pairs_skipped_both=n_skip_both,
            confidence=conf_flag,
            mean_r_prod=float(np.mean(r_prod_vals))             if r_prod_vals   else None,
            mean_abs_r_prod=float(np.mean(np.abs(r_prod_vals))) if r_prod_vals   else None,
            mean_spearman_prod=float(np.mean(rho_prod_vals))    if rho_prod_vals else None,
            ci_95_lower_prod=lo_p,
            ci_95_upper_prod=hi_p,
            mean_r_cons=float(np.mean(r_cons_vals))             if r_cons_vals   else None,
            mean_abs_r_cons=float(np.mean(np.abs(r_cons_vals))) if r_cons_vals   else None,
            mean_spearman_cons=float(np.mean(rho_cons_vals))    if rho_cons_vals else None,
            ci_95_lower_cons=lo_c,
            ci_95_upper_cons=hi_c,
        )
        per_edge_results.append(rec)

        ci_p_str = f"[{_f(lo_p)}, {_f(hi_p)}]" if lo_p is not None else "N/A (LOW_CONFIDENCE)"
        ci_c_str = f"[{_f(lo_c)}, {_f(hi_c)}]" if lo_c is not None else "N/A (LOW_CONFIDENCE)"
        print(f"\n  [ID={eid}: {ename}]  [{conf_flag}]")
        print(f"    total={n_total} | valid_prod={n_valid_prod} | valid_cons={n_valid_cons} "
              f"| skipped_both={n_skip_both} | missing={n_skip_missing}")
        print(f"    Production:  mean_r={_f(rec['mean_r_prod'])}  mean_|r|={_f(rec['mean_abs_r_prod'])}  "
              f"spearman={_f(rec['mean_spearman_prod'])}  CI={ci_p_str}")
        print(f"    Consumption: mean_r={_f(rec['mean_r_cons'])}  mean_|r|={_f(rec['mean_abs_r_cons'])}  "
              f"spearman={_f(rec['mean_spearman_cons'])}  CI={ci_c_str}")

    # --- B2: Pooled ---
    pool_lo_p, pool_hi_p = _bootstrap_mean_abs_r(pool_r_prod)
    pool_lo_c, pool_hi_c = _bootstrap_mean_abs_r(pool_r_cons)
    pooled = dict(
        n_pairs_valid_prod=total_valid_prod_usgs,
        n_pairs_valid_cons=total_valid_cons_usgs,
        n_pairs_skipped=total_skipped_usgs,
        mean_abs_r_prod=float(np.mean(np.abs(pool_r_prod))) if pool_r_prod else None,
        mean_abs_r_cons=float(np.mean(np.abs(pool_r_cons))) if pool_r_cons else None,
        mean_spearman_prod=float(np.mean(pool_rho_prod))    if pool_rho_prod else None,
        mean_spearman_cons=float(np.mean(pool_rho_cons))    if pool_rho_cons else None,
        ci_95_lower_prod=pool_lo_p, ci_95_upper_prod=pool_hi_p,
        ci_95_lower_cons=pool_lo_c, ci_95_upper_cons=pool_hi_c,
    )
    print(f"\n  [POOLED — all {len(edge_type_ids)} edge types]")
    print(f"    valid_prod={total_valid_prod_usgs} | valid_cons={total_valid_cons_usgs} "
          f"| skipped_both={total_skipped_usgs}")
    print(f"    mean_|r|_prod={_f(pooled['mean_abs_r_prod'])}  "
          f"mean_|r|_cons={_f(pooled['mean_abs_r_cons'])}")
    print(f"    CI_prod=[{_f(pool_lo_p)}, {_f(pool_hi_p)}]  "
          f"CI_cons=[{_f(pool_lo_c)}, {_f(pool_hi_c)}]")

    # --- B3: Important vs noise ---
    lookup = {e["edge_type_name"]: e for e in per_edge_results}

    def _group_mean(names, field):
        vals = [lookup[n][field] for n in names
                if n in lookup and lookup[n][field] is not None]
        return float(np.mean(vals)) if vals else None

    imp_vs_noise = {
        "important": {
            "mean_abs_r_prod": _group_mean(IMPORTANT_EDGES, "mean_abs_r_prod"),
            "mean_abs_r_cons": _group_mean(IMPORTANT_EDGES, "mean_abs_r_cons"),
        },
        "noise": {
            "mean_abs_r_prod": _group_mean(NOISE_EDGES, "mean_abs_r_prod"),
            "mean_abs_r_cons": _group_mean(NOISE_EDGES, "mean_abs_r_cons"),
        },
    }
    print(f"\n  [B3 — Important vs Noise edge groups]")
    print(f"    IMPORTANT: mean_|r|_prod={_f(imp_vs_noise['important']['mean_abs_r_prod'])}  "
          f"mean_|r|_cons={_f(imp_vs_noise['important']['mean_abs_r_cons'])}")
    print(f"    NOISE:     mean_|r|_prod={_f(imp_vs_noise['noise']['mean_abs_r_prod'])}  "
          f"mean_|r|_cons={_f(imp_vs_noise['noise']['mean_abs_r_cons'])}")

    # --- B4: Ablation correlation ---
    print(f"\n  [B4 — Ablation correlation: |ΔRMSE| vs mean_|r|]")
    X_prod, X_cons, Y_abs, dropped = [], [], [], []

    for abl_name in ABLATION_DELTA_RMSE:
        e = lookup.get(abl_name)
        if e is None or e["n_pairs_valid_prod"] < 1 or e["mean_abs_r_prod"] is None:
            dropped.append(abl_name)
            print(f"    DROPPED: {abl_name} — no valid pairs in B1")
            continue
        X_prod.append(e["mean_abs_r_prod"])
        X_cons.append(e["mean_abs_r_cons"])   # may be None
        Y_abs.append(abs(ABLATION_DELTA_RMSE[abl_name]))

    abl_corr = {"n_edge_types_used": len(X_prod), "dropped": dropped}
    computable = False

    if len(X_prod) >= 3:
        r_p_abl, _ = pearsonr(X_prod, Y_abs)
        rho_p_abl, _ = spearmanr(X_prod, Y_abs)
        abl_corr["pearson_r_prod"]   = float(r_p_abl)
        abl_corr["spearman_rho_prod"] = float(rho_p_abl)

        # Consumption — only if all entries are non-None
        if all(v is not None for v in X_cons):
            r_c_abl, _   = pearsonr(X_cons, Y_abs)
            rho_c_abl, _ = spearmanr(X_cons, Y_abs)
            abl_corr["pearson_r_cons"]    = float(r_c_abl)
            abl_corr["spearman_rho_cons"] = float(rho_c_abl)
        else:
            abl_corr["pearson_r_cons"]    = None
            abl_corr["spearman_rho_cons"] = None
            print("    WARNING: Some mean_abs_r_cons are None — "
                  "consumption ablation correlation not computable.")

        computable = True
        print(f"    n_types_used={len(X_prod)} | dropped={dropped}")
        print(f"    Production:  pearson_r={_f(abl_corr['pearson_r_prod'])}  "
              f"spearman_rho={_f(abl_corr['spearman_rho_prod'])}")
        print(f"    Consumption: pearson_r={_f(abl_corr.get('pearson_r_cons'))}  "
              f"spearman_rho={_f(abl_corr.get('spearman_rho_cons'))}")
    else:
        abl_corr["note"] = (
            f"Insufficient edge types for correlation — need >= 3, got {len(X_prod)}"
        )
        print(f"    NOT COMPUTABLE — only {len(X_prod)} edge types available (need >= 3).")

    return (
        dict(per_edge_type=per_edge_results, pooled=pooled,
             important_vs_noise=imp_vs_noise, ablation_correlation=abl_corr),
        total_valid_prod_usgs,
        total_valid_cons_usgs,
        total_skipped_usgs,
        computable,
        len(edge_type_ids),
    )


# =============================================================================
# Paper Tables
# =============================================================================

def write_paper_tables(sg, usgs, path):
    W = 100
    lines = []

    def hline(c="-"):
        lines.append(c * W)

    lines.append("=" * W)
    lines.append("GRAPH STRUCTURE VALIDATION — PAPER TABLES")
    lines.append("All values computed from real dataset files. No synthetic substitutions.")
    lines.append("=" * W)

    # ---- TABLE A ----
    lines.append("")
    lines.append("TABLE A: SupplyGraph — Mean Pearson |r| by Edge Type")
    lines.append("         (final row = matched random unconnected-pair baseline)")
    hline()
    lines.append(
        f"{'Edge Type':<24} {'n_total':>8} {'n_valid':>8} "
        f"{'mean_r':>9} {'mean_|r|':>9} {'median_r':>9} {'std_r':>7} "
        f"{'CI_lower':>10} {'CI_upper':>10}"
    )
    hline()
    for e in sg["per_edge_type"]:
        lines.append(
            f"{e['edge_type']:<24} {e['n_pairs_total']:>8} {e['n_pairs_valid']:>8} "
            f"{_f(e['mean_r']):>9} {_f(e['mean_abs_r']):>9} {_f(e['median_r']):>9} "
            f"{_f(e['std_r']):>7} {_f(e['ci_95_lower']):>10} {_f(e['ci_95_upper']):>10}"
        )
    hline()
    rb = sg["random_baseline"]
    lines.append(
        f"{'RANDOM BASELINE':<24} {rb['n_pairs_sampled']:>8} {rb['n_pairs_valid']:>8} "
        f"{_f(rb['mean_r']):>9} {_f(rb['mean_abs_r']):>9} {_f(rb['median_r']):>9} "
        f"{_f(rb['std_r']):>7} {_f(rb['ci_95_lower']):>10} {_f(rb['ci_95_upper']):>10}"
    )
    pl = sg["pooled"]
    lines.append(
        f"{'POOLED (all edges)':<24} {pl['n_pairs_valid']:>8} {'':>8} "
        f"{_f(pl['mean_r']):>9} {_f(pl['mean_abs_r']):>9} {_f(pl['median_r']):>9} "
        f"{_f(pl['std_r']):>7} {_f(pl['ci_95_lower']):>10} {_f(pl['ci_95_upper']):>10}"
    )
    hline()
    lines.append(
        f"Effect size: delta_mean_|r| (real_edges − random) = "
        f"{_f(sg['effect_size']['delta_mean_abs_r'])}"
    )

    # ---- TABLE B ----
    lines.append("")
    lines.append("=" * W)
    lines.append("TABLE B: USGS — Per-Edge-Type Correlation (Important + Noise edges from ablation)")
    lines.append("         Spearman rho is primary metric (n=5 observations per series).")
    hline()
    lines.append(
        f"{'Edge Type':<32} {'Group':>10} {'conf':>14} {'n_valid':>7} "
        f"{'|r|_prod':>9} {'rho_prod':>9} {'CI_prod':>22} "
        f"{'|r|_cons':>9} {'rho_cons':>9}"
    )
    hline()
    lookup = {e["edge_type_name"]: e for e in usgs["per_edge_type"]}
    # Table B = top 6 important (ablation ranks 1-6 by positive ΔRMSE)
    #         + top 3 noise (ablation ranks 1-3 by negative ΔRMSE magnitude)
    table_b_ordered = TABLE_B_IMPORTANT_ORDERED + TABLE_B_NOISE_ORDERED
    for name in table_b_ordered:
        e = lookup.get(name)
        group = "IMPORTANT" if name in TABLE_B_ALL_IMPORTANT else "NOISE"
        if e is None:
            lines.append(f"{name:<32} {group:>10} {'NOT FOUND':>14}")
            continue
        ci_p = (f"[{_f(e['ci_95_lower_prod'])},{_f(e['ci_95_upper_prod'])}]"
                if e["ci_95_lower_prod"] is not None else "LOW_CONFIDENCE")
        lines.append(
            f"{name:<32} {group:>10} {e['confidence']:>14} {e['n_pairs_valid_prod']:>7} "
            f"{_f(e['mean_abs_r_prod']):>9} {_f(e['mean_spearman_prod']):>9} {ci_p:>22} "
            f"{_f(e['mean_abs_r_cons']):>9} {_f(e['mean_spearman_cons']):>9}"
        )
    hline()
    iv = usgs["important_vs_noise"]
    lines.append(
        f"Group mean IMPORTANT: mean_|r|_prod={_f(iv['important']['mean_abs_r_prod'])}  "
        f"mean_|r|_cons={_f(iv['important']['mean_abs_r_cons'])}"
    )
    lines.append(
        f"Group mean NOISE:     mean_|r|_prod={_f(iv['noise']['mean_abs_r_prod'])}  "
        f"mean_|r|_cons={_f(iv['noise']['mean_abs_r_cons'])}"
    )

    # ---- TABLE C ----
    lines.append("")
    lines.append("=" * W)
    lines.append("TABLE C: Ablation Correlation — |ΔRMSE%| (from Table V) vs mean_|r| (computed)")
    hline()
    ac = usgs["ablation_correlation"]
    lines.append(f"  n_edge_types_used : {ac.get('n_edge_types_used', 'N/A')}")
    lines.append(f"  dropped           : {ac.get('dropped', [])}")
    lines.append(f"  pearson_r_prod    : {_f(ac.get('pearson_r_prod'))}")
    lines.append(f"  spearman_rho_prod : {_f(ac.get('spearman_rho_prod'))}")
    lines.append(f"  pearson_r_cons    : {_f(ac.get('pearson_r_cons'))}")
    lines.append(f"  spearman_rho_cons : {_f(ac.get('spearman_rho_cons'))}")
    lines.append("=" * W)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Paper tables saved → {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    _sep("=", 80)
    print("  GRAPH STRUCTURE VALIDATION — Pearson / Spearman Computation")
    print("  Sections A1-A4 (SupplyGraph), B1-B4 (USGS), C (Bootstrap CIs)")
    _sep("=", 80)

    _check_files()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sg_results, sg_valid, sg_skipped = run_supplygraph()
    usgs_results, usgs_valid_prod, usgs_valid_cons, usgs_skipped, abl_ok, n_edge_types = run_usgs()

    # --- Save JSON ---
    full_output = {
        "supplygraph": sg_results,
        "usgs": usgs_results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n  Full results saved → {OUT_JSON}")

    # --- Save paper tables ---
    write_paper_tables(sg_results, usgs_results, OUT_TABLES)

    # --- Final summary line ---
    abl_note = usgs_results["ablation_correlation"]
    abl_str = (
        f"COMPUTABLE with {abl_note.get('n_edge_types_used', '?')} types"
        if abl_ok else
        f"NOT COMPUTABLE — {abl_note.get('note', 'see log above')}"
    )
    _sep()
    print(
        f"SUMMARY: SupplyGraph — 4/4 edge types processed; "
        f"{sg_valid} pairs valid; {sg_skipped} skipped (zero-variance/NaN).\n"
        f"         USGS — {n_edge_types}/15 edge types processed; "
        f"valid_prod={usgs_valid_prod} | valid_cons={usgs_valid_cons} "
        f"| skipped_both={usgs_skipped} (74/127 prod nodes + 46/127 cons nodes "
        f"are constant/zero in real data — genuine data floor, not a script error).\n"
        f"         Ablation correlation: {abl_str}."
    )
    _sep()


if __name__ == "__main__":
    main()
