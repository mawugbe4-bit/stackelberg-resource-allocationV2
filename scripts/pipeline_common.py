"""
pipeline_common.py -- single source of truth for the allocation pipeline
-----------------------------------------------------------------------
Every analysis script imports from here, so the catalogue, weights,
person-level aggregation, allocators and outcome metrics cannot drift
apart between scripts (version drift between copied functions was a
recurring source of error in earlier versions of this pipeline).

Contents
  - AHP weights (baseline and calibrated)           -> BASELINE_WEIGHTS, CALIBRATED_WEIGHTS
  - Resource catalogue, with provenance and variants  -> make_catalog(), CATALOG_SOURCES
  - Person-level loader (per-week MEANS)             -> load_person_level()
  - Utility scores                                   -> threat_scores(), rank_normalised_scores()
  - Allocators                                       -> greedy_allocate(), sa_allocate(), solve_exact()
  - Outcome metrics                                  -> coverage_stats(), auc_score(), subgroup_stats()

Catalogue basis (important). Every effectiveness value is
    (average per-breach saving attributable to the control, in USD)
    / (global average breach cost, USD 4.44M, IBM 2025)
and every saving is taken on the SAME basis: "each factor in isolation,
measured against the global average" (IBM Cost of a Data Breach Report,
the 'factors that increase or decrease breach costs' figure). Earlier
versions took Incident Response from a different, older measure
("IR team plus tested plan versus neither", USD 1.5M-2.7M), which is not
comparable and inflated IR roughly 6-fold relative to the other controls.
That older value is retained ONLY as an explicit sensitivity case.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:                       # default: the repo's results/ folder, whatever directory Python was launched from
    import config as _cfg
    OUTPUT_DIR = str(_cfg.RESULTS_DIR)
except Exception:
    OUTPUT_DIR = "."       # run_all.py also sets this explicitly


def out_path(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, name)


# ----------------------------------------------------------------------
# 1. WEIGHTS
# ----------------------------------------------------------------------
CRITERIA = ["impact", "propagation", "confidence", "evasion"]
BASELINE_WEIGHTS = {"impact": 0.25, "propagation": 0.25, "confidence": 0.25, "evasion": 0.25}
# Calibration-wave weights carried at 4 d.p. so they sum to exactly 1.
# (The 3 d.p. rounding 0.224/0.252/0.247/0.278 sums to 1.001.)
CALIBRATED_WEIGHTS = {"impact": 0.2235, "propagation": 0.2517, "confidence": 0.2473, "evasion": 0.2775}
ARMS = {"baseline": BASELINE_WEIGHTS, "calibrated": CALIBRATED_WEIGHTS}


# ----------------------------------------------------------------------
# 2. CATALOGUE
# ----------------------------------------------------------------------
GLOBAL_AVG_BREACH_COST = 4_440_000.0   # IBM Cost of a Data Breach Report 2025

# Per-breach saving in USD, same "factor in isolation" basis throughout.
SAVINGS_USD = {
    "training":      192_266.0,   # IBM 2025, "Employee training"
    "monitoring":    212_061.0,   # IBM 2025, "Security analytics or SIEM"
    "access_review": 189_838.0,   # IBM 2025, "IAM"
    "dlp":           157_456.0,   # IBM 2025, "Data security and protection software"
    "ir":            248_000.0,   # IBM 2024 (IBM Think summary), IR team, in-isolation basis
}
IR_SAVING_LEGACY_CONTRAST_USD = 1_500_000.0   # older "IR team + tested plan vs neither" contrast
DLP_SAVING_FORRESTER_USD = 225_000.0          # vendor-commissioned (Forrester TEI, Microsoft Purview)

# Cost per person per year in USD: (low end, central, high end) of the sourced ranges.
COSTS = {
    "training":      (12.0, 24.0, 36.0),      # KnowBe4/Proofpoint pricing
    "monitoring":    (12.0, 36.0, 60.0),      # SIEM $12k-$60k/yr over CERT's ~1,000 users
    "access_review": (24.0, 114.0, 204.0),    # Okta pricing, SSO to privileged access
    "dlp":           (20.0, 63.25, 200.0),    # mean of 4 named vendors (34, 52, 96, 71) = 63.25
    "ir":            (5.0, 15.0, 25.0),       # IR retainer $5k-$25k/yr over ~1,000 users
}
CAPACITY = {"training": 100_000, "monitoring": 10, "access_review": 5, "dlp": 100_000, "ir": 10}
RESOURCE_ORDER = ["training", "monitoring", "access_review", "dlp", "ir"]

CATALOG_SOURCES = {
    "effectiveness": "IBM Cost of a Data Breach Report 2025, Figure 39 (training, SIEM, IAM, data security "
                     "software); IBM 2024 (IR team, IBM Think summary). Same 'factor in isolation' basis. "
                     "Divided by the 2025 global average breach cost (USD 4.44M).",
    "cost": "Published vendor/market pricing (KnowBe4, Proofpoint, Okta, SIEMCostCalculator, Petronella, "
            "Atlant, Strac); flat organisational fees divided by CERT's ~1,000 users.",
    "capacity": "SANS SOC Survey 2024/2025 (2-10 staff) for SIEM and IR; NinjaOne 1 admin per 200 users for "
               "access review; Training and DLP treated as automated (training capacity varied in Scenario B).",
}


def make_catalog(training_capacity=100_000, cost_level="central", eff_scale=1.0,
                 ir_eff=None, ir_capacity=10, dlp_eff=None):
    """Return the resource catalogue as a DataFrame.

    training_capacity : 100000 (Scenario A, unconstrained) or 5 (Scenario B).
    cost_level        : 'low' | 'central' | 'high' end of each sourced cost range.
    eff_scale         : multiplies every effectiveness (sensitivity).
    ir_eff, dlp_eff   : override the Incident Response / DLP effectiveness (absolute values,
                        applied AFTER eff_scale is NOT applied to overrides).
    ir_capacity       : Incident Response capacity (central 10; sensitivity 2 and 20).
    """
    idx = {"low": 0, "central": 1, "high": 2}[cost_level]
    rows = []
    for rid in RESOURCE_ORDER:
        eff = SAVINGS_USD[rid] / GLOBAL_AVG_BREACH_COST * eff_scale
        if rid == "ir" and ir_eff is not None:
            eff = ir_eff
        if rid == "dlp" and dlp_eff is not None:
            eff = dlp_eff
        cap = CAPACITY[rid]
        if rid == "training":
            cap = training_capacity
        if rid == "ir":
            cap = ir_capacity
        rows.append({"resource_id": rid, "cost": COSTS[rid][idx], "effectiveness": eff, "capacity": cap})
    return pd.DataFrame(rows)


CATALOG_SCENARIOS = {
    "A_training_unconstrained": make_catalog(training_capacity=100_000),
    "B_training_constrained":   make_catalog(training_capacity=5),
}
IR_EFF_LEGACY_CONTRAST = IR_SAVING_LEGACY_CONTRAST_USD / GLOBAL_AVG_BREACH_COST   # 0.3378


# ----------------------------------------------------------------------
# 3. PERSON-LEVEL LOADER  (per-week MEANS, not sums -- see changelog)
# ----------------------------------------------------------------------
def load_person_level(features_csv, insiders_csv, dataset="4.2", verbose=True):
    """One row per person. Impact/Propagation/Evasion/Confidence are MEANS over
    that person's active weeks (summing a weekly proportion such as Evasion is
    meaningless, and summing any feature confounds tenure with intensity).
    Features are then min-max re-normalised at the person level.
    Also returns n_weeks (active weeks), malicious (planted r4.2 insider) and scenario."""
    df = pd.read_csv(features_csv)
    ins = pd.read_csv(insiders_csv)
    ins["dataset"] = ins["dataset"].astype(str).str.strip()
    r42 = ins[ins["dataset"] == dataset]
    scen = dict(zip(r42["user"], r42["scenario"].astype(int)))
    df["user"] = df["threat_id"].str.split("_", n=1, expand=True)[0]
    agg = df.groupby("user").agg(
        impact=("impact", "mean"), propagation=("propagation", "mean"),
        evasion=("evasion", "mean"), confidence=("confidence", "mean"),
        n_weeks=("threat_id", "count"),
    ).reset_index()
    agg["malicious"] = agg["user"].isin(scen).astype(int)
    agg["scenario"] = agg["user"].map(scen).fillna(0).astype(int)
    for col in ["impact", "propagation", "evasion"]:
        lo, hi = agg[col].min(), agg[col].max()
        agg[col] = (agg[col] - lo) / (hi - lo) if hi > lo else 0.0
    if verbose:
        print(f"Aggregated {len(df):,} user-week records to {len(agg):,} person-level records "
              f"({int(agg['malicious'].sum())} malicious).")
    return agg


# ----------------------------------------------------------------------
# 4. UTILITY SCORES
# ----------------------------------------------------------------------
def threat_scores(df, weights, drop=()):
    """U_i = sum_c w_c * x_ic. `drop` removes criteria and renormalises the rest."""
    w = {c: weights[c] for c in CRITERIA if c not in drop}
    tot = sum(w.values())
    return sum((w[c] / tot) * df[c].to_numpy() for c in w)


def rank_normalised_scores(df, weights, drop=()):
    """Same weighted sum, but each criterion is replaced by its percentile rank in [0,1]
    (robust to the heavy tails that make min-max normalisation sensitive to outliers)."""
    w = {c: weights[c] for c in CRITERIA if c not in drop}
    tot = sum(w.values())
    n = len(df)
    return sum((w[c] / tot) * ((rankdata(df[c].to_numpy(), method="average") - 1) / max(n - 1, 1)) for c in w)


# ----------------------------------------------------------------------
# 5. ALLOCATORS
# ----------------------------------------------------------------------
def greedy_allocate(scores, budget, catalog):
    """Ratio-sorted greedy over (person, resource) pairs; at most one resource per person,
    per-resource capacity, total budget. Returns (assigned, used_capacity, spent)."""
    n = len(scores)
    costs = catalog["cost"].to_numpy(dtype=float)
    effs = catalog["effectiveness"].to_numpy(dtype=float)
    caps = catalog["capacity"].to_numpy(dtype=float)
    n_res = len(catalog)
    ratio = (scores[:, None] * effs[None, :]) / costs[None, :]
    order = np.argsort(-ratio.ravel(), kind="stable")
    t_idx_all, r_idx_all = np.unravel_index(order, (n, n_res))
    assigned = np.full(n, -1, dtype=np.int16)
    used = np.zeros(n_res, dtype=np.int64)
    spent = 0.0
    for t, r in zip(t_idx_all, r_idx_all):
        if assigned[t] != -1 or used[r] >= caps[r]:
            continue
        if spent + costs[r] <= budget:
            assigned[t] = r
            used[r] += 1
            spent += costs[r]
    return assigned, used, spent


def allocation_value(scores, assigned, catalog):
    effs = catalog["effectiveness"].to_numpy(dtype=float)
    m = assigned >= 0
    return float((scores[m] * effs[assigned[m]]).sum())


def cooling_rate_for(initial_temp, min_temp, max_iter):
    """Geometric cooling that reaches min_temp at exactly max_iter iterations."""
    return (min_temp / initial_temp) ** (1.0 / max_iter)


def sa_allocate(scores, budget, initial_assigned, catalog, initial_temp, min_temp, max_iter, seed,
                boundary_prob=0.8, pool_size=500):
    """Simulated annealing seeded from `initial_assigned` (greedy, or all -1 for an empty start).
    Moves: single reassignment (any resource or none) and pairwise swap. Best-so-far is tracked.
    Returns dict(best_assigned, best_value, iterations, accepted, final_temp)."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    costs = catalog["cost"].to_numpy(dtype=float)
    effs = catalog["effectiveness"].to_numpy(dtype=float)
    caps = catalog["capacity"].to_numpy(dtype=float)
    n_res = len(catalog)
    cooling = cooling_rate_for(initial_temp, min_temp, max_iter)

    def pv(i, r):
        return scores[i] * effs[r] if r >= 0 else 0.0

    def pc(r):
        return costs[r] if r >= 0 else 0.0

    assigned = initial_assigned.copy()
    spent = float(sum(pc(r) for r in assigned))
    value = float(sum(pv(i, r) for i, r in enumerate(assigned)))
    used = np.array([(assigned == r).sum() for r in range(n_res)], dtype=np.int64)
    best_assigned, best_value = assigned.copy(), value

    # boundary pool: marginally admitted and marginally rejected threats
    pool_n = min(pool_size, n)
    per_value = np.array([pv(i, r) if r >= 0 else np.inf for i, r in enumerate(assigned)])
    admitted = np.argsort(per_value)[:pool_n]
    admitted = admitted[assigned[admitted] != -1]
    best_alt = np.full(n, -np.inf)
    for r in range(n_res):
        best_alt = np.maximum(best_alt, np.where(assigned != r, scores * effs[r], -np.inf))
    rejected = np.argsort(-np.where(assigned == -1, best_alt, -np.inf))[:pool_n]
    boundary = np.unique(np.concatenate([admitted, rejected]))
    if len(boundary) < 2:
        boundary = np.arange(n)
    everyone = np.arange(n)

    temp, it, accepted = float(initial_temp), 0, 0
    while temp > min_temp and it < max_iter:
        pool = boundary if rng.random() < boundary_prob else everyone
        if rng.random() < 0.5:                                  # single reassignment
            t = rng.choice(pool)
            new_r = int(rng.integers(-1, n_res))
            old_r = int(assigned[t])
            if new_r != old_r:
                cap_ok = new_r == -1 or used[new_r] < caps[new_r]
                dc = pc(new_r) - pc(old_r)
                if cap_ok and spent + dc <= budget + 1e-9:
                    dv = pv(t, new_r) - pv(t, old_r)
                    if dv > 0 or rng.random() < np.exp(dv / max(temp, 1e-12)):
                        if old_r != -1:
                            used[old_r] -= 1
                        if new_r != -1:
                            used[new_r] += 1
                        assigned[t] = new_r
                        spent += dc
                        value += dv
                        accepted += 1
                        if value > best_value:
                            best_assigned, best_value = assigned.copy(), value
        else:                                                   # pairwise swap (cost-neutral)
            if len(pool) >= 2:
                i, j = rng.choice(pool, size=2, replace=False)
                ri, rj = int(assigned[i]), int(assigned[j])
                if ri != rj:
                    dv = (pv(i, rj) + pv(j, ri)) - (pv(i, ri) + pv(j, rj))
                    if dv > 0 or rng.random() < np.exp(dv / max(temp, 1e-12)):
                        assigned[i], assigned[j] = rj, ri
                        value += dv
                        accepted += 1
                        if value > best_value:
                            best_assigned, best_value = assigned.copy(), value
        temp *= cooling
        it += 1
    return {"best_assigned": best_assigned, "best_value": float(best_value),
            "iterations": it, "accepted": accepted, "final_temp": temp}


def delta_distribution(scores, assigned, catalog, n_proposals=1000, seed=0):
    """|change in total value| over random single-move proposals from `assigned`
    (used to put SA's temperature on the objective's scale). Returns array of |dv|>0."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    effs = catalog["effectiveness"].to_numpy(dtype=float)
    n_res = len(catalog)
    out = []
    for _ in range(n_proposals):
        t = int(rng.integers(0, n))
        new_r = int(rng.integers(-1, n_res))
        old_r = int(assigned[t])
        if new_r == old_r:
            continue
        dv = (scores[t] * effs[new_r] if new_r >= 0 else 0.0) - (scores[t] * effs[old_r] if old_r >= 0 else 0.0)
        if dv != 0:
            out.append(abs(dv))
    return np.array(out)


def solve_exact(scores, budget, catalog, time_limit_s=300, mip_rel_gap=1e-6):
    """Exact optimum by mixed-integer programming (scipy.optimize.milp / HiGHS).
    Returns (best_value, status_text, elapsed_s, upper_bound, assignment).
    status 'Optimal' => proven optimum; otherwise best found, with a proven upper bound."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix, csr_matrix
    sc = np.asarray(scores, dtype=float)
    costs = catalog["cost"].to_numpy(dtype=float)
    effs = catalog["effectiveness"].to_numpy(dtype=float)
    caps = catalog["capacity"].to_numpy(dtype=float)
    n, n_res = len(sc), len(catalog)
    nv = n * n_res
    c = -(sc[:, None] * effs[None, :]).ravel()
    A = lil_matrix((n + n_res + 1, nv))
    ub = np.empty(n + n_res + 1)
    for i in range(n):
        for r in range(n_res):
            A[i, i * n_res + r] = 1.0
        ub[i] = 1.0
    for r in range(n_res):
        for i in range(n):
            A[n + r, i * n_res + r] = 1.0
        ub[n + r] = caps[r]
    for i in range(n):
        for r in range(n_res):
            A[n + n_res, i * n_res + r] = costs[r]
    ub[n + n_res] = float(budget)
    t0 = time.perf_counter()
    res = milp(c, constraints=LinearConstraint(csr_matrix(A), ub=ub), integrality=np.ones(nv),
               bounds=Bounds(0, 1),
               options={"time_limit": float(time_limit_s), "disp": False, "mip_rel_gap": mip_rel_gap})
    elapsed = time.perf_counter() - t0
    status = {0: "Optimal", 1: "Time/iteration limit reached", 2: "Infeasible",
              3: "Unbounded"}.get(res.status, f"status {res.status}")
    if res.x is None:
        return float("nan"), status, elapsed, float("nan"), None
    dual = getattr(res, "mip_dual_bound", None)
    upper = float(-dual) if dual is not None and np.isfinite(dual) else float("nan")
    x = res.x.reshape(n, n_res).round().astype(int)
    assignment = np.where(x.sum(axis=1) >= 1, x.argmax(axis=1), -1)
    return float(-res.fun), status, elapsed, upper, assignment


# ----------------------------------------------------------------------
# 6. OUTCOME METRICS
# ----------------------------------------------------------------------
def auc_score(scores, malicious):
    """P(a random insider outranks a random non-insider on U), ties counted 1/2 (Mann-Whitney)."""
    m = np.asarray(malicious).astype(bool)
    n1, n0 = int(m.sum()), int((~m).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[m].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def coverage_stats(df, assigned, catalog):
    """Integer-count outcome measures (no rounding): k of n_ins insiders covered, etc."""
    covered = np.asarray(assigned) >= 0
    mal = df["malicious"].to_numpy() == 1
    n_ins, n = int(mal.sum()), len(df)
    k = int((covered & mal).sum())
    n_cov = int(covered.sum())
    effs = catalog["effectiveness"].to_numpy(dtype=float)
    eff_cov = float(sum(effs[r] for r in np.asarray(assigned)[mal] if r >= 0)) / n_ins if n_ins else float("nan")
    rate = k / n_ins if n_ins else float("nan")
    base = n_cov / n if n else float("nan")
    return {"k": k, "n_ins": n_ins, "rate": rate, "n_covered": n_cov, "n": n,
            "overall_cov_rate": base, "eff_weighted": eff_cov,
            "enrichment": rate / base if base and base > 0 else float("nan")}


def subgroup_stats(df, assigned, scenarios=(1, 2, 3)):
    """k covered / n insiders in each r4.2 insider scenario."""
    covered = np.asarray(assigned) >= 0
    out = {}
    for s in scenarios:
        m = df["scenario"].to_numpy() == s
        out[s] = (int((covered & m).sum()), int(m.sum()))
    return out
