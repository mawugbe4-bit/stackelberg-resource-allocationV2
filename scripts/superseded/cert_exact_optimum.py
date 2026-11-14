"""
Exact Optimum via Integer Programming
-----------------------------------------------------------------------
Directly addresses the audit's repeated FATAL finding: the claim that
"SA finds nothing better than greedy" has only ever been tested against
greedy itself, never against the TRUE optimal allocation. Greedy being
unbeatable by SA is consistent with greedy being near-optimal, but it is
equally consistent with SA simply being a weak search -- without solving
the actual optimum, these two explanations cannot be distinguished.

This script solves the EXACT capacity- and budget-constrained assignment
problem via mixed-integer programming (scipy.optimize.milp, HiGHS solver,
open-source, no extra install), for all four arm x scenario cells (baseline/calibrated x
Scenario A/B), and reports:
  - the exact optimal value
  - greedy's value
  - the gap, as both an absolute number and a percentage

This is the single most requested FATAL item across both audit rounds:
"G4 Exact optimum (integer programme) and greedy gap as a number -- Not
computed" / "Q23 Null finding established -- Iterations, optimum,
resolution and degeneracy all still failing".

BEFORE RUNNING: edit DATA_PATH and INSIDERS_CSV below (same files as
cert_sa_greedy_pipeline_v6.py). Requires only numpy, pandas, scipy (>= 1.9) -- no extra solver package.
"""

import time
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
import scipy

DATA_PATH = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
BUDGET = 7000.0

CALIBRATED_WEIGHTS = {"impact": 0.224, "propagation": 0.252, "confidence": 0.247, "evasion": 0.278}
BASELINE_WEIGHTS = {"impact": 0.25, "propagation": 0.25, "confidence": 0.25, "evasion": 0.25}

CATALOG_SCENARIOS = {
    "A_training_unconstrained": pd.DataFrame([
        {"resource_id": "training",      "cost": 24.0,  "effectiveness": 0.0428, "capacity": 100_000},
        {"resource_id": "monitoring",    "cost": 36.0,  "effectiveness": 0.0473, "capacity": 10},
        {"resource_id": "access_review", "cost": 114.0, "effectiveness": 0.0428, "capacity": 5},
        {"resource_id": "dlp",           "cost": 63.2,  "effectiveness": 0.0507, "capacity": 100_000},
        {"resource_id": "ir",            "cost": 15.0,  "effectiveness": 0.3378, "capacity": 10},
    ]),
    "B_training_constrained": pd.DataFrame([
        {"resource_id": "training",      "cost": 24.0,  "effectiveness": 0.0428, "capacity": 5},
        {"resource_id": "monitoring",    "cost": 36.0,  "effectiveness": 0.0473, "capacity": 10},
        {"resource_id": "access_review", "cost": 114.0, "effectiveness": 0.0428, "capacity": 5},
        {"resource_id": "dlp",           "cost": 63.2,  "effectiveness": 0.0507, "capacity": 100_000},
        {"resource_id": "ir",            "cost": 15.0,  "effectiveness": 0.3378, "capacity": 10},
    ]),
}


def load_and_aggregate_to_person_level():
    df = pd.read_csv(DATA_PATH)
    insiders = pd.read_csv(INSIDERS_CSV)
    insiders["dataset"] = insiders["dataset"].astype(str).str.strip()
    r42_insider_users = set(insiders[insiders["dataset"] == "4.2"]["user"].unique())
    split = df["threat_id"].str.split("_", n=1, expand=True)
    df["user"] = split[0]
    agg = df.groupby("user").agg(
        impact=("impact", "mean"), propagation=("propagation", "mean"),
        evasion=("evasion", "mean"), confidence=("confidence", "mean"),
    ).reset_index()
    agg["malicious"] = agg["user"].isin(r42_insider_users).astype(int)
    for col in ["impact", "propagation", "evasion"]:
        lo, hi = agg[col].min(), agg[col].max()
        agg[col] = (agg[col] - lo) / (hi - lo) if hi > lo else 0.0
    return agg


def threat_scores(df, weights):
    return (
        weights["impact"] * df["impact"] + weights["propagation"] * df["propagation"]
        + weights["confidence"] * df["confidence"] + weights["evasion"] * df["evasion"]
    ).to_numpy()


def greedy_allocate(threat_score, budget, catalog):
    n = len(threat_score)
    costs = catalog["cost"].to_numpy()
    effs = catalog["effectiveness"].to_numpy()
    caps = catalog["capacity"].to_numpy()
    n_res = len(catalog)
    value_matrix = threat_score[:, None] * effs[None, :]
    cost_matrix = np.tile(costs, (n, 1))
    ratio_matrix = value_matrix / cost_matrix
    order = np.argsort(-ratio_matrix.ravel())
    t_idx_all, r_idx_all = np.unravel_index(order, (n, n_res))
    assigned = np.full(n, -1, dtype=np.int8)
    used_capacity = np.zeros(n_res, dtype=np.int64)
    spent = 0.0
    for t_idx, r_idx in zip(t_idx_all, r_idx_all):
        if assigned[t_idx] != -1 or used_capacity[r_idx] >= caps[r_idx]:
            continue
        cost = costs[r_idx]
        if spent + cost <= budget:
            assigned[t_idx] = r_idx
            used_capacity[r_idx] += 1
            spent += cost
    value = sum(threat_score[i] * effs[r] for i, r in enumerate(assigned) if r != -1)
    return assigned, value


def solve_exact_optimum(threat_score, budget, catalog, time_limit_s=300):
    """Exact solve via mixed-integer programming (scipy.optimize.milp, which
    wraps the HiGHS solver). Same formulation as the greedy/SA allocators:
    at most one resource per person, a capacity limit per resource, and a
    total budget constraint.

    Returns (optimal_value, status_text, elapsed_s, upper_bound):
      - status_text 'Optimal' means HiGHS PROVED this is the true optimum.
      - Otherwise the best solution found is returned together with
        HiGHS's proven upper bound on the true optimum, so the greedy gap
        can still be bounded rigorously even without proof of optimality.
    """
    scores = np.asarray(threat_score, dtype=float)
    costs = catalog["cost"].to_numpy(dtype=float)
    effs = catalog["effectiveness"].to_numpy(dtype=float)
    caps = catalog["capacity"].to_numpy(dtype=float)
    n = len(scores)
    n_res = len(catalog)
    n_var = n * n_res                       # variable k = i * n_res + r

    # Objective: maximise sum score_i * eff_r * x_ir  ->  minimise the negative
    c = -(scores[:, None] * effs[None, :]).ravel()

    # Constraints, built sparse (1000 persons x 5 resources = 5000 variables)
    n_rows = n + n_res + 1
    A = lil_matrix((n_rows, n_var))
    ub = np.empty(n_rows)
    for i in range(n):                      # at most one resource per person
        for r in range(n_res):
            A[i, i * n_res + r] = 1.0
        ub[i] = 1.0
    for r in range(n_res):                  # capacity per resource
        for i in range(n):
            A[n + r, i * n_res + r] = 1.0
        ub[n + r] = caps[r]
    for i in range(n):                      # total budget
        for r in range(n_res):
            A[n + n_res, i * n_res + r] = costs[r]
    ub[n + n_res] = float(budget)
    A = csr_matrix(A)

    t0 = time.perf_counter()
    res = milp(
        c,
        constraints=LinearConstraint(A, ub=ub),
        integrality=np.ones(n_var),
        bounds=Bounds(0, 1),
        options={"time_limit": float(time_limit_s), "disp": False},
    )
    elapsed = time.perf_counter() - t0

    status_map = {0: "Optimal", 1: "Time/iteration limit reached", 2: "Infeasible",
                  3: "Unbounded", 4: "Solver error / other"}
    status = status_map.get(res.status, f"status {res.status}")

    if res.x is None:
        return float("nan"), status, elapsed, float("nan")

    optimal_value = float(-res.fun)
    dual = getattr(res, "mip_dual_bound", None)
    upper_bound = float(-dual) if dual is not None and np.isfinite(dual) else float("nan")
    return optimal_value, status, elapsed, upper_bound


def main():
    print(f"SciPy version in use: {scipy.__version__}  (solver: HiGHS via scipy.optimize.milp)")
    df = load_and_aggregate_to_person_level()
    n = len(df)
    print(f"Loaded {n:,} person-level records.\n")

    results = []
    for catalog_name, catalog in CATALOG_SCENARIOS.items():
        print("\n" + "#" * 70)
        print(f"CATALOG SCENARIO: {catalog_name}")
        print("#" * 70)
        for label, weights in [("baseline", BASELINE_WEIGHTS), ("calibrated", CALIBRATED_WEIGHTS)]:
            scores = threat_scores(df, weights)
            _, greedy_value = greedy_allocate(scores, BUDGET, catalog)

            print(f"\nSolving exact optimum for [{catalog_name}, {label}] "
                  f"({n} persons x {len(catalog)} resources = {n*len(catalog)} binary variables)...")
            print("This may take a while -- solver time limit set to 300s.")
            opt_value, status, elapsed, upper_bound = solve_exact_optimum(scores, BUDGET, catalog)

            gap_abs = opt_value - greedy_value
            gap_pct = 100 * gap_abs / opt_value if opt_value else float("nan")

            print(f"  Greedy value:  {greedy_value:.6f}")
            print(f"  Optimal value: {opt_value:.6f}  (solver status: {status}, {elapsed:.1f}s)")
            print(f"  Gap: {gap_abs:+.6f} absolute ({gap_pct:.4f}% of optimum)")
            if status != "Optimal" and np.isfinite(upper_bound):
                max_gap = upper_bound - greedy_value
                print(f"  NOT proven optimal. Proven upper bound on the true optimum: {upper_bound:.6f}")
                print(f"  => greedy is within at most {max_gap:.6f} "
                      f"({100*max_gap/upper_bound:.4f}%) of the true optimum")

            results.append({
                "catalog_scenario": catalog_name, "weight_set": label,
                "greedy_value": greedy_value, "optimal_value": opt_value,
                "gap_absolute": gap_abs, "gap_percent": gap_pct,
                "solver_status": status, "solve_time_s": elapsed,
                "proven_upper_bound": upper_bound,
            })

    print("\n" + "=" * 70)
    print("SUMMARY -- paste into Results chapter / supervisor update")
    print("=" * 70)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    summary.to_csv("cert_exact_optimum_results.csv", index=False)
    print("\nSaved: cert_exact_optimum_results.csv")


if __name__ == "__main__":
    main()
