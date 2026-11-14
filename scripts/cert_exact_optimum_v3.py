"""
Exact optimum via mixed-integer programming -- v3
-----------------------------------------------------------------------
Solves the capacity- and budget-constrained assignment problem EXACTLY (scipy.optimize.milp / HiGHS) for
all four weight-arm x catalogue-scenario cells and reports greedy's value, the optimum, and the gap.

v3 changes from v2: runs on the shared pipeline_common module, so it uses the SAME (current,
consistent-basis) catalogue as every other script. v2 carried its own copy of the old catalogue and must
not be used. Labelling is unchanged from v2: a cell is "PROVEN optimal" only when HiGHS proves it;
otherwise the value is "best found" and HiGHS's proven upper bound caps how far greedy can be from the
true optimum.

The MIP formulation is verified against exhaustive brute force in verify_formulation_bruteforce.py.
"""

import numpy as np
import pandas as pd
import pipeline_common as C

try:
    import config as _cfg
    CONFIDENCE_CSV = str(_cfg.CONFIDENCE_CSV)
    INSIDERS_CSV = str(_cfg.INSIDERS_CSV)
    BUDGET = float(_cfg.BUDGET)
except Exception:
    CONFIDENCE_CSV = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
    INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
    BUDGET = 7000.0

TIME_LIMIT_S = 300       # raise (e.g. 1800) to try to PROVE optimality in a hard cell
MIP_REL_GAP = 1e-6


def _clean(v, tol=1e-9):
    return 0.0 if abs(v) < tol else v


def main():
    df = C.load_person_level(CONFIDENCE_CSV, INSIDERS_CSV)
    n = len(df)
    results = []
    for scen_name, catalog in C.CATALOG_SCENARIOS.items():
        rids = list(catalog["resource_id"])
        print("\n" + "#" * 70 + f"\nCATALOG SCENARIO: {scen_name}\n" + "#" * 70)
        for arm, weights in C.ARMS.items():
            scores = C.threat_scores(df, weights)
            g, gu, gs = C.greedy_allocate(scores, BUDGET, catalog)
            gv = C.allocation_value(scores, g, catalog)
            print(f"\n[{scen_name}, {arm}] solving ({n} persons x {len(catalog)} resources = {n * len(catalog)} binary variables; "
                  f"limit {TIME_LIMIT_S}s, relative gap {MIP_REL_GAP})...")
            best, status, elapsed, ub, assign = C.solve_exact(scores, BUDGET, catalog, TIME_LIMIT_S, MIP_REL_GAP)
            proven = status == "Optimal"
            if not np.isfinite(best):
                print(f"  NO feasible solution within {TIME_LIMIT_S}s (status {status}); raise TIME_LIMIT_S")
                results.append({"catalog_scenario": scen_name, "weight_set": arm, "greedy_value": gv, "solver_status": status})
                continue
            gap = _clean(best - gv)
            gap_pct = 100 * gap / best if best else float("nan")
            max_gap = _clean(ub - gv) if np.isfinite(ub) else float("nan")
            max_pct = 100 * max_gap / ub if np.isfinite(ub) and ub else float("nan")
            print(f"  Greedy value:  {gv:.6f}")
            if proven:
                print(f"  PROVEN optimal value: {best:.6f} ({elapsed:.1f}s) -> greedy is {gap:.6f} ({gap_pct:.4f}%) below the true optimum")
            else:
                print(f"  Best value FOUND: {best:.6f} (status: {status}, {elapsed:.1f}s) -- NOT proven optimal")
                print(f"  Proven upper bound {ub:.6f} => greedy is at most {max_gap:.6f} ({max_pct:.4f}%) below the true optimum; "
                      f"true gap lies in [{gap_pct:.4f}%, {max_pct:.4f}%]")
            gm = {r: int((g == k).sum()) for k, r in enumerate(rids)}
            bm = {r: int((assign == k).sum()) for k, r in enumerate(rids)}
            spent_b = sum(catalog.cost[k] * bm[r] for k, r in enumerate(rids))
            print(f"  Resource mix greedy: {gm}  (spent {gs:.2f})")
            print(f"  Resource mix {'optimum' if proven else 'best found'}: {bm}  (spent {spent_b:.2f})")
            row = {"catalog_scenario": scen_name, "weight_set": arm, "greedy_value": gv, "best_found_value": best,
                   "proven_optimal": proven, "solver_status": status, "solve_time_s": elapsed, "proven_upper_bound": ub,
                   "gap_pct_vs_best_found": gap_pct, "max_possible_gap_pct": max_pct,
                   "greedy_spent": gs, "best_spent": spent_b}
            for r in rids:
                row[f"greedy_n_{r}"], row[f"best_n_{r}"] = gm[r], bm[r]
            results.append(row)
    R = pd.DataFrame(results)
    R.to_csv(C.out_path("cert_exact_optimum_results_v3.csv"), index=False)
    show = ["catalog_scenario", "weight_set", "greedy_value", "best_found_value", "proven_optimal",
            "gap_pct_vs_best_found", "max_possible_gap_pct", "solve_time_s"]
    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    print(R.reindex(columns=show).to_string(index=False))
    print("\nSaved: cert_exact_optimum_results_v3.csv  (repeated 'HighsMipSolverData' lines are harmless solver messages)")


if __name__ == "__main__":
    main()
