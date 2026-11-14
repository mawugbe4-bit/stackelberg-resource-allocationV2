"""
SA / Greedy Allocation Pipeline -- v8
-----------------------------------------------------------------------
Supersedes v4..v7 (v7 was issued but never run). Root causes found and fixed, in order of discovery:

  (1) COOLING SCHEDULE (v6). 0.95 cooling from T0=1000 reached the minimum after ~225 iterations,
      so the documented 10,000 iterations never ran. Fixed in v6 (schedule now reaches its minimum at
      exactly max_iter).
  (2) MIS-SCALED TEMPERATURE (found after v6). T0=1000 is ~5 orders of magnitude above any single-move
      change in the objective (<= ~0.3). SA then accepts nearly every move, random-walks away from the
      greedy start and never climbs back above it, giving the exactly-zero gain / zero variance
      reported in every earlier version, whatever the iteration budget.
  (3) EMPTY BOUNDARY POOL (found while porting the code into pipeline_common). The "marginally
      admitted" half of the boundary pool was built by taking the 500 SMALLEST values with unassigned
      threats set to -inf; unassigned threats therefore sorted first and were then filtered out, leaving
      the admitted half EMPTY whenever >= 500 threats were unassigned (705 in this study). 80% of all
      proposals could therefore only try to ADD an assignment under an almost-exhausted budget.
      Fixed in pipeline_common.sa_allocate.

What v8 does
  - Measures the distribution of |delta value| over ~1,000 single-move proposals from the greedy
    solution (per cell). The "rule-based primary" schedule is T0 = median |delta| (the supervisor's
    recommendation). Schedules at 3x, 0.3x, 0.1x, 0.03x the median, plus the LEGACY T0=1000
    schedule, are run for every cell, so the artefact stays visible and NO schedule is declared
    correct in advance: on synthetic instances with a known gap, SA recovered nothing at T0 >= the
    median |delta| and recovered most of the gap at T0 <= 0.3x the median, so the sweep itself is the
    finding. The empirically best schedule in each cell is flagged separately from the rule-based one.
    The same schedule set is applied to BOTH arms (baseline, calibrated) and both catalogue
    scenarios -- no arm-specific tuning. Total schedules tried per cell: 6 (reported in full).
  - Every schedule reaches its minimum temperature (T0 x 1e-3) at exactly max_iter = 10,000.
  - Logs the REAL iteration count, accepted moves and final temperature for every run, and writes a
    per-seed table (sa_per_seed_results.csv).
  - Reports, for each cell, spend / unspent budget / resource counts, and WHY no further assignment
    was made (unspent budget vs the cheapest resource with spare capacity).
  - Reports the exact optimum (HiGHS, short limit) as a reference so each schedule's gain can be read
    as a share of the gap that actually exists.
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

SA_MAX_ITER = 10000
SA_SEEDS = [42, 123, 456, 789, 1010, 2024, 3141, 5926, 5358, 9793]     # 10 seeds
T0_MULTIPLIERS = [3.0, 1.0, 0.3, 0.1, 0.03]                             # of the median |delta|; 1.0 = primary
PRIMARY_MULTIPLIER = 1.0
LEGACY_T0, LEGACY_TMIN = 1000.0, 0.01
EXACT_REFERENCE = True
EXACT_TIME_LIMIT_S = 60


def schedules_for(median_delta):
    s = {"legacy T0=1000": (LEGACY_T0, LEGACY_TMIN)}
    for m in T0_MULTIPLIERS:
        t0 = m * median_delta
        s[f"T0={m:g} x median|d|"] = (t0, t0 * 1e-3)
    return s


def resource_report(assigned, used, spent, catalog, budget):
    rids = list(catalog["resource_id"])
    print("  Resource usage under greedy:")
    for r, rid in enumerate(rids):
        print(f"    {rid:14s} used {int(used[r]):>6,} of capacity {int(catalog.capacity[r]):>7,}")
    print(f"    {'(unassigned)':14s} {int((assigned == -1).sum()):>6,}")
    unspent = budget - spent
    spare = [(rid, catalog.cost[r]) for r, rid in enumerate(rids) if used[r] < catalog.capacity[r]]
    cheapest = min(spare, key=lambda t: t[1]) if spare else (None, np.inf)
    print(f"  Spend {spent:.2f} of {budget:.0f}; UNSPENT {unspent:.2f}. Cheapest resource with spare capacity: "
          f"{cheapest[0]} at {cheapest[1]:.2f} -> "
          f"{'a further assignment is affordable (the loop should have made it)' if unspent >= cheapest[1] - 1e-9 else 'unspent budget is below that cost, so no further assignment is possible'}")


def main():
    df = C.load_person_level(CONFIDENCE_CSV, INSIDERS_CSV)
    per_seed, summary = [], []
    for scen_name, catalog in C.CATALOG_SCENARIOS.items():
        print("\n" + "#" * 78 + f"\nCATALOG SCENARIO: {scen_name}\n" + "#" * 78)
        for arm, weights in C.ARMS.items():
            print("\n" + "=" * 78 + f"\nARM: {arm}\n" + "=" * 78)
            scores = C.threat_scores(df, weights)
            g_assigned, g_used, g_spent = C.greedy_allocate(scores, BUDGET, catalog)
            g_value = C.allocation_value(scores, g_assigned, catalog)
            print(f"Greedy value = {g_value:.6f}")
            resource_report(g_assigned, g_used, g_spent, catalog, BUDGET)

            d = C.delta_distribution(scores, g_assigned, catalog, n_proposals=1000, seed=0)
            med = float(np.median(d))
            print(f"|delta value| over {len(d)} single-move proposals from greedy: median {med:.5f}, "
                  f"IQR [{np.percentile(d, 25):.5f}, {np.percentile(d, 75):.5f}], p5 {np.percentile(d, 5):.5f}, "
                  f"p95 {np.percentile(d, 95):.5f}, max {d.max():.5f}")
            print(f"Legacy T0 = {LEGACY_T0:g} is {LEGACY_T0 / med:,.0f} x the median |delta|.")

            ref_value, ref_status = np.nan, ""
            if EXACT_REFERENCE:
                ref_value, ref_status, _, ub, _ = C.solve_exact(scores, BUDGET, catalog, EXACT_TIME_LIMIT_S)
                print(f"Exact reference: {ref_value:.6f} ({ref_status}); available gap over greedy = "
                      f"{ref_value - g_value:+.6f} ({100 * (ref_value - g_value) / ref_value:.3f}%)")

            print(f"SA, {len(SA_SEEDS)} seeds per schedule, {SA_MAX_ITER} iterations each:")
            cell_rows = []
            for sname, (t0, tmin) in schedules_for(med).items():
                vals = []
                for seed in SA_SEEDS:
                    r = C.sa_allocate(scores, BUDGET, g_assigned, catalog, t0, tmin, SA_MAX_ITER, seed)
                    vals.append(r["best_value"])
                    per_seed.append({"catalog_scenario": scen_name, "arm": arm, "schedule": sname,
                                     "T0": t0, "T_min": tmin, "seed": seed, "greedy_value": g_value,
                                     "sa_value": r["best_value"], "gain": r["best_value"] - g_value,
                                     "iterations": r["iterations"], "accepted_moves": r["accepted"],
                                     "final_temp": r["final_temp"]})
                v = np.array(vals)
                share = (v.mean() - g_value) / (ref_value - g_value) if ref_value - g_value > 1e-4 * abs(ref_value) else np.nan   # only meaningful if gap >= 0.01% of optimum
                primary = " <- RULE-BASED PRIMARY (T0 = median |delta|)" if sname.startswith(f"T0={PRIMARY_MULTIPLIER:g} x") else ""
                print(f"  {sname:22s} T0={t0:<10.5g} mean {v.mean():.6f} SD {v.std(ddof=1):.6f} min {v.min():.6f} "
                      f"max {v.max():.6f} | gain mean {v.mean() - g_value:+.6f}, best {v.max() - g_value:+.6f}"
                      + (f" | mean gain = {100 * share:.1f}% of available gap" if np.isfinite(share) else "") + primary)
                cell_rows.append({"catalog_scenario": scen_name, "arm": arm, "schedule": sname, "T0": t0,
                                "greedy_value": g_value, "exact_reference": ref_value, "exact_status": ref_status,
                                "sa_mean": v.mean(), "sa_sd": v.std(ddof=1), "sa_min": v.min(), "sa_max": v.max(),
                                "mean_gain": v.mean() - g_value, "best_gain": v.max() - g_value,
                                "mean_gain_share_of_gap": share, "median_abs_delta": med,
                                "rule_based_primary": primary != "", "best_in_cell": False})
            best_i = int(np.argmax([r["sa_mean"] for r in cell_rows]))
            if cell_rows[best_i]["mean_gain"] > 1e-9:
                cell_rows[best_i]["best_in_cell"] = True
                print(f"  Empirically best schedule in this cell (highest mean value): {cell_rows[best_i]['schedule']} "
                      f"(mean gain {cell_rows[best_i]['mean_gain']:+.6f}) -- selected AFTER seeing results; reported as a "
                      f"sensitivity finding, not as a pre-specified choice.")
            else:
                print("  No schedule improved on greedy in this cell (mean gain <= 1e-9 for every schedule).")
            summary.extend(cell_rows)
    ps = pd.DataFrame(per_seed)
    ps.to_csv(C.out_path("sa_per_seed_results.csv"), index=False)
    pd.DataFrame(summary).to_csv(C.out_path("sa_summary.csv"), index=False)
    print(f"\nIterations actually run: min {ps.iterations.min()}, max {ps.iterations.max()} "
          f"(target {SA_MAX_ITER}). Saved: sa_per_seed_results.csv, sa_summary.csv")


if __name__ == "__main__":
    main()
