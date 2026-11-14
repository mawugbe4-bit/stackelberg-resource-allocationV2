"""
Ablation grid for the SA refinement (supervisor item Q20) -- three components, each measured separately
------------------------------------------------------------------------------------------------------
  (a) PROPOSAL DISTRIBUTION : boundary-pool proposals (80% boundary / 20% uniform) vs purely uniform.
  (b) SEEDING               : greedy-seeded SA vs empty-start SA (start from nothing allocated), reported as a
                              number: the share of greedy's value each reaches in the same iteration budget.
  (c) CAPACITY CONSTRAINTS  : constraints on vs off (every capacity lifted to 100,000), for greedy, the
                              exact optimum, and the resource mix -- how much the capacity limits shape the result.

Every SA cell uses the same 10 seeds and a cold schedule (T0 = 0.1 x the median |delta value|), the same for both
arms and both catalogue scenarios. NOTE: that temperature was chosen after the schedule sweep in
cert_sa_greedy_pipeline_v8.py (where hotter schedules recovered none of the gap); it is a sensitivity
setting, not a pre-specified one. The exact optimum (HiGHS, short limit) is the reference for
"share of the available gap".
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

SEEDS = [42, 123, 456, 789, 1010, 2024, 3141, 5926, 5358, 9793]
MAX_ITER = 10000
T0_MULT = 0.1
EXACT_TIME_LIMIT_S = 60


def run_sa(scores, catalog, start, t0, boundary_prob):
    vals, acc = [], []
    for s in SEEDS:
        r = C.sa_allocate(scores, BUDGET, start, catalog, t0, t0 * 1e-3, MAX_ITER, s, boundary_prob=boundary_prob)
        vals.append(r["best_value"]); acc.append(r["accepted"])
    return np.array(vals), np.array(acc)


def main():
    df = C.load_person_level(CONFIDENCE_CSV, INSIDERS_CSV)
    rows = []
    for scen_name, catalog in C.CATALOG_SCENARIOS.items():
        uncapped = catalog.copy()
        uncapped["capacity"] = 100_000
        for arm, w in C.ARMS.items():
            scores = C.threat_scores(df, w)
            g, gu, gs = C.greedy_allocate(scores, BUDGET, catalog)
            gv = C.allocation_value(scores, g, catalog)
            ref, st, _, ub, _ = C.solve_exact(scores, BUDGET, catalog, EXACT_TIME_LIMIT_S)
            med = float(np.median(C.delta_distribution(scores, g, catalog, 1000, 0)))
            t0 = T0_MULT * med
            gap = ref - gv
            print("\n" + "=" * 90 + f"\n{scen_name} | {arm} | greedy {gv:.6f} | exact {ref:.6f} ({st}) | gap {gap:+.6f} | T0 = {T0_MULT} x median = {t0:.5f}\n" + "=" * 90)

            def record(component, setting, vals, acc=None, extra=None):
                share = (vals.mean() - gv) / gap if gap > 1e-4 * abs(ref) else np.nan   # only meaningful if gap >= 0.01% of optimum
                row = {"catalog_scenario": scen_name, "arm": arm, "component": component, "setting": setting,
                       "sa_mean": vals.mean(), "sa_sd": vals.std(ddof=1), "mean_gain_over_greedy": vals.mean() - gv,
                       "share_of_available_gap": share, "share_of_greedy_value": vals.mean() / gv,
                       "mean_accepted_moves": float(np.mean(acc)) if acc is not None else np.nan,
                       "greedy_value": gv, "exact_reference": ref}
                row.update(extra or {})
                rows.append(row)
                print(f"  [{component}] {setting:34s} mean {vals.mean():.6f} SD {vals.std(ddof=1):.6f} | gain {vals.mean() - gv:+.6f}"
                      + (f" | {100 * share:5.1f}% of gap" if np.isfinite(share) else "")
                      + f" | {100 * vals.mean() / gv:6.2f}% of greedy value"
                      + (f" | accepted {np.mean(acc):.0f}" if acc is not None else ""))

            # (a) proposal distribution
            for label, bp in (("boundary 80% / uniform 20%", 0.8), ("uniform only", 0.0)):
                v, a = run_sa(scores, catalog, g, t0, bp)
                record("a: proposals", label, v, a)
            # (b) seeding
            empty = np.full(len(scores), -1, dtype=np.int16)
            v, a = run_sa(scores, catalog, g, t0, 0.8)
            record("b: seeding", "greedy-seeded", v, a)
            v, a = run_sa(scores, catalog, empty, t0, 0.8)
            record("b: seeding", "empty start", v, a)
            # (c) capacity constraints on / off
            gu_assigned, _, _ = C.greedy_allocate(scores, BUDGET, uncapped)
            gv_off = C.allocation_value(scores, gu_assigned, uncapped)
            ref_off, st_off, _, _, _ = C.solve_exact(scores, BUDGET, uncapped, EXACT_TIME_LIMIT_S)
            mix_on = {rid: int((g == k).sum()) for k, rid in enumerate(catalog.resource_id)}
            mix_off = {rid: int((gu_assigned == k).sum()) for k, rid in enumerate(catalog.resource_id)}
            print(f"  [c: capacity] constraints ON : greedy {gv:.6f}, exact {ref:.6f}, mix {mix_on}")
            print(f"  [c: capacity] constraints OFF: greedy {gv_off:.6f}, exact {ref_off:.6f} ({st_off}), mix {mix_off}")
            rows.append({"catalog_scenario": scen_name, "arm": arm, "component": "c: capacity", "setting": "constraints on",
                         "sa_mean": np.nan, "greedy_value": gv, "exact_reference": ref, "mix": str(mix_on)})
            rows.append({"catalog_scenario": scen_name, "arm": arm, "component": "c: capacity", "setting": "constraints off",
                         "sa_mean": np.nan, "greedy_value": gv_off, "exact_reference": ref_off, "mix": str(mix_off)})
    pd.DataFrame(rows).to_csv(C.out_path("sa_ablation.csv"), index=False)
    print("\nSaved: sa_ablation.csv")


if __name__ == "__main__":
    main()
