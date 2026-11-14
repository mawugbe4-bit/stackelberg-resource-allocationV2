"""
Budget sweep -- both catalogue scenarios, both weight arms
-----------------------------------------------------------------------
Why this exists (supervisor review, items "Budget sweep" and N7): B = 7,000 was carried over from an
earlier illustrative catalogue in which Training cost 1 abstract unit. In the current catalogue's real
dollars it is $7 per person per year, and nothing in this study justifies it. Rather than defend one
number, this script sweeps B across a wide range and reports, at every B, how many insiders each arm
covers (exact integers), who gains and loses when the weights change, and a paired bootstrap interval
for the difference.

Context for reading the range (not a justification of any point in it):
  * Total cover of all 1,000 people with the cheapest control (Training) costs 1,000 x $24 = $24,000, so
    the budget stops binding well below $100k; above that, only capacities limit the allocation.
  * Published benchmarks for a WHOLE security programme are far above this range: IANS Research /
    Artico Search (2025) report about $3,800 of security spend per employee for small and mid-market
    firms (security staffing about 1.5 FTE per 100 employees below $50M revenue). Those figures
    describe commercial organisations, not Ghanaian higher-education institutions, and a whole programme
    rather than the five marginal controls modelled here, so they bound the question; they do not
    answer it. In this model the 'resource-constrained' regime exists only for B below roughly $24-63k.
"""

import numpy as np
import pandas as pd
import pipeline_common as C
from cert_primary_analysis_v4 import _outcomes

try:
    import config as _cfg
    CONFIDENCE_CSV = str(_cfg.CONFIDENCE_CSV)
    INSIDERS_CSV = str(_cfg.INSIDERS_CSV)
except Exception:
    CONFIDENCE_CSV = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
    INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"

BUDGETS = [1000, 2000, 3500, 5000, 7000, 10000, 15000, 20000, 30000, 50000, 75000, 100000, 150000, 250000]
N_BOOT = 300
SEED = 42


def main():
    df = C.load_person_level(CONFIDENCE_CSV, INSIDERS_CSV)
    X = df[C.CRITERIA].to_numpy(dtype=float)
    mal = df["malicious"].to_numpy() == 1
    scen = df["scenario"].to_numpy()
    n = len(df)
    rows = []
    for label, catalog in C.CATALOG_SCENARIOS.items():
        print("\n" + "=" * 100 + f"\nBUDGET SWEEP -- {label}   (n={n}, insiders={int(mal.sum())}, B per person = B/{n})\n" + "=" * 100)
        print(f"{'B ($)':>8} {'$/person':>8} {'covered':>8} {'k_base':>7} {'k_cal':>6} {'gain':>5} {'lost':>5} "
              f"{'diff rate [95% boot interval]':>34} {'enrich_b':>9} {'enrich_c':>9}")
        for B in BUDGETS:
            ob, _, ab = _outcomes(X, mal, scen, C.BASELINE_WEIGHTS, catalog, float(B))
            oc, _, ac = _outcomes(X, mal, scen, C.CALIBRATED_WEIGHTS, catalog, float(B))
            cb, cc = ab >= 0, ac >= 0
            gained, lost = int((cc & ~cb & mal).sum()), int((cb & ~cc & mal).sum())
            rng = np.random.default_rng(SEED)
            diffs = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                sb, _, _ = _outcomes(X[idx], mal[idx], scen[idx], C.BASELINE_WEIGHTS, catalog, float(B))
                sc, _, _ = _outcomes(X[idx], mal[idx], scen[idx], C.CALIBRATED_WEIGHTS, catalog, float(B))
                diffs.append(sc["rate"] - sb["rate"])
            diffs = np.array(diffs)
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            rows.append({"catalog_scenario": label, "budget": B, "per_person": B / n,
                         "covered_persons_baseline": ob["n_covered"], "covered_persons_calibrated": oc["n_covered"],
                         "k_baseline": ob["k"], "k_calibrated": oc["k"], "insiders_gained": gained,
                         "insiders_lost": lost, "n_insiders": ob["n_ins"],
                         "rate_baseline": ob["rate"], "rate_calibrated": oc["rate"],
                         "diff_boot_mean": np.nanmean(diffs), "diff_boot_sd": np.nanstd(diffs, ddof=1),
                         "diff_p2.5": lo, "diff_p97.5": hi,
                         "enrichment_baseline": ob["enrichment"], "enrichment_calibrated": oc["enrichment"],
                         "spent_baseline": ob["spent"], "spent_calibrated": oc["spent"]})
            print(f"{B:>8,} {B / n:>8.1f} {oc['n_covered']:>8} {ob['k']:>7} {oc['k']:>6} {gained:>5} {lost:>5} "
                  f"{np.nanmean(diffs):>+10.4f} [{lo:+.4f}, {hi:+.4f}]{'':>6} {ob['enrichment']:>9.2f} {oc['enrichment']:>9.2f}")
    R = pd.DataFrame(rows)
    R.to_csv(C.out_path("budget_sweep.csv"), index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for label, g in R.groupby("catalog_scenario"):
            ax[0].plot(g.budget, g.k_baseline, "o--", label=f"{label} baseline", alpha=.7)
            ax[0].plot(g.budget, g.k_calibrated, "s-", label=f"{label} calibrated")
            ax[1].plot(g.budget, g.k_calibrated - g.k_baseline, "o-", label=label)
            ax[1].fill_between(g.budget, g["diff_p2.5"] * g.n_insiders, g["diff_p97.5"] * g.n_insiders, alpha=.15)
        for a in ax:
            a.set_xscale("log"); a.set_xlabel("Budget B ($ per year; 1,000 persons)"); a.axvline(7000, color="grey", ls=":")
        ax[0].set_ylabel("Insiders covered (of 70)"); ax[0].legend(fontsize=7)
        ax[1].set_ylabel("Paired difference: calibrated - baseline (insiders)"); ax[1].axhline(0, color="k", lw=.8)
        ax[1].legend(fontsize=8)
        plt.tight_layout(); plt.savefig(C.out_path("budget_sweep.png"), dpi=150); plt.close()
        print("\nSaved: budget_sweep.csv, budget_sweep.png (dotted line = B = 7,000)")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
