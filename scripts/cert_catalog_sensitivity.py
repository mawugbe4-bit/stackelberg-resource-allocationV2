"""
Catalogue sensitivity -- how much do the conclusions depend on the catalogue's inputs?
-----------------------------------------------------------------------------------------
Supervisor review asked for at least three catalogues (low / central / high effectiveness, with Incident
Response capacity at 2, 10 and 20), both arms. This script runs a full grid and adds three targeted cases:

  GRID (27 variants x 2 Training-capacity scenarios = 54 cells)
     effectiveness scale : low = 0.5x, central = 1x, high = 1.5x   (applied to all five controls)
     IR capacity         : 2, 10, 20
     cost level          : low end, central, high end of every sourced cost range
  TARGETED
     IR_legacy_contrast  : IR effectiveness = 0.3378 (the older "IR team + tested plan vs neither"
                           contrast, USD 1.5M / 4.44M) -- the value the earlier drafts used as headline.
     DLP_forrester       : DLP effectiveness = 0.0507 (vendor-commissioned Forrester TEI figure).
     drop_confidence     : the Confidence criterion removed from U (weights renormalised). Confidence is a
                           classifier output trained on the same insider labels it is then evaluated on, so
                           this shows how much of the result flows through that circular criterion.

For each cell: exact integer counts (k insiders covered by each arm, gained/lost), and a paired cluster
bootstrap (persons; B = 200; seed 42) of the calibrated - baseline difference in insider coverage rate,
classified by whether its 95% percentile interval lies above zero, below zero, or includes zero.
"""

import itertools
import numpy as np
import pandas as pd
import pipeline_common as C
from cert_primary_analysis_v4 import _outcomes

try:
    import config as _cfg
    CONFIDENCE_CSV = str(_cfg.CONFIDENCE_CSV)
    INSIDERS_CSV = str(_cfg.INSIDERS_CSV)
    BUDGET = float(_cfg.BUDGET)
except Exception:
    CONFIDENCE_CSV = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
    INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
    BUDGET = 7000.0

N_BOOT = 200
SEED = 42


def _drop_conf(w):
    t = 1.0 - w["confidence"]
    return {c: (0.0 if c == "confidence" else v / t) for c, v in w.items()}


def variants():
    v = []
    for eff, irc, cost in itertools.product([("low", 0.5), ("central", 1.0), ("high", 1.5)], [2, 10, 20],
                                            ["low", "central", "high"]):
        v.append((f"eff={eff[0]}|IRcap={irc}|cost={cost}",
                  dict(eff_scale=eff[1], ir_capacity=irc, cost_level=cost), "all"))
    v.append(("IR_legacy_contrast (eff 0.3378)", dict(ir_eff=C.IR_EFF_LEGACY_CONTRAST), "all"))
    v.append(("DLP_forrester (eff 0.0507)", dict(dlp_eff=C.DLP_SAVING_FORRESTER_USD / C.GLOBAL_AVG_BREACH_COST), "all"))
    v.append(("drop_confidence (3 criteria)", dict(), "drop_conf"))
    return v


def main():
    df = C.load_person_level(CONFIDENCE_CSV, INSIDERS_CSV)
    X = df[C.CRITERIA].to_numpy(dtype=float)
    mal = df["malicious"].to_numpy() == 1
    scen = df["scenario"].to_numpy()
    n = len(df)
    rows = []
    for tname, tcap in (("A_training_unconstrained", 100_000), ("B_training_constrained", 5)):
        for vname, kw, mode in variants():
            catalog = C.make_catalog(training_capacity=tcap, **kw)
            wb = C.BASELINE_WEIGHTS if mode == "all" else _drop_conf(C.BASELINE_WEIGHTS)
            wc = C.CALIBRATED_WEIGHTS if mode == "all" else _drop_conf(C.CALIBRATED_WEIGHTS)
            ob, _, ab = _outcomes(X, mal, scen, wb, catalog, BUDGET)
            oc, _, ac = _outcomes(X, mal, scen, wc, catalog, BUDGET)
            cb, cc = ab >= 0, ac >= 0
            gained, lost = int((cc & ~cb & mal).sum()), int((cb & ~cc & mal).sum())
            rng = np.random.default_rng(SEED)
            d = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                sb, _, _ = _outcomes(X[idx], mal[idx], scen[idx], wb, catalog, BUDGET)
                sc, _, _ = _outcomes(X[idx], mal[idx], scen[idx], wc, catalog, BUDGET)
                d.append(sc["rate"] - sb["rate"])
            d = np.array(d)
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            verdict = "above 0" if lo > 0 else "below 0" if hi < 0 else "includes 0"
            rows.append({"training_scenario": tname, "variant": vname, "k_baseline": ob["k"], "k_calibrated": oc["k"],
                         "insiders_gained": gained, "insiders_lost": lost, "n_insiders": ob["n_ins"],
                         "covered_persons": oc["n_covered"], "diff_boot_mean": np.nanmean(d), "diff_p2.5": lo,
                         "diff_p97.5": hi, "interval": verdict, "enrichment_baseline": ob["enrichment"],
                         "enrichment_calibrated": oc["enrichment"]})
        print(f"... finished {tname}")
    R = pd.DataFrame(rows)
    R.to_csv(C.out_path("catalog_sensitivity.csv"), index=False)

    print("\n" + "=" * 90 + "\nSUMMARY ACROSS ALL CATALOGUE VARIANTS (paired bootstrap, calibrated - baseline)\n" + "=" * 90)
    for t, g in R.groupby("training_scenario"):
        print(f"\n{t}: {len(g)} variants")
        print(f"  95% interval above 0: {(g.interval == 'above 0').sum()}   below 0: {(g.interval == 'below 0').sum()}   "
              f"includes 0: {(g.interval == 'includes 0').sum()}")
        print(f"  net insiders (calibrated - baseline) across variants: min {int((g.k_calibrated - g.k_baseline).min())}, "
              f"median {float((g.k_calibrated - g.k_baseline).median()):.0f}, max {int((g.k_calibrated - g.k_baseline).max())}")
        print(f"  insiders covered, baseline: min {int(g.k_baseline.min())}, max {int(g.k_baseline.max())} of {int(g.n_insiders.iloc[0])}")
    print("\nTARGETED VARIANTS")
    show = R[~R.variant.str.startswith("eff=")]
    print(show[["training_scenario", "variant", "k_baseline", "k_calibrated", "insiders_gained", "insiders_lost",
                "diff_boot_mean", "diff_p2.5", "diff_p97.5", "interval"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSaved: catalog_sensitivity.csv")


if __name__ == "__main__":
    main()
