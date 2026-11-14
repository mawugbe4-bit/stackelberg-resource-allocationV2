"""
Primary outcome analysis -- v4   (replaces cert_bootstrap_ate_v3 and cert_cate_by_scenario_v3)
-----------------------------------------------------------------------------------------------
Compares the two deterministic allocation policies (equal weights vs AHP-calibrated weights,
both greedy) on the same population, under both catalogue scenarios.

Why this script replaces the previous two (supervisor review):
  * RESOLUTION. The outcome was a proportion over 70 insiders, so its finest step is 1/70 =
    0.0143, larger than the effects reported (+0.0012, -0.0026). v4 therefore reports (a) the
    EXACT integer counts k/70 for each arm, (b) how many insiders GAINED and LOST coverage when the
    weights changed, and (c) continuous outcomes with far finer resolution: the AUC of the utility
    score for ranking insiders above non-insiders, and effectiveness-weighted coverage.
  * POINT ESTIMATES vs BOOTSTRAP. Full-sample point estimates are reported separately from the
    bootstrap mean +/- SD. (v3 printed only replicate means, e.g. 0.5396 x 70 = 37.77 insiders.)
  * EVERY interval is printed as numbers, for every subgroup. v3 replaced intervals with the text
    "no detectable effect" when the interval was narrower than a threshold.
  * SUBGROUPS are reported as k/30, k/30, k/10 with percentile intervals, and are called
    "subgroup differences" -- not CATEs: no causal estimator is used (the two policies are
    deterministic functions of the same population, so there is no confounded treatment assignment).
  * ENRICHMENT (insider coverage rate / overall coverage rate) is reported next to the headline,
    together with the person-level base rate (70/1,000 = 7%). Part of it is circular, because the
    Confidence criterion is a classifier trained on the same insider labels.

Design: cluster bootstrap resampling PERSONS with replacement (B = 1,000; seed 42); both arms and both
catalogue scenarios are evaluated on every identical resample, so differences are paired.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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

N_BOOTSTRAP = 1000
SEED = 42
OUTCOMES = ["rate", "eff_weighted", "auc", "enrichment", "overall_cov_rate", "sub1", "sub2", "sub3"]


def _scores(X, weights):
    return X @ np.array([weights[c] for c in C.CRITERIA])


def _outcomes(X, mal, scen, weights, catalog, budget):
    """All outcomes for one arm on (a possibly resampled) population given as arrays."""
    scores = _scores(X, weights)
    assigned, used, spent = C.greedy_allocate(scores, budget, catalog)
    covered = assigned >= 0
    n_ins, n = int(mal.sum()), len(mal)
    k = int((covered & mal).sum())
    effs = catalog["effectiveness"].to_numpy(dtype=float)
    eff_cov = float(effs[assigned[mal & covered]].sum()) / n_ins if n_ins else np.nan
    base = covered.mean()
    out = {"k": k, "n_ins": n_ins, "rate": k / n_ins if n_ins else np.nan, "eff_weighted": eff_cov,
           "auc": C.auc_score(scores, mal), "overall_cov_rate": base,
           "enrichment": (k / n_ins) / base if n_ins and base > 0 else np.nan,
           "n_covered": int(covered.sum()), "spent": spent}
    for s in (1, 2, 3):
        m = scen == s
        out[f"sub{s}"] = (covered & m).sum() / m.sum() if m.sum() else np.nan
        out[f"sub{s}_k"] = int((covered & m).sum())
        out[f"sub{s}_n"] = int(m.sum())
    return out, scores, assigned


def point_estimates(df, catalog, budget, label):
    X = df[C.CRITERIA].to_numpy(dtype=float)
    mal = df["malicious"].to_numpy() == 1
    scen = df["scenario"].to_numpy()
    res = {}
    for arm, w in C.ARMS.items():
        res[arm] = _outcomes(X, mal, scen, w, catalog, budget)
    ob, sb, ab = res["baseline"]
    oc, sc, ac = res["calibrated"]
    cov_b, cov_c = ab >= 0, ac >= 0
    gained = int((cov_c & ~cov_b & mal).sum())
    lost = int((cov_b & ~cov_c & mal).sum())
    jacc = float((cov_b & cov_c).sum() / max((cov_b | cov_c).sum(), 1))
    rho = float(spearmanr(sb, sc)[0])
    n_ins = int(mal.sum())
    rows = [("covered persons (of %d)" % len(df), ob["n_covered"], oc["n_covered"]),
            ("insiders covered k (of %d)" % n_ins, ob["k"], oc["k"]),
            ("insiders GAINED by calibration", np.nan, gained),
            ("insiders LOST by calibration", np.nan, lost),
            ("net insiders (calibrated - baseline)", np.nan, oc["k"] - ob["k"]),
            ("Jaccard overlap of covered sets", np.nan, jacc),
            ("persons whose coverage differs", np.nan, int((cov_b != cov_c).sum())),
            ("Spearman rho of U between arms", np.nan, rho),
            ("coverage rate of insiders k/n", ob["rate"], oc["rate"]),
            ("overall coverage rate", ob["overall_cov_rate"], oc["overall_cov_rate"]),
            ("enrichment (insider rate / overall rate)", ob["enrichment"], oc["enrichment"]),
            ("effectiveness-weighted insider coverage", ob["eff_weighted"], oc["eff_weighted"]),
            ("AUC of U (insiders vs non-insiders)", ob["auc"], oc["auc"]),
            ("budget spent", ob["spent"], oc["spent"])]
    for s in (1, 2, 3):
        rows.append((f"scenario {s}: insiders covered (k of {ob[f'sub{s}_n']})", ob[f"sub{s}_k"], oc[f"sub{s}_k"]))
    tab = pd.DataFrame(rows, columns=["metric", "baseline", "calibrated"])
    tab.insert(0, "catalog_scenario", label)
    print(f"\n{'=' * 78}\nPOINT ESTIMATES, full sample (exact; no bootstrap) -- {label}\n{'=' * 78}")
    print(f"person-level base rate of insiders = {n_ins}/{len(df)} = {n_ins / len(df):.3f};  "
          f"metric resolution = 1/{n_ins} = {1 / n_ins:.4f}")
    print(tab[["metric", "baseline", "calibrated"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return tab


def bootstrap(df, catalog, budget, label, n_boot=N_BOOTSTRAP, seed=SEED):
    X = df[C.CRITERIA].to_numpy(dtype=float)
    mal = df["malicious"].to_numpy() == 1
    scen = df["scenario"].to_numpy()
    n = len(df)
    rng = np.random.default_rng(seed)
    reps = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)                 # resample PERSONS with replacement
        Xs, ms, ss = X[idx], mal[idx], scen[idx]
        ob, _, _ = _outcomes(Xs, ms, ss, C.BASELINE_WEIGHTS, catalog, budget)
        oc, _, _ = _outcomes(Xs, ms, ss, C.CALIBRATED_WEIGHTS, catalog, budget)
        row = {"rep": b}
        for key in OUTCOMES:
            row[f"{key}_baseline"], row[f"{key}_calibrated"] = ob[key], oc[key]
            row[f"{key}_diff"] = oc[key] - ob[key]
        reps.append(row)
        if (b + 1) % 250 == 0:
            print(f"   ... {b + 1}/{n_boot} replicates")
    R = pd.DataFrame(reps)
    R.to_csv(C.out_path(f"primary_analysis_bootstrap_replicates_{label}.csv"), index=False)

    rows = []
    for key in OUTCOMES:
        for kind in ("baseline", "calibrated", "diff"):
            v = R[f"{key}_{kind}"].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
            rows.append({"catalog_scenario": label, "outcome": key, "quantity": kind, "n_valid": len(v),
                         "mean": v.mean(), "sd": v.std(ddof=1), "p2.5": lo, "p97.5": hi,
                         "share_gt0": float((v > 1e-12).mean()), "share_lt0": float((v < -1e-12).mean()),
                         "share_eq0": float((np.abs(v) <= 1e-12).mean())})
    S = pd.DataFrame(rows)
    print(f"\n{'=' * 78}\nCLUSTER BOOTSTRAP (persons, B={n_boot}, seed={seed}) -- {label}\n{'=' * 78}")
    names = {"rate": "insider coverage rate", "eff_weighted": "effectiveness-weighted coverage",
             "auc": "AUC of U", "enrichment": "enrichment", "overall_cov_rate": "overall coverage rate",
             "sub1": "subgroup 1 (rapid theft) rate", "sub2": "subgroup 2 (gradual theft) rate",
             "sub3": "subgroup 3 (sabotage, n=10) rate"}
    for key in OUTCOMES:
        d = S[(S.outcome == key) & (S.quantity == "diff")].iloc[0]
        b = S[(S.outcome == key) & (S.quantity == "baseline")].iloc[0]
        c = S[(S.outcome == key) & (S.quantity == "calibrated")].iloc[0]
        print(f"{names[key]:34s} baseline {b['mean']:.4f}+/-{b['sd']:.4f}  calibrated {c['mean']:.4f}+/-{c['sd']:.4f}  "
              f"| diff {d['mean']:+.4f} (SD {d['sd']:.4f}) 95% interval [{d['p2.5']:+.4f}, {d['p97.5']:+.4f}]  "
              f"(>0 in {100 * d['share_gt0']:.0f}%, =0 in {100 * d['share_eq0']:.0f}%, <0 in {100 * d['share_lt0']:.0f}% of replicates)")
    return S


def main():
    df = C.load_person_level(CONFIDENCE_CSV, INSIDERS_CSV)
    pts, boots = [], []
    for label, catalog in C.CATALOG_SCENARIOS.items():
        pts.append(point_estimates(df, catalog, BUDGET, label))
        boots.append(bootstrap(df, catalog, BUDGET, label))
    pd.concat(pts).to_csv(C.out_path("primary_analysis_point_estimates.csv"), index=False)
    pd.concat(boots).to_csv(C.out_path("primary_analysis_bootstrap.csv"), index=False)
    print("\nSaved: primary_analysis_point_estimates.csv, primary_analysis_bootstrap.csv, "
          "primary_analysis_bootstrap_replicates_<scenario>.csv")
    print("NOTE: subgroup contrasts are SUBGROUP DIFFERENCES between two deterministic policies, not causal "
          "treatment effects (no causal estimator is used).")


if __name__ == "__main__":
    main()
