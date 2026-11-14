"""
Normalisation sensitivity -- does the preprocessing choice matter more than the AHP weights?
-----------------------------------------------------------------------------------------------
Found in the first full run: replacing min-max normalisation with percentile-rank normalisation raised the
AUC of the utility score from 0.69 to 0.86 and the insiders covered from 46 to 57 (Scenario A) and from 17 to 31
(Scenario B), while the two weight vectors differ by at most ~0.03 per criterion (rank correlation of the two
score vectors 0.9995). The features are extremely heavy-tailed (Impact 80% zeros, Evasion 87% zeros at
user-week level), so a single extreme value compresses everyone else under min-max.

This script puts intervals on both effects, under BOTH normalisations, with the same paired person-level
cluster bootstrap (B = 1,000, seed 42) as the primary analysis:
  (1) WEIGHT EFFECT  : calibrated - baseline, under min-max and under rank normalisation;
  (2) NORMALISATION EFFECT : rank - min-max, for each arm.
Min-max is the specified (primary) normalisation and stays the primary analysis; rank normalisation is a
robustness check. It is NOT adopted because it performs better -- choosing a preprocessing step after seeing which
one scores higher on the evaluation labels would be selection on the outcome.
Percentile ranks and min-max constants are computed once on the full population; resampled persons keep them.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
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

N_BOOT = 1000
SEED = 42


def rank_matrix(df):
    n = len(df)
    return np.column_stack([(rankdata(df[c].to_numpy(), method="average") - 1) / max(n - 1, 1) for c in C.CRITERIA])


def main():
    df = C.load_person_level(CONFIDENCE_CSV, INSIDERS_CSV)
    mal = df["malicious"].to_numpy() == 1
    scen = df["scenario"].to_numpy()
    n = len(df)
    mats = {"minmax": df[C.CRITERIA].to_numpy(dtype=float), "rank": rank_matrix(df)}
    rows = []
    for label, catalog in C.CATALOG_SCENARIOS.items():
        print("\n" + "=" * 100 + f"\n{label}\n" + "=" * 100)
        pts = {}
        for norm, X in mats.items():
            for arm, w in C.ARMS.items():
                o, s, a = _outcomes(X, mal, scen, w, catalog, BUDGET)
                pts[(norm, arm)] = (o, a)
                print(f"  point estimate [{norm:6s} | {arm:10s}] insiders covered {o['k']:>3}/{o['n_ins']}  AUC of U {o['auc']:.4f}  "
                      f"enrichment {o['enrichment']:.2f}  covered persons {o['n_covered']}")
        for norm in mats:
            cb, cc = pts[(norm, "baseline")][1] >= 0, pts[(norm, "calibrated")][1] >= 0
            print(f"  [{norm:6s}] calibrated vs baseline: gained {int((cc & ~cb & mal).sum())}, lost {int((cb & ~cc & mal).sum())}, "
                  f"Jaccard {(cb & cc).sum() / (cb | cc).sum():.3f}")
        for arm in C.ARMS:
            cm, cr = pts[("minmax", arm)][1] >= 0, pts[("rank", arm)][1] >= 0
            print(f"  [{arm:10s}] rank vs min-max: Jaccard {(cm & cr).sum() / (cm | cr).sum():.3f}; insiders only under rank "
                  f"{int((cr & ~cm & mal).sum())}, only under min-max {int((cm & ~cr & mal).sum())}")

        rng = np.random.default_rng(SEED)
        reps = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            r = {}
            for norm, X in mats.items():
                for arm, w in C.ARMS.items():
                    o, _, _ = _outcomes(X[idx], mal[idx], scen[idx], w, catalog, BUDGET)
                    r[(norm, arm)] = o
            row = {}
            for norm in mats:
                row[f"weight_effect_{norm}_rate"] = r[(norm, "calibrated")]["rate"] - r[(norm, "baseline")]["rate"]
                row[f"weight_effect_{norm}_auc"] = r[(norm, "calibrated")]["auc"] - r[(norm, "baseline")]["auc"]
            for arm in C.ARMS:
                row[f"norm_effect_{arm}_rate"] = r[("rank", arm)]["rate"] - r[("minmax", arm)]["rate"]
                row[f"norm_effect_{arm}_auc"] = r[("rank", arm)]["auc"] - r[("minmax", arm)]["auc"]
            reps.append(row)
        R = pd.DataFrame(reps)
        print(f"\n  PAIRED CLUSTER BOOTSTRAP (persons, B={N_BOOT}, seed {SEED}): mean [95% percentile interval]  (share of replicates >0 / =0 / <0)")
        for col in R.columns:
            v = R[col].to_numpy(dtype=float); v = v[~np.isnan(v)]
            lo, hi = np.percentile(v, [2.5, 97.5])
            kind = "WEIGHT effect (calibrated - baseline)" if col.startswith("weight") else "NORMALISATION effect (rank - min-max)"
            print(f"   {col:34s} {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  ({100 * (v > 1e-12).mean():.0f}% / {100 * (np.abs(v) <= 1e-12).mean():.0f}% / {100 * (v < -1e-12).mean():.0f}%)   <- {kind}")
            rows.append({"catalog_scenario": label, "quantity": col, "mean": v.mean(), "sd": v.std(ddof=1),
                         "p2.5": lo, "p97.5": hi, "share_gt0": float((v > 1e-12).mean()),
                         "share_eq0": float((np.abs(v) <= 1e-12).mean()), "share_lt0": float((v < -1e-12).mean())})
    pd.DataFrame(rows).to_csv(C.out_path("normalisation_sensitivity.csv"), index=False)
    print("\nSaved: normalisation_sensitivity.csv")


if __name__ == "__main__":
    main()
