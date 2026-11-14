"""
Feature diagnostics (supervisor items Q11)
-----------------------------------------------------------------------
  1. RAW FEATURE DISTRIBUTIONS (min, median, 99th percentile, max, mean, share of zeros, skewness) at
     user-week level and at person level, from the un-normalised features written by
     cert_local_aggregation_v3.py. Heavy tails matter because the utility score is a weighted sum of
     min-max normalised features: a single extreme value compresses everyone else toward 0.
  2. TENURE. Number of active weeks per person, by insider scenario vs everyone else. This is the
     quantity that made summed person-level scores (v1-v3) mechanically lower for early leavers
     (Scenario 1), and is why per-week MEANS are used.
  3. RANK-NORMALISATION CHECK. The whole allocation is re-run with each criterion replaced by its
     percentile rank (robust to tails), and compared with the min-max version: Jaccard overlap of the
     covered set, number of insiders covered under each, and the AUC of U.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
import pipeline_common as C

try:
    import config as _cfg
    CONFIDENCE_CSV = str(_cfg.CONFIDENCE_CSV)
    INSIDERS_CSV = str(_cfg.INSIDERS_CSV)
    RAW_CSV = str(_cfg.RESULTS_DIR / "cert_threat_resource_features_raw.csv")
    BUDGET = float(_cfg.BUDGET)
except Exception:
    CONFIDENCE_CSV = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
    INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
    RAW_CSV = r"A:\PhD IT\results\cert_threat_resource_features_raw.csv"
    BUDGET = 7000.0


def describe(series, name, level):
    v = series.to_numpy(dtype=float)
    return {"level": level, "feature": name, "n": len(v), "min": v.min(), "median": np.median(v),
            "p99": np.percentile(v, 99), "max": v.max(), "mean": v.mean(),
            "share_zero": float((v == 0).mean()), "skewness": float(skew(v)),
            "max_over_p99": float(v.max() / np.percentile(v, 99)) if np.percentile(v, 99) > 0 else np.nan}


def main():
    raw = pd.read_csv(RAW_CSV)
    raw["user"] = raw["threat_id"].str.split("_", n=1, expand=True)[0]
    feats = ["impact_raw", "propagation_raw", "evasion_raw", "total_events"]

    rows = [describe(raw[f], f, "user-week") for f in feats]
    person = raw.groupby("user")[feats].mean()
    rows += [describe(person[f], f + " (person mean)", "person") for f in feats]
    D = pd.DataFrame(rows)
    D.to_csv(C.out_path("feature_distributions.csv"), index=False)
    print("RAW FEATURE DISTRIBUTIONS\n" + D.round(4).to_string(index=False))

    df = C.load_person_level(CONFIDENCE_CSV, INSIDERS_CSV, verbose=False)
    print("\nTENURE: active weeks per person")
    grp = df.assign(group=df["scenario"].map({0: "not an insider", 1: "scenario 1 (rapid theft)",
                                              2: "scenario 2 (gradual theft)", 3: "scenario 3 (sabotage)"}))
    T = grp.groupby("group")["n_weeks"].agg(["count", "min", "median", "mean", "max"])
    print(T.round(2).to_string())
    T.to_csv(C.out_path("tenure_by_group.csv"))

    out = []
    print("\nRANK-NORMALISATION CHECK (greedy, B=%.0f)" % BUDGET)
    for scen_name, catalog in C.CATALOG_SCENARIOS.items():
        for arm, w in C.ARMS.items():
            s_mm = C.threat_scores(df, w)
            s_rk = C.rank_normalised_scores(df, w)
            a_mm, _, _ = C.greedy_allocate(s_mm, BUDGET, catalog)
            a_rk, _, _ = C.greedy_allocate(s_rk, BUDGET, catalog)
            cm, cr = a_mm >= 0, a_rk >= 0
            mal = df["malicious"].to_numpy() == 1
            jacc = (cm & cr).sum() / max((cm | cr).sum(), 1)
            out.append({"catalog_scenario": scen_name, "arm": arm,
                        "k_minmax": int((cm & mal).sum()), "k_rank": int((cr & mal).sum()),
                        "insiders_in_both": int((cm & cr & mal).sum()),
                        "covered_minmax": int(cm.sum()), "covered_rank": int(cr.sum()),
                        "jaccard_covered_sets": float(jacc),
                        "auc_minmax": C.auc_score(s_mm, mal), "auc_rank": C.auc_score(s_rk, mal)})
    O = pd.DataFrame(out)
    O.to_csv(C.out_path("rank_normalisation_check.csv"), index=False)
    print(O.round(4).to_string(index=False))
    print("\nSaved: feature_distributions.csv, tenure_by_group.csv, rank_normalisation_check.csv")


if __name__ == "__main__":
    main()
