"""
Classifier error analysis -- v3  (replaces cert_confusion_matrix_v2)
-----------------------------------------------------------------------
v3 adds what the supervisor asked for beyond a single confusion matrix at 0.5 (items Q8/54, Q3(v)):
  * a THRESHOLD SWEEP (TP/FP/FN/TN, TPR, FPR, precision, F1, share of flags that fall on non-insiders);
  * the PRECISION-RECALL CURVE with its area and the prevalence baseline (the right curve under 0.5% positives);
  * a DESCRIPTION OF THE FALSE NEGATIVES (by insider scenario; insiders never flagged in any week) and of the
    FALSE POSITIVES (how many distinct people, how concentrated).
Uses the labels exported by cert_confidence_classifier_v2.py, so every script agrees on what is positive.

Reading it: the Confidence score is used in this thesis only as ONE weighted input to a retrospective allocation
ranking. At any threshold where recall is high, almost all flags fall on staff who did nothing; an insider-threat
flag against an innocent person carries reputational, employment and trust costs that are not symmetric with a
missed insider, so the score is not proposed as a basis for flagging individuals.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score)
import pipeline_common as C

try:
    import config as _cfg
    CONFIDENCE_CSV = str(_cfg.CONFIDENCE_CSV)
    LABELS_CSV = str(_cfg.RESULTS_DIR / "cert_userweek_labels.csv")
    INSIDERS_CSV = str(_cfg.INSIDERS_CSV)
except Exception:
    CONFIDENCE_CSV = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
    LABELS_CSV = r"A:\PhD IT\results\cert_userweek_labels.csv"
    INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    f = pd.read_csv(CONFIDENCE_CSV)
    lab = pd.read_csv(LABELS_CSV)
    df = f.merge(lab, on="threat_id", how="inner")
    assert len(df) == len(f) == len(lab), "feature and label files do not align"
    y, p = df["malicious"].to_numpy(), df["confidence"].to_numpy()
    n, n_pos = len(y), int(y.sum())
    print(f"n = {n:,} user-weeks; positives = {n_pos:,} ({n_pos / n:.3%}); negatives = {n - n_pos:,}")

    rows = []
    for t in THRESHOLDS:
        pred = p >= t
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = n_pos - tp, (n - n_pos) - fp
        rows.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "TPR_recall": tp / n_pos, "FPR": fp / (n - n_pos),
                     "precision": tp / (tp + fp) if tp + fp else np.nan,
                     "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                     "flags": tp + fp, "share_of_flags_on_non_insiders": fp / (tp + fp) if tp + fp else np.nan})
    S = pd.DataFrame(rows)
    S.to_csv(C.out_path("confidence_threshold_sweep.csv"), index=False)
    print("\nTHRESHOLD SWEEP\n" + S.round(4).to_string(index=False))

    r = S[S.threshold == 0.5].iloc[0]
    print(f"\nAT 0.5: TP={int(r.TP):,} FP={int(r.FP):,} FN={int(r.FN):,} TN={int(r.TN):,} | recall {r.TPR_recall:.4f} FPR {r.FPR:.4f} "
          f"precision {r.precision:.4f} F1 {r.F1:.4f}")
    print(f"        {100 * r.share_of_flags_on_non_insiders:.1f}% of all flags fall on user-weeks of people who did nothing "
          f"(base rate of positives is {100 * n_pos / n:.2f}%).")
    pd.DataFrame([[r.TN, r.FP], [r.FN, r.TP]], index=["actual not malicious", "actual malicious"],
                 columns=["pred not malicious", "pred malicious"]).to_csv(C.out_path("confidence_confusion_matrix.csv"))

    ap, auc = average_precision_score(y, p), roc_auc_score(y, p)
    print(f"\nPR-AUC (average precision) = {ap:.4f} against a prevalence baseline of {n_pos / n:.4f} "
          f"({ap / (n_pos / n):.1f}x baseline); ROC-AUC = {auc:.4f}")
    pr, rc, _ = precision_recall_curve(y, p)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(rc, pr, lw=2, label=f"classifier (AP = {ap:.3f})")
        ax.axhline(n_pos / n, color="grey", ls="--", label=f"prevalence = {n_pos / n:.4f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall (out-of-fold)")
        ax.legend(); plt.tight_layout(); plt.savefig(C.out_path("confidence_pr_curve.png"), dpi=150); plt.close()
    except Exception as e:
        print("figure skipped:", e)

    # ---- who are the false negatives / false positives (threshold 0.5) ----
    ins = pd.read_csv(INSIDERS_CSV)
    ins["dataset"] = ins["dataset"].astype(str).str.strip()
    scen = dict(zip(ins[ins.dataset == "4.2"]["user"], ins[ins.dataset == "4.2"]["scenario"].astype(int)))
    df["pred"] = (p >= 0.5).astype(int)
    pos = df[df.malicious == 1].copy()
    pos["scenario"] = pos["user"].map(scen)
    print("\nFALSE NEGATIVES at 0.5 -- insider-window weeks missed, by insider scenario")
    fnrows = []
    for s, g in pos.groupby("scenario"):
        fn = int((g.pred == 0).sum())
        never = int((g.groupby("user")["pred"].max() == 0).sum())
        fnrows.append({"scenario": int(s), "positive_weeks": len(g), "missed_weeks": fn, "missed_share": fn / len(g),
                       "insiders": g.user.nunique(), "insiders_never_flagged_in_any_window_week": never})
    FN = pd.DataFrame(fnrows)
    print(FN.round(3).to_string(index=False))
    FN.to_csv(C.out_path("confidence_false_negatives_by_scenario.csv"), index=False)

    neg = df[df.malicious == 0]
    fp = neg[neg.pred == 1]
    per_user = fp.groupby("user").size().sort_values(ascending=False)
    top10 = per_user.head(max(len(per_user) // 10, 1)).sum()
    print(f"\nFALSE POSITIVES at 0.5: {len(fp):,} user-weeks from {per_user.size:,} distinct people "
          f"(of {neg.user.nunique():,} people with at least one non-malicious week); per-person flagged weeks: "
          f"median {per_user.median():.0f}, 90th percentile {per_user.quantile(.9):.0f}, max {per_user.max()}; "
          f"the most-flagged 10% of those people account for {100 * top10 / len(fp):.1f}% of false flags.")
    print("Mean features, false positives vs correctly-cleared weeks: "
          + ", ".join(f"{c} {fp[c].mean():.3f} vs {neg[neg.pred == 0][c].mean():.3f}" for c in ["impact", "propagation", "evasion"]))
    print("\nSaved: confidence_threshold_sweep.csv, confidence_confusion_matrix.csv, confidence_pr_curve.png, "
          "confidence_false_negatives_by_scenario.csv")


if __name__ == "__main__":
    main()
