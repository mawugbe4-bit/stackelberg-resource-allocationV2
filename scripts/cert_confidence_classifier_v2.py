"""
CERT r4.2 Classification Confidence Builder -- v2
---------------------------------------------------
Stages:
  1. Label each user-week row malicious (1) / not (0) from the CERT ground-truth
     insiders.csv windows (a user-week is positive if it overlaps that insider's window).
  2. Train a class-weighted Random Forest with StratifiedGroupKFold (folds grouped by
     user, so no person's weeks span both sides of a fold) and take each row's
     OUT-OF-FOLD predicted probability as its Confidence value.

v2 changes from v1 (supervisor review):
  - PER-FOLD METRICS (recall, precision, F1, ROC-AUC, PR-AUC at threshold 0.5, plus the
    number of test rows / positives / users in each fold) are computed and saved to
    cert_confidence_fold_metrics.csv. v1 reported only pooled figures.
  - The user-week LABELS are exported (cert_userweek_labels.csv) and `label_features`
    is importable, so no other script has to re-derive them. The v4 timing script
    re-derived them incorrectly (it marked every week of every insider as positive,
    3,156 positives instead of the 322 insider-window weeks), which made its
    classifier timing a different task from the one reported.
  - Paths are read from config.py when present.

Hyperparameters (fixed a priori): RandomForestClassifier(n_estimators=300, max_depth=6,
class_weight='balanced', random_state=42), 5 folds, shuffle seed 42.
HYPERPARAMETER_SEARCH_RUN below records whether ANY search was run; the author must
confirm it before the Methods sentence "no hyperparameter search was performed" is used.
"""

import numpy as np
import pandas as pd

try:
    import config as _cfg
    INSIDERS_CSV = str(_cfg.INSIDERS_CSV)
    FEATURES_CSV = str(_cfg.FEATURES_CSV)
    OUTPUT_CSV = str(_cfg.CONFIDENCE_CSV)
    LABELS_CSV = str(_cfg.RESULTS_DIR / "cert_userweek_labels.csv")
    FOLD_METRICS_CSV = str(_cfg.RESULTS_DIR / "cert_confidence_fold_metrics.csv")
except Exception:                                       # standalone use: edit these
    INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
    FEATURES_CSV = r"A:\PhD IT\results\cert_threat_resource_features.csv"
    OUTPUT_CSV = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
    LABELS_CSV = r"A:\PhD IT\results\cert_userweek_labels.csv"
    FOLD_METRICS_CSV = r"A:\PhD IT\results\cert_confidence_fold_metrics.csv"

TARGET_DATASET = "4.2"
N_SPLITS = 5
RF_PARAMS = dict(n_estimators=300, class_weight="balanced", max_depth=6, random_state=42, n_jobs=-1)
HYPERPARAMETER_SEARCH_RUN = False      # AUTHOR TO CONFIRM (True if any search was ever run)


def load_insider_windows(insiders_csv=None):
    df = pd.read_csv(insiders_csv or INSIDERS_CSV)
    df["dataset"] = df["dataset"].astype(str).str.strip()
    r42 = df[df["dataset"] == TARGET_DATASET].copy()
    r42["start"] = pd.to_datetime(r42["start"])
    r42["end"] = pd.to_datetime(r42["end"])
    print(f"Loaded {len(r42)} insider windows for dataset {TARGET_DATASET}")
    return r42[["user", "start", "end"]]


def label_features(features, insiders, verbose=True):
    """Add user, week_start, week_end and malicious (1 if the user-week overlaps the user's insider window)."""
    features = features.copy()
    split = features["threat_id"].str.split("_", n=1, expand=True)
    features["user"] = split[0]
    week_range = split[1].str.split("/", expand=True)
    features["week_start"] = pd.to_datetime(week_range[0])
    features["week_end"] = pd.to_datetime(week_range[1])
    windows_by_user = {}
    for _, row in insiders.iterrows():
        windows_by_user.setdefault(row["user"], []).append((row["start"], row["end"]))

    def is_malicious(row):
        for w_start, w_end in windows_by_user.get(row["user"], ()):
            if row["week_start"] <= w_end and row["week_end"] >= w_start:
                return 1
        return 0

    features["malicious"] = features.apply(is_malicious, axis=1)
    if verbose:
        n_pos = int(features["malicious"].sum())
        print(f"Labeled {len(features):,} rows: {n_pos:,} malicious user-weeks ({n_pos / len(features):.3%}), "
              f"{int(features['user'].isin(windows_by_user).sum()):,} rows belong to a known insider user.")
    return features


def build_confidence(features):
    """Out-of-fold probabilities via StratifiedGroupKFold(groups = user), with per-fold metrics."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

    feature_cols = ["impact", "propagation", "evasion"]
    X = features[feature_cols].fillna(0.0).to_numpy()
    y = features["malicious"].to_numpy()
    groups = features["user"].to_numpy()
    n_splits = max(min(N_SPLITS, int(y.sum())), 2) if y.sum() >= 2 else 2
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Training with {n_splits}-fold StratifiedGroupKFold grouped by user "
          f"({int(y.sum())} positive rows, {len(set(groups))} distinct users)...")
    proba = np.zeros(len(y))
    rows = []
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups), start=1):
        clf = RandomForestClassifier(**RF_PARAMS).fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        proba[te] = p
        pred = (p >= 0.5).astype(int)
        has_pos = y[te].sum() > 0
        rows.append({
            "fold": fold, "n_test_rows": len(te), "n_test_positive": int(y[te].sum()),
            "n_test_users": len(set(groups[te])), "n_train_users": len(set(groups[tr])),
            "recall": recall_score(y[te], pred, zero_division=0),
            "precision": precision_score(y[te], pred, zero_division=0),
            "f1": f1_score(y[te], pred, zero_division=0),
            "roc_auc": roc_auc_score(y[te], p) if has_pos else np.nan,
            "pr_auc": average_precision_score(y[te], p) if has_pos else np.nan,
            "user_overlap_train_test": len(set(groups[tr]) & set(groups[te])),   # must be 0
        })
    fm = pd.DataFrame(rows)
    assert (fm["user_overlap_train_test"] == 0).all(), "a user appears on both sides of a fold"
    features = features.copy()
    features["confidence"] = proba
    return features, fm


def main():
    insiders = load_insider_windows()
    features = pd.read_csv(FEATURES_CSV)
    labeled = label_features(features, insiders)
    labeled[["threat_id", "user", "malicious"]].to_csv(LABELS_CSV, index=False)
    print(f"Saved labels: {LABELS_CSV}")

    result, fold_metrics = build_confidence(labeled)
    fold_metrics.to_csv(FOLD_METRICS_CSV, index=False)
    print("\nPER-FOLD METRICS (threshold 0.5):")
    print(fold_metrics.round(4).to_string(index=False))
    print("\nMean +/- SD across folds:")
    for c in ["recall", "precision", "f1", "roc_auc", "pr_auc"]:
        print(f"  {c:10s} {fold_metrics[c].mean():.4f} +/- {fold_metrics[c].std(ddof=1):.4f}")
    print(f"Hyperparameter search run: {HYPERPARAMETER_SEARCH_RUN}")

    out = result[["threat_id", "resource_id", "impact", "propagation", "confidence", "evasion", "cost"]]
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(out["confidence"].describe())


if __name__ == "__main__":
    main()
