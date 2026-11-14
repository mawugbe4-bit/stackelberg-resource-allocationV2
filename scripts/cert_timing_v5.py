"""
Computational timing -- v5
-----------------------------------------------------------------------
v5 changes from v4 (supervisor review):
  * CLASSIFIER LABELS. v4 marked every week of every insider as positive (3,156 positives) instead of
    only the weeks overlapping an insider window (322), so the 8.1 s classifier timing came from a
    different task. v5 uses the labels written by cert_confidence_classifier_v2.py
    (cert_userweek_labels.csv), and times the classifier through that script's own build_confidence(),
    so the timed code IS the reported code (5-fold StratifiedGroupKFold, per-fold metrics included).
  * CPU MODEL is recorded (previously "Intel processor"), with library versions.
  * Allocation timings use the shared pipeline_common allocators and the current catalogue; SA is timed at a
    cold schedule (T0 = 0.1 x median |delta|) for 10 seeds x 10,000 iterations.
"""

import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import pipeline_common as C

try:
    import config as _cfg
    CONFIDENCE_CSV = str(_cfg.CONFIDENCE_CSV)
    INSIDERS_CSV = str(_cfg.INSIDERS_CSV)
    LABELS_CSV = str(_cfg.RESULTS_DIR / "cert_userweek_labels.csv")
    BUDGET = float(_cfg.BUDGET)
except Exception:
    CONFIDENCE_CSV = r"A:\PhD IT\results\cert_threat_resource_features_with_confidence.csv"
    INSIDERS_CSV = r"A:\PhD IT\results\answers\answers\insiders.csv"
    LABELS_CSV = r"A:\PhD IT\results\cert_userweek_labels.csv"
    BUDGET = 7000.0

SA_SEEDS = [42, 123, 456, 789, 1010, 2024, 3141, 5926, 5358, 9793]
SA_MAX_ITER = 10000


def cpu_name():
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info()["brand_raw"]
    except Exception:
        pass
    if platform.system() == "Windows":
        for cmd in (["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                    ["wmic", "cpu", "get", "name"]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
                lines = [l.strip() for l in out.splitlines() if l.strip() and l.strip().lower() != "name"]
                if lines:
                    return lines[0]
            except Exception:
                continue
    return platform.processor() or "unknown"


def lib_versions():
    out = {"python": sys.version.split()[0]}
    for m in ("numpy", "pandas", "scipy", "sklearn", "shap", "matplotlib", "openpyxl"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = "not installed"
    return out


def main():
    res = {"cpu": cpu_name(), "os": platform.platform(), "logical_cores": __import__("os").cpu_count()}
    res.update({f"ver_{k}": v for k, v in lib_versions().items()})
    print(f"CPU: {res['cpu']} | cores: {res['logical_cores']} | {res['os']}")
    print("Libraries:", lib_versions())

    df = C.load_person_level(CONFIDENCE_CSV, INSIDERS_CSV)
    n = len(df)
    for scen_name, catalog in C.CATALOG_SCENARIOS.items():
        for arm, w in C.ARMS.items():
            scores = C.threat_scores(df, w)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                g, _, _ = C.greedy_allocate(scores, BUDGET, catalog)
                ts.append(time.perf_counter() - t0)
            ts = np.array(ts)
            med = float(np.median(C.delta_distribution(scores, g, catalog, 1000, 0)))
            t_sa = time.perf_counter()
            for s in SA_SEEDS:
                C.sa_allocate(scores, BUDGET, g, catalog, 0.1 * med, 0.1 * med * 1e-3, SA_MAX_ITER, s)
            sa_total = time.perf_counter() - t_sa
            print(f"[{scen_name} | {arm}] greedy {ts.mean():.5f}s +/- {ts.std(ddof=1):.5f}s (5 repeats, n={n:,} persons); "
                  f"SA {sa_total:.3f}s total for {len(SA_SEEDS)} seeds ({sa_total / len(SA_SEEDS):.4f}s/seed, {SA_MAX_ITER} iterations)")
            res[f"greedy_{scen_name}_{arm}_mean_s"] = ts.mean()
            res[f"greedy_{scen_name}_{arm}_sd_s"] = ts.std(ddof=1)
            res[f"sa_{scen_name}_{arm}_total_s"] = sa_total
            res[f"sa_{scen_name}_{arm}_per_seed_s"] = sa_total / len(SA_SEEDS)

    # classifier timing: the real code path, on the real labels
    import cert_confidence_classifier_v2 as clf
    feats = pd.read_csv(CONFIDENCE_CSV)
    labels = pd.read_csv(LABELS_CSV)
    feats = feats.merge(labels, on="threat_id", how="left")
    feats["malicious"] = feats["malicious"].fillna(0).astype(int)
    print(f"\nClassifier timing on {len(feats):,} user-weeks with {int(feats['malicious'].sum()):,} positives "
          f"(labels from cert_userweek_labels.csv -- must be the 322-style insider-window count, not every insider week)")
    t0 = time.perf_counter()
    _, fold_metrics = clf.build_confidence(feats)
    cv_total = time.perf_counter() - t0
    print(f"5-fold StratifiedGroupKFold training + out-of-fold prediction: {cv_total:.4f}s (~{cv_total / len(fold_metrics):.4f}s per fold)")
    X = feats[["impact", "propagation", "evasion"]].fillna(0.0).to_numpy()
    from sklearn.ensemble import RandomForestClassifier
    final = RandomForestClassifier(**clf.RF_PARAMS).fit(X, feats["malicious"].to_numpy())
    t0 = time.perf_counter()
    final.predict_proba(X)
    inf = time.perf_counter() - t0
    print(f"Inference on {len(X):,} instances: {inf:.4f}s ({inf / len(X) * 1000:.5f} ms/instance)")
    res.update({"classifier_cv_total_s": cv_total, "classifier_cv_per_fold_s": cv_total / len(fold_metrics),
                "inference_total_s": inf, "inference_per_instance_ms": inf / len(X) * 1000,
                "n_user_weeks": len(X), "n_positive_labels": int(feats["malicious"].sum()), "n_persons": n})
    pd.DataFrame([res]).to_csv(C.out_path("cert_timing_results_v5.csv"), index=False)
    print("Saved: cert_timing_results_v5.csv")


if __name__ == "__main__":
    main()
