"""tests_pipeline.py -- sanity and regression tests for pipeline_common (run: python tests_pipeline.py)"""
import itertools, os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)      # imports work from any working directory
import pipeline_common as C


def find_script(name):
    """Locate a script next to this file, or in superseded/; None if absent."""
    for p in (os.path.join(HERE, name), os.path.join(HERE, "superseded", name)):
        if os.path.exists(p):
            return p
    return None

def test_weights():
    for name, w in C.ARMS.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, (name, sum(w.values()))
    print("PASS weights sum to exactly 1 (calibrated = %.4f)" % sum(C.CALIBRATED_WEIGHTS.values()))

def test_catalog():
    cat = C.make_catalog()
    assert list(cat.resource_id) == C.RESOURCE_ORDER
    assert abs(cat.set_index("resource_id").loc["ir", "effectiveness"] - 248000/4_440_000) < 1e-12
    assert C.make_catalog(training_capacity=5).set_index("resource_id").loc["training", "capacity"] == 5
    print("PASS catalogue:\n" + cat.round(5).to_string(index=False))

def test_greedy_feasible_and_deterministic():
    rng = np.random.default_rng(1); sc = rng.random(500)
    cat = C.make_catalog(training_capacity=5)
    a1, u1, s1 = C.greedy_allocate(sc, 7000.0, cat); a2, u2, s2 = C.greedy_allocate(sc, 7000.0, cat)
    assert (a1 == a2).all() and s1 <= 7000.0 + 1e-9
    assert all(u1[r] <= cat.capacity[r] for r in range(5)) and (np.bincount(a1[a1 >= 0], minlength=5) == u1).all()
    print(f"PASS greedy feasible/deterministic (spent {s1:.2f}/7000, used {dict(zip(cat.resource_id, u1))})")

def test_greedy_matches_legacy():
    import importlib.util
    path = find_script("cert_sa_greedy_pipeline_v6.py")
    if path is None:
        print("SKIP legacy-greedy regression (superseded/cert_sa_greedy_pipeline_v6.py not present)"); return
    sp = importlib.util.spec_from_file_location("v6", path)
    v6 = importlib.util.module_from_spec(sp); sys.modules["v6"] = v6; sp.loader.exec_module(v6)
    rng = np.random.default_rng(2); sc = rng.random(800)
    for ts in (100_000, 5):
        cat = C.make_catalog(training_capacity=ts)
        old, _, spent_old = v6.greedy_allocate(sc, 7000.0, cat)
        new, _, spent_new = C.greedy_allocate(sc, 7000.0, cat)
        assert (old.astype(int) == new.astype(int)).all() and abs(spent_old - spent_new) < 1e-9
    print("PASS shared greedy reproduces legacy greedy exactly (2 catalogues)")

def test_exact_vs_bruteforce():
    rng = np.random.default_rng(11)
    for trial in range(8):
        n = 7; sc = rng.random(n)
        cat = pd.DataFrame([{"resource_id": k, "cost": rng.uniform(1, 15), "effectiveness": rng.uniform(.02, .6),
                             "capacity": int(rng.integers(1, 4))} for k in "abcde"])
        B = float(rng.uniform(15, 40)); costs, effs, caps = cat.cost.values, cat.effectiveness.values, cat.capacity.values
        best = 0.0
        for combo in itertools.product(range(-1, 5), repeat=n):
            if sum(costs[r] for r in combo if r >= 0) > B: continue
            if any(sum(1 for r in combo if r == k) > caps[k] for k in range(5)): continue
            best = max(best, sum(sc[i]*effs[r] for i, r in enumerate(combo) if r >= 0))
        val, status, _, ub, assign = C.solve_exact(sc, B, cat)
        spent = sum(costs[r] for r in assign if r >= 0)
        assert abs(val - best) < 1e-9 and spent <= B + 1e-9 and status == "Optimal"
    print("PASS exact MIP == brute force on 8 random tight instances (n=7, 5 resources)")

def test_auc():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3); y = (rng.random(400) < .08).astype(int); s = rng.random(400) + 0.3*y
    s[:20] = 0.5; s[20:40] = 0.5     # ties
    assert abs(C.auc_score(s, y) - roc_auc_score(y, s)) < 1e-12
    print("PASS AUC equals sklearn roc_auc_score (including ties)")

def test_sa_basic():
    rng = np.random.default_rng(4); sc = np.clip(rng.beta(1.2, 6, 600) * 1.6, 0, 1)
    cat = C.make_catalog(training_capacity=5)
    g, u, s = C.greedy_allocate(sc, 7000.0, cat); gv = C.allocation_value(sc, g, cat)
    r = C.sa_allocate(sc, 7000.0, g, cat, 0.002, 2e-6, 3000, seed=5)
    assert r["iterations"] == 3000 and r["best_value"] >= gv - 1e-12
    ba = r["best_assigned"]; used = np.bincount(ba[ba >= 0], minlength=5)
    assert sum(cat.cost[k] * used[k] for k in range(5)) <= 7000.0 + 1e-6 and all(used[k] <= cat.capacity[k] for k in range(5))
    r2 = C.sa_allocate(sc, 7000.0, g, cat, 0.002, 2e-6, 3000, seed=5)
    assert r2["best_value"] == r["best_value"] and r2["accepted"] == r["accepted"]
    print(f"PASS SA feasible, deterministic, runs exactly max_iter (gain {r['best_value']-gv:+.6f}, accepted {r['accepted']})")

def test_boundary_pool_bug_in_legacy():
    """Legacy v6 built the 'marginally admitted' half of the boundary pool by taking the 500 SMALLEST
    values with unassigned threats set to -inf, so unassigned threats sorted FIRST and were then filtered
    out -> the admitted half was EMPTY whenever >= 500 threats were unassigned (always, in this study)."""
    rng = np.random.default_rng(6); sc = rng.random(1000); cat = C.make_catalog()
    g, _, _ = C.greedy_allocate(sc, 7000.0, cat)
    effs = cat.effectiveness.values
    legacy = np.array([sc[i]*effs[r] if r >= 0 else -np.inf for i, r in enumerate(g)])
    legacy_admitted = np.argsort(legacy)[:500]; legacy_admitted = legacy_admitted[g[legacy_admitted] != -1]
    fixed = np.array([sc[i]*effs[r] if r >= 0 else np.inf for i, r in enumerate(g)])
    fixed_admitted = np.argsort(fixed)[:500]; fixed_admitted = fixed_admitted[g[fixed_admitted] != -1]
    print(f"INFO unassigned={int((g==-1).sum())}: legacy 'admitted' pool size = {len(legacy_admitted)}, fixed = {len(fixed_admitted)}")
    assert len(legacy_admitted) == 0 and len(fixed_admitted) > 0
    print("PASS confirmed: legacy boundary pool had NO marginally-admitted threats; shared module fixes it")

def test_label_consistency():
    """Every script that labels user-weeks must give identical labels (the timing script once did not)."""
    import importlib.util, sys, tempfile, os
    import make_synthetic_fixture as mf
    import cert_confidence_classifier_v2 as clf
    d = tempfile.mkdtemp()
    feats, wins = mf.build(d, seed=7, n_users=150)
    ins = clf.load_insider_windows(os.path.join(d, "insiders.csv"))
    ref = clf.label_features(feats, ins, verbose=False)["malicious"].to_numpy()
    for modname in ("cert_confusion_matrix_v2", "cert_shap_confidence_v4"):
        path = find_script(modname + ".py")
        assert path is not None, f"{modname}.py not found next to tests_pipeline.py or in superseded/"
        sp = importlib.util.spec_from_file_location(modname, path)
        m = importlib.util.module_from_spec(sp); sys.modules[modname] = m; sp.loader.exec_module(m)
        m.DATA_PATH = os.path.join(d, "cert_threat_resource_features.csv"); m.INSIDERS_CSV = os.path.join(d, "insiders.csv")
        other = m.load_labeled_data()["malicious"].to_numpy()
        assert (other == ref).all(), f"{modname} labels differ from the classifier's"
    n_pos, n_insider_rows = int(ref.sum()), int(feats.threat_id.str.split("_").str[0].isin(set(wins.user)).sum())
    assert n_pos < n_insider_rows      # window-overlap labels are a strict subset of 'every week of an insider'
    print(f"PASS label consistency: classifier == confusion matrix == SHAP labels ({n_pos} positives; the old timing rule would have used {n_insider_rows})")

if __name__ == "__main__":
    for t in [test_weights, test_catalog, test_greedy_feasible_and_deterministic, test_greedy_matches_legacy,
              test_exact_vs_bruteforce, test_auc, test_sa_basic, test_boundary_pool_bug_in_legacy, test_label_consistency]:
        t()
    print("\nALL TESTS PASSED")
