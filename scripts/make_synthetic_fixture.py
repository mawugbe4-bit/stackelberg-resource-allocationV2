"""make_synthetic_fixture.py -- builds a SYNTHETIC dataset with the study's structure, for TESTING THE
MACHINERY ONLY (never for results): 1,000 people, 70 planted insiders in scenarios of 30/30/10, variable
tenure (Scenario-1 insiders leave early), insider-window weeks carry a weak signal. Usage:
    python make_synthetic_fixture.py <out_dir>
Writes <out_dir>/cert_threat_resource_features.csv and <out_dir>/insiders.csv"""
import sys, os
import numpy as np, pandas as pd

def build(out_dir, seed=2026, n_users=1000):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    start0 = pd.Timestamp("2010-01-04")
    users = [f"U{i:04d}" for i in range(n_users)]
    ins_users = list(rng.choice(users, 70, replace=False))
    scen = {u: (1 if k < 30 else 2 if k < 60 else 3) for k, u in enumerate(ins_users)}
    rows, windows = [], []
    for u in users:
        s = scen.get(u, 0)
        n_weeks = {1: int(rng.integers(14, 30)), 2: int(rng.integers(55, 78)), 3: int(rng.integers(40, 60))}.get(s, int(rng.integers(25, 78)))
        first = int(rng.integers(0, 78 - n_weeks + 1)) if n_weeks < 78 else 0
        wk = list(range(first, first + n_weeks))
        if s:
            span = {1: 2, 2: 8, 3: 1}[s]
            w_start_idx = wk[-span]
            ws = start0 + pd.Timedelta(days=7 * w_start_idx) + pd.Timedelta(days=2)
            we = start0 + pd.Timedelta(days=7 * wk[-1]) + pd.Timedelta(days=4)
            windows.append({"dataset": "4.2", "scenario": s, "details": "x", "user": u,
                            "start": ws.strftime("%Y-%m-%d %H:%M:%S"), "end": we.strftime("%Y-%m-%d %H:%M:%S")})
        for w in wk:
            wsd = start0 + pd.Timedelta(days=7 * w)
            in_win = s and w >= wk[-{1: 2, 2: 8, 3: 1}[s]]
            imp = rng.gamma(2.0, 20.0) * (1.8 if in_win else 1.0)
            prop = rng.gamma(2.5, 4.0) * (0.85 if in_win else 1.0)
            eva = float(np.clip(rng.beta(1.3, 6.0) + (0.25 if in_win else 0.0) * rng.random(), 0, 1))
            rows.append({"threat_id": f"{u}_{wsd.date()}/{(wsd + pd.Timedelta(days=6)).date()}",
                         "impact_raw": imp, "propagation_raw": prop, "evasion_raw": eva})
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"threat_id": df["threat_id"], "resource_id": "TBD"})
    for c in ("impact", "propagation", "evasion"):
        v = df[c + "_raw"]; out[c] = (v - v.min()) / (v.max() - v.min())
    out["confidence"] = np.nan; out["cost"] = 1.0
    out = out[["threat_id", "resource_id", "impact", "propagation", "confidence", "evasion", "cost"]]
    out.to_csv(os.path.join(out_dir, "cert_threat_resource_features.csv"), index=False)
    df.to_csv(os.path.join(out_dir, "cert_threat_resource_features_raw.csv"), index=False)
    pd.DataFrame(windows).to_csv(os.path.join(out_dir, "insiders.csv"), index=False)
    return out, pd.DataFrame(windows)

if __name__ == "__main__":
    o, w = build(sys.argv[1] if len(sys.argv) > 1 else "fixture")
    print(f"synthetic fixture: {len(o):,} user-weeks, {o.threat_id.str.split('_').str[0].nunique()} users, {len(w)} insider windows")
