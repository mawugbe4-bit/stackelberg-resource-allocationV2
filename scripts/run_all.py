"""
run_all.py -- one-command run of the whole pipeline, with an automatic run manifest
-----------------------------------------------------------------------------------
    python run_all.py                         # everything, in order
    python run_all.py --from-features         # skip aggregation (start from the committed feature CSVs)
    python run_all.py --only primary,sa       # selected steps
    python run_all.py --skip timing,ablation  # all but these
    python run_all.py --smoke                 # fast smoke test (few replicates/seeds) -- NOT for results
    python run_all.py --list                  # show the steps

Paths come from config.py. Each step's console output is saved to results/logs/<step>.log. At the end the
script writes results/RUN_MANIFEST.md and results/run_manifest.json: UTC timestamps, python and library
versions, CPU model, the git commit hash (if this is a git checkout), and the status and duration of every
step. The manifest is the analysis log: it records every configuration actually run, with the date.
"""

import contextlib
import importlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as cfg
import pipeline_common as C

RES = str(cfg.RESULTS_DIR)

# name, module, constants to set on the module, description
STEPS = [
    ("aggregate", "cert_local_aggregation_v3", {}, "CERT streams -> user-week features (raw + normalised), row counts, attrition check"),
    ("classifier", "cert_confidence_classifier_v2", {}, "labels + StratifiedGroupKFold Random Forest -> out-of-fold Confidence, per-fold metrics"),
    ("shap", "cert_shap_confidence_v4", {"DATA_PATH": str(cfg.CONFIDENCE_CSV), "INSIDERS_CSV": str(cfg.INSIDERS_CSV), "OUTPUT_DIR": RES}, "SHAP magnitude and direction"),
    ("confusion", "cert_confusion_matrix_v3", {}, "confusion matrix, threshold sweep, PR curve, description of false negatives and false positives"),
    ("primary", "cert_primary_analysis_v4", {}, "point estimates (k/70, gained/lost), continuous outcomes, enrichment, person bootstrap, subgroup differences"),
    ("sa", "cert_sa_greedy_pipeline_v8", {}, "greedy + SA temperature-schedule sweep, per-seed table, spend/unspent explanation"),
    ("exact", "cert_exact_optimum_v3", {}, "exact optimum (MIP) and greedy gap"),
    ("budget", "cert_budget_sweep", {}, "budget sweep with paired bootstrap intervals"),
    ("catalog", "cert_catalog_sensitivity", {}, "catalogue sensitivity (effectiveness x IR capacity x cost; IR legacy; DLP Forrester; drop Confidence)"),
    ("features", "cert_feature_diagnostics", {}, "raw feature distributions, tenure, rank-normalisation check"),
    ("normalisation", "cert_normalisation_sensitivity", {}, "min-max vs rank normalisation: weight effect and normalisation effect with bootstrap intervals"),
    ("ablation", "cert_sa_ablation", {}, "SA ablation grid (proposals, seeding, capacity constraints)"),
    ("bruteforce", "verify_formulation_bruteforce", {}, "MIP formulation vs exhaustive brute force (specified instances)"),
    ("timing", "cert_timing_v5", {}, "timings, CPU model, library versions"),
    ("ahp_weights", "ahp_weights_by_institution", {"SURVEY_XLSX": str(cfg.SURVEY_XLSX)}, "AHP weights by institution type"),
    ("ahp_cr", "ahp_cr_sensitivity", {"SURVEY_XLSX": str(cfg.SURVEY_XLSX)}, "AHP consistency-ratio sensitivity"),
    ("ahp_split", "ahp_split_seed_sensitivity", {"SURVEY_XLSX": str(cfg.SURVEY_XLSX)}, "AHP split-seed sensitivity"),
]
AHP = {"ahp_weights", "ahp_cr", "ahp_split"}
SMOKE = {  # overrides that make a step fast; for smoke tests only
    "cert_primary_analysis_v4": {"N_BOOTSTRAP": 20}, "cert_sa_greedy_pipeline_v8": {"SA_SEEDS": [42, 123], "SA_MAX_ITER": 800, "EXACT_TIME_LIMIT_S": 5},
    "cert_exact_optimum_v3": {"TIME_LIMIT_S": 10}, "cert_budget_sweep": {"N_BOOT": 10, "BUDGETS": [3500, 7000, 30000]},
    "cert_catalog_sensitivity": {"N_BOOT": 10}, "cert_normalisation_sensitivity": {"N_BOOT": 20}, "cert_sa_ablation": {"SEEDS": [42, 123], "MAX_ITER": 800, "EXACT_TIME_LIMIT_S": 5},
    "verify_formulation_bruteforce": {"N_INSTANCES": 4},
}


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.REPO_ROOT, capture_output=True, text=True, timeout=15)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cfg.REPO_ROOT, capture_output=True, text=True, timeout=15).stdout.strip()
        return (out.stdout.strip() or "unknown") + (" (UNCOMMITTED CHANGES PRESENT)" if dirty else "")
    except Exception:
        return "not a git checkout"


def environment():
    import cert_timing_v5 as t
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": t.cpu_name(),
            "logical_cores": os.cpu_count(), "libraries": t.lib_versions(), "git_commit": git_commit()}


def run_step(name, modname, consts, smoke):
    mod = importlib.import_module(modname)
    mod = importlib.reload(mod)
    for k, v in {**consts, **(SMOKE.get(modname, {}) if smoke else {})}.items():
        setattr(mod, k, v)
    buf = io.StringIO()
    t0 = time.time()
    status, err = "ok", ""
    try:
        with contextlib.redirect_stdout(Tee(sys.__stdout__, buf)):
            mod.main()
    except SystemExit as e:
        status, err = "failed", f"SystemExit: {e}"
    except Exception:
        status, err = "failed", traceback.format_exc()
    dur = time.time() - t0
    os.makedirs(os.path.join(RES, "logs"), exist_ok=True)
    with open(os.path.join(RES, "logs", f"{name}.log"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue() + (f"\n\nFAILED:\n{err}\n" if err else ""))
    return status, err, dur


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--list" in argv:
        for n, m, _, d in STEPS:
            print(f"  {n:12s} {m:34s} {d}")
        return 0
    only = next((a.split("=", 1)[1] for a in argv if a.startswith("--only=")), None)
    skip = next((a.split("=", 1)[1] for a in argv if a.startswith("--skip=")), "")
    for i, a in enumerate(argv):                    # allow "--only x,y" as well as "--only=x,y"
        if a == "--only" and i + 1 < len(argv): only = argv[i + 1]
        if a == "--skip" and i + 1 < len(argv): skip = argv[i + 1]
    only = set(only.split(",")) if only else None
    skip = set(skip.split(",")) if skip else set()
    if "--from-features" in argv:
        skip.add("aggregate")
    smoke = "--smoke" in argv

    os.makedirs(RES, exist_ok=True)
    os.makedirs(str(cfg.WORK_DIR), exist_ok=True)
    os.chdir(RES)
    C.OUTPUT_DIR = RES
    env = environment()
    try:                                            # lockfile of the environment that ACTUALLY ran the analysis
        fz = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
        with open(os.path.join(RES, "pip_freeze.txt"), "w", encoding="utf-8") as f:
            f.write(fz)
    except Exception as e:
        print("pip freeze failed:", e)
    print(f"Run started {datetime.now(timezone.utc).isoformat()}  |  commit {env['git_commit']}  |  CPU {env['cpu']}")
    records, stop = [], False
    for name, modname, consts, desc in STEPS:
        if (only and name not in only) or name in skip:
            continue
        if name in AHP and not os.path.exists(cfg.SURVEY_XLSX):
            print(f"\n--- {name}: SKIPPED (survey file not found at {cfg.SURVEY_XLSX}; it is deliberately not in the repo)")
            records.append({"step": name, "module": modname, "status": "skipped (survey file absent)", "seconds": 0})
            continue
        if stop:
            records.append({"step": name, "module": modname, "status": "not run (earlier step failed)", "seconds": 0})
            continue
        print("\n" + "=" * 100 + f"\nSTEP {name}: {desc}\n" + "=" * 100)
        status, err, dur = run_step(name, modname, consts, smoke)
        print(f"\n--- {name}: {status} in {dur:.1f}s")
        records.append({"step": name, "module": modname, "status": status, "seconds": round(dur, 1),
                        "error": err.strip().splitlines()[-1] if err else ""})
        if status == "failed" and name in ("aggregate", "classifier"):
            stop = True                               # everything downstream depends on these
    ok = sum(r["status"] == "ok" for r in records)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "mode": "SMOKE TEST (not results)" if smoke else "full",
                "environment": env, "steps": records, "steps_ok": ok, "steps_total": len(records)}
    with open(os.path.join(RES, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(RES, "RUN_MANIFEST.md"), "w", encoding="utf-8") as f:
        f.write(f"# Run manifest\n\nMode: **{manifest['mode']}**  \nUTC: {manifest['started_utc']}  \n"
                f"Git commit: `{env['git_commit']}`  \nPython {env['python']} on {env['platform']}  \nCPU: {env['cpu']} "
                f"({env['logical_cores']} logical cores)  \nLibraries: {env['libraries']}\n\n"
                "| step | module | status | seconds |\n|---|---|---|---|\n" +
                "\n".join(f"| {r['step']} | {r['module']} | {r['status']} | {r['seconds']} |" for r in records) +
                f"\n\n{ok} of {len(records)} steps completed. Each step's console output: `results/logs/<step>.log`.\n")
    print(f"\nManifest written: {os.path.join(RES, 'RUN_MANIFEST.md')}  ({ok}/{len(records)} steps ok)")
    return 0 if ok == len(records) or all(r["status"].startswith(("ok", "skipped")) for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
