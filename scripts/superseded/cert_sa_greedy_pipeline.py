"""
AHP-Calibrated Stackelberg-SA Allocation Pipeline
--------------------------------------------------
Runs the greedy heuristic and Simulated Annealing (SA) allocator over a
threat-resource dataset (intended: CERT r4.2-derived, ~4,000 user-week
records) under both the arbitrary baseline weights and the AHP
calibration-wave weights, and prints an environment block you can paste
directly into Methods §2.9.

WHAT TO EDIT BEFORE RUNNING
----------------------------
1. DATA_PATH        -> path to your CERT-derived, feature-engineered CSV.
                        Expected columns (one row per threat-resource pairing):
                          threat_id, resource_id,
                          impact, propagation, confidence, evasion,
                          cost   (resource cost, for the budget constraint)
2. BUDGET            -> total resource budget for the allocation round.
3. BASELINE_WEIGHTS   -> the "no calibration" comparator, set to equal
                        weighting (0.25 each), as justified in Methods
                        Section 2.7. No expert-assigned vector was fixed
                        prior to this study, and deriving a second weight
                        vector from the same practitioner population used
                        for AHP calibration would confound the baseline
                        with the treatment -- so equal weighting is the
                        deliberate, methodologically justified baseline,
                        not a placeholder. Leave this as-is unless your
                        supervisor specifically wants a different
                        justified baseline.

Everything else (calibration weights, SA hyperparameters) is already set
from your proposal/Methods draft and the AHP survey computation.
"""

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 0. CONFIG -- edit this block for your real run
# ----------------------------------------------------------------------
DATA_PATH = "cert_threat_resource_features.csv"   # <-- replace with real path
BUDGET = 100.0                                     # <-- replace with real budget

# AHP calibration-wave weights, computed from the real survey (Table 1)
CALIBRATED_WEIGHTS = {
    "impact": 0.224,
    "propagation": 0.252,
    "confidence": 0.247,
    "evasion": 0.278,
}

# "No calibration" baseline: equal weighting (see Methods Section 2.7 for
# why this, rather than a second expert-elicited vector, is the correct
# comparator here).
BASELINE_WEIGHTS = {
    "impact": 0.25,
    "propagation": 0.25,
    "confidence": 0.25,
    "evasion": 0.25,
}

SA_PARAMS = dict(
    initial_temp=1000.0,
    cooling_rate=0.95,
    min_temp=0.01,
    max_iter=10000,
)

RANDOM_SEEDS = [42, 123, 456, 789, 1010]  # for the multi-run reproducibility
# control in Methods §2.10 -- edit the seed list/count to match what you
# actually use.


# ----------------------------------------------------------------------
# 1. ENVIRONMENT CAPTURE -- prints the block to paste into Methods §2.9
# ----------------------------------------------------------------------
def get_package_version(pkg_name: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return "not installed"


def get_ram_gb() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return f"{kb / 1024 / 1024:.1f} GB"
        elif platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "hw.memsize"]).decode()
            bytes_ = int(out.strip().split(":")[1])
            return f"{bytes_ / 1024**3:.1f} GB"
    except Exception:
        pass
    return "unknown (record manually, e.g. from Task Manager / About This Mac)"


def get_cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode().strip()
            return out
    except Exception:
        pass
    return platform.processor() or "unknown"


def capture_environment() -> dict:
    return {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_model": get_cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "ram": get_ram_gb(),
        "numpy_version": get_package_version("numpy"),
        "pandas_version": get_package_version("pandas"),
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def print_methods_2_9_block(env: dict, sa_params: dict) -> None:
    print("\n" + "=" * 70)
    print("PASTE THIS BLOCK INTO METHODS SECTION 2.9")
    print("=" * 70)
    print(
        f"All experiments were conducted on a {env['cpu_model']} "
        f"({env['cpu_count_logical']} logical cores, {env['ram']} RAM) "
        f"running {env['os']}. The software environment used Python "
        f"{env['python_version']}, NumPy {env['numpy_version']}, and "
        f"pandas {env['pandas_version']}. Simulated Annealing was configured "
        f"with an initial temperature of {sa_params['initial_temp']:.0f}, "
        f"a geometric cooling rate of {sa_params['cooling_rate']}, a minimum "
        f"temperature of {sa_params['min_temp']}, and a maximum of "
        f"{sa_params['max_iter']:,} iterations. Run captured on "
        f"{env['captured_at']}."
    )
    print("=" * 70 + "\n")


# ----------------------------------------------------------------------
# 2. UTILITY FUNCTION  U(T_i, R_j) = sum(alpha_k * feature_k)
# ----------------------------------------------------------------------
def utility(row: pd.Series, weights: dict) -> float:
    return (
        weights["impact"] * row["impact"]
        + weights["propagation"] * row["propagation"]
        + weights["confidence"] * row["confidence"]
        + weights["evasion"] * row["evasion"]
    )


# ----------------------------------------------------------------------
# 3. GREEDY HEURISTIC
# ----------------------------------------------------------------------
def greedy_allocate(df: pd.DataFrame, weights: dict, budget: float) -> pd.DataFrame:
    scored = df.copy()
    scored["utility"] = scored.apply(utility, axis=1, weights=weights)
    scored["ratio"] = scored["utility"] / scored["cost"].replace(0, np.nan)
    scored = scored.sort_values("ratio", ascending=False)

    chosen, spent = [], 0.0
    for _, row in scored.iterrows():
        if spent + row["cost"] <= budget:
            chosen.append(row)
            spent += row["cost"]
    return pd.DataFrame(chosen)


# ----------------------------------------------------------------------
# 4. SIMULATED ANNEALING
# ----------------------------------------------------------------------
def sa_objective(selection_mask: np.ndarray, df: pd.DataFrame, weights: dict) -> float:
    return float((df["utility"].to_numpy() * selection_mask).sum())


def total_cost(selection_mask: np.ndarray, df: pd.DataFrame) -> float:
    return float((df["cost"].to_numpy() * selection_mask).sum())


def simulated_annealing(
    df: pd.DataFrame,
    weights: dict,
    budget: float,
    initial_temp: float,
    cooling_rate: float,
    min_temp: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["utility"] = df.apply(utility, axis=1, weights=weights)

    n = len(df)
    # start from a feasible all-zero selection
    current = np.zeros(n, dtype=int)
    current_obj = sa_objective(current, df, weights)
    best, best_obj = current.copy(), current_obj

    temp = initial_temp
    it = 0
    while temp > min_temp and it < max_iter:
        candidate = current.copy()
        flip = rng.integers(0, n)
        candidate[flip] = 1 - candidate[flip]

        if total_cost(candidate, df) <= budget:
            candidate_obj = sa_objective(candidate, df, weights)
            delta = candidate_obj - current_obj
            if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-9)):
                current, current_obj = candidate, candidate_obj
                if current_obj > best_obj:
                    best, best_obj = current.copy(), current_obj

        temp *= cooling_rate
        it += 1

    return best, best_obj


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------
def main():
    env = capture_environment()
    print_methods_2_9_block(env, SA_PARAMS)

    if not os.path.exists(DATA_PATH):
        print(
            f"[NOTE] '{DATA_PATH}' not found -- this run only demonstrates "
            f"the environment capture above. Point DATA_PATH at your real "
            f"CERT-derived feature file to get actual allocation results.\n"
        )
        return

    df = pd.read_csv(DATA_PATH)
    results = {}

    for label, weights in [("baseline", BASELINE_WEIGHTS), ("calibrated", CALIBRATED_WEIGHTS)]:
        greedy_sel = greedy_allocate(df, weights, BUDGET)
        greedy_util = greedy_sel["utility"].sum() if len(greedy_sel) else 0.0

        sa_runs = []
        for seed in RANDOM_SEEDS:
            _, obj = simulated_annealing(df, weights, BUDGET, seed=seed, **SA_PARAMS)
            sa_runs.append(obj)

        results[label] = {
            "greedy_utility": greedy_util,
            "greedy_n_selected": int(len(greedy_sel)),
            "sa_utility_mean": float(np.mean(sa_runs)),
            "sa_utility_sd": float(np.std(sa_runs)),
            "sa_runs": sa_runs,
        }

    with open("allocation_results.json", "w") as f:
        json.dump({"environment": env, "results": results}, f, indent=2)

    print("Results written to allocation_results.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
