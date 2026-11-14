"""
Formulation verification against exhaustive brute force -- the instances are SPECIFIED here
------------------------------------------------------------------------------------------
Supervisor item (Q8/53): the earlier claim that greedy "matched the brute-force optimum on small
synthetic instances" never said what those instances were. This script fixes that.

Specification (all drawn from numpy default_rng(SEED), SEED = 20260930):
  count      : N_INSTANCES = 40
  size       : n = 7 persons x 5 resources (each person gets one resource or none -> 6^7 = 279,936 allocations)
  scores     : U_i ~ Uniform(0, 1)
  resources  : cost_r ~ Uniform(1, 15), effectiveness_r ~ Uniform(0.02, 0.60), capacity_r ~ integer {1,2,3}
  budget     : B ~ Uniform(15, 40)   (tight: constraints bind in nearly every instance)
For each instance it enumerates every allocation, then checks that (i) the exact MIP (scipy.optimize.milp /
HiGHS) returns the brute-force optimum, and (ii) the MIP's returned assignment is feasible and achieves its
reported value; and it records the greedy value and gap. Every instance's full specification and results are
saved to bruteforce_instances.csv so the instances can be reproduced and inspected.

What this does and does not show: it verifies the MIP FORMULATION and quantifies greedy's gap on small tight
instances. It does not show that greedy is optimal at n = 1,000 (use cert_exact_optimum_v3.py for that).
"""

import itertools
import numpy as np
import pandas as pd
import pipeline_common as C

SEED = 20260930
N_INSTANCES = 40
N_PERSONS = 7
N_RES = 5


def brute_force(scores, catalog, budget):
    costs, effs, caps = catalog.cost.values, catalog.effectiveness.values, catalog.capacity.values
    best = 0.0
    for combo in itertools.product(range(-1, N_RES), repeat=len(scores)):
        if sum(costs[r] for r in combo if r >= 0) > budget:
            continue
        if any(sum(1 for r in combo if r == k) > caps[k] for k in range(N_RES)):
            continue
        best = max(best, sum(scores[i] * effs[r] for i, r in enumerate(combo) if r >= 0))
    return best


def main():
    rng = np.random.default_rng(SEED)
    rows, n_ok = [], 0
    for inst in range(N_INSTANCES):
        scores = rng.random(N_PERSONS)
        cat = pd.DataFrame([{"resource_id": f"r{k}", "cost": rng.uniform(1, 15), "effectiveness": rng.uniform(0.02, 0.60),
                             "capacity": int(rng.integers(1, 4))} for k in range(N_RES)])
        budget = float(rng.uniform(15, 40))
        bf = brute_force(scores, cat, budget)
        val, status, _, _, assign = C.solve_exact(scores, budget, cat)
        spent = sum(cat.cost.values[r] for r in assign if r >= 0)
        feasible = spent <= budget + 1e-9 and all((assign == k).sum() <= cat.capacity.values[k] for k in range(N_RES))
        realised = sum(scores[i] * cat.effectiveness.values[r] for i, r in enumerate(assign) if r >= 0)
        ok = abs(bf - val) < 1e-9 and feasible and abs(realised - val) < 1e-9 and status == "Optimal"
        n_ok += ok
        g, _, _ = C.greedy_allocate(scores, budget, cat)
        gv = C.allocation_value(scores, g, cat)
        rows.append({"instance": inst, "n_persons": N_PERSONS, "n_resources": N_RES, "budget": budget,
                     "scores": ";".join(f"{s:.6f}" for s in scores),
                     "costs": ";".join(f"{c:.4f}" for c in cat.cost), "effectiveness": ";".join(f"{e:.4f}" for e in cat.effectiveness),
                     "capacities": ";".join(str(c) for c in cat.capacity),
                     "bruteforce_optimum": bf, "mip_value": val, "mip_matches": ok, "greedy_value": gv,
                     "greedy_gap_pct": 100 * (bf - gv) / bf if bf > 0 else 0.0})
    R = pd.DataFrame(rows)
    R.to_csv(C.out_path("bruteforce_instances.csv"), index=False)
    gaps = R.greedy_gap_pct.to_numpy()
    print(f"{N_INSTANCES} instances (n={N_PERSONS}, {N_RES} resources, seed {SEED}): MIP matched brute force on {n_ok}/{N_INSTANCES}")
    print(f"Greedy gap on these tight instances: optimal in {(gaps < 1e-9).sum()}/{N_INSTANCES}; "
          f"median {np.median(gaps):.2f}%, mean {gaps.mean():.2f}%, max {gaps.max():.2f}%")
    assert n_ok == N_INSTANCES, "formulation verification FAILED"
    print("Saved: bruteforce_instances.csv")


if __name__ == "__main__":
    main()
