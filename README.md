# AHP-calibrated allocation of cybersecurity controls: person-level evaluation on CERT r4.2

Code for the PhD thesis *"A Stackelberg Game-Theoretic Model for Optimizing Cybersecurity Resource Allocation in
Resource-Constrained Organizations: A Causal ML and Simulated Annealing Approach"* (Dennis Redeemer Korda,
PG5145120, Dept. of Computer Science, KNUST). The title is under review; this README describes what the code
actually does, which is narrower than the title: a weighted-utility, budget- and capacity-constrained assignment
model whose criterion weights are either equal or AHP-calibrated, evaluated on the CERT r4.2 insider-threat corpus.
No game is solved and no causal estimator is used.

## One-command run

```bash
pip install -r requirements.txt
# edit the three paths marked "<-- EDIT" in scripts/config.py, once
cd scripts
python tests_pipeline.py              # ~1 min: sanity + regression tests (no CERT data needed)
python run_all.py                     # whole pipeline, in order; writes results/ and RUN_MANIFEST.md
python run_all.py --from-features     # skip the slow CERT aggregation, start from results/cert_threat_resource_features*.csv
python run_all.py --smoke --from-features   # fast smoke test of the machinery (NOT results)
```

`run_all.py` records, in `results/RUN_MANIFEST.md` and `results/run_manifest.json`: UTC time, git commit hash,
python/library versions, CPU model, and the status and duration of every step. It also writes `results/pip_freeze.txt`,
the lockfile of the environment that actually produced the results. Each step's console output is in `results/logs/`.

## Pipeline

| step | script | what it does |
|---|---|---|
| aggregate | `cert_local_aggregation_v3.py` | streams the CERT archive; user-week features (raw + normalised); row counts at every step; zero-attrition check; after-hours = outside [06:00, 20:00) |
| classifier | `cert_confidence_classifier_v2.py` | labels user-weeks from the insider windows; class-weighted Random Forest, 5-fold StratifiedGroupKFold grouped by user; out-of-fold Confidence; **per-fold metrics**; exports labels |
| shap | `cert_shap_confidence_v4.py` | SHAP magnitude and direction (value-SHAP correlation); prints the baseline identity that explains the global signed mean |
| confusion | `cert_confusion_matrix_v3.py` | confusion matrix at 0.5, **threshold sweep, precision-recall curve**, description of false negatives (by scenario) and false positives (how many people, how concentrated) |
| primary | `cert_primary_analysis_v4.py` | **exact k/70 per arm**, insiders gained/lost, Jaccard, Spearman, AUC of U, effectiveness-weighted coverage, enrichment; person-level cluster bootstrap (B=1000) with percentile intervals for every outcome and subgroup |
| sa | `cert_sa_greedy_pipeline_v8.py` | greedy + SA; temperature from the measured scale of objective changes; 6-schedule sweep incl. the legacy T0=1000; per-seed table; spend/unspent explanation |
| exact | `cert_exact_optimum_v3.py` | exact optimum by MIP (HiGHS via `scipy.optimize.milp`), proven vs best-found, greedy gap |
| budget | `cert_budget_sweep.py` | B from $1k to $250k, both scenarios and arms, paired bootstrap intervals, figure |
| catalog | `cert_catalog_sensitivity.py` | 27-variant grid (effectiveness x IR capacity x cost) + IR legacy contrast + DLP Forrester + Confidence dropped |
| features | `cert_feature_diagnostics.py` | raw feature distributions, tenure by group, rank-normalisation robustness check |
| normalisation | `cert_normalisation_sensitivity.py` | min-max vs percentile-rank normalisation; paired bootstrap intervals for the weight effect under each and for the normalisation effect |
| ablation | `cert_sa_ablation.py` | boundary vs uniform proposals; greedy-seeded vs empty start; capacity constraints on vs off |
| bruteforce | `verify_formulation_bruteforce.py` | MIP formulation vs exhaustive enumeration on 40 specified instances |
| timing | `cert_timing_v5.py` | allocation and classifier timings, CPU model, library versions |
| ahp_* | `ahp_weights_by_institution.py`, `ahp_cr_sensitivity.py`, `ahp_split_seed_sensitivity.py` | AHP analyses (skipped if the survey file is absent; see Data) |

Everything shares `pipeline_common.py` (weights, catalogue, person-level loader, allocators, metrics), so copies of
the same logic cannot drift apart. `scripts/superseded/` holds earlier versions for history; **no number in the
thesis should come from them.**

## Catalogue (Table 2) and its provenance

Every effectiveness value = (average per-breach saving, USD) / (global average breach cost, USD 4.44M, IBM 2025), all on the
**same basis**: IBM's "each factor in isolation vs the average". Costs are per person per year; flat organisational fees are
divided by CERT's ~1,000 users.

| control | cost low / central / high | saving (USD) | effectiveness | source |
|---|---|---|---|---|
| Security awareness training | 12 / 24 / 36 | 192,266 | 0.0433 | IBM 2025 Fig. 39 "Employee training"; KnowBe4/Proofpoint pricing |
| SIEM / enhanced logging | 12 / 36 / 60 | 212,061 | 0.0478 | IBM 2025 Fig. 39 "Security analytics or SIEM"; SIEM cost calculators |
| Access review / IAM | 24 / 114 / 204 | 189,838 | 0.0428 | IBM 2025 Fig. 39 "IAM"; Okta pricing |
| DLP | 20 / 63.25 / 200 | 157,456 | 0.0355 | IBM 2025 Fig. 39 "Data security and protection software"; mean of 4 named vendors (34, 52, 96, 71) |
| Incident response | 5 / 15 / 25 | 248,000 | 0.0559 | IBM 2024 (IBM Think summary), IR team, same in-isolation basis |

Capacities: SIEM 10 and IR 10 (SANS SOC Survey 2-10 staff), Access review 5 (NinjaOne 1 admin per 200 users), DLP unconstrained;
**Training capacity is the one unsourced value**, so two scenarios are always reported together: A = unconstrained,
B = 5.

**What is NOT in the headline catalogue, and why.** Earlier drafts used 0.3378 for IR ($1.5M / $4.44M). That figure comes
from IBM's older "IR team *plus tested plan* versus neither" contrast, a different and larger measure than the
"factor in isolation" figures used for the other controls, and the 2025 report contains no IR entry at all. It is kept
as an explicit sensitivity case (`IR_legacy_contrast` in `catalog_sensitivity.csv`). IBM's Figure 39 caption states its
differences are measured from the USD 4.88M average although the 2025 headline average is USD 4.44M; the choice of
denominator scales all effectiveness values by 0.91 and is covered by the effectiveness-scale sensitivity.
The per-breach, organisation-level saving is applied as a per-person proportional reduction; that conversion is an
assumption, not a measurement.

**Budget.** B = 7,000 is $7 per person per year in this catalogue. It was inherited from an earlier illustrative
catalogue and has no independent justification; IANS (2025) reports about $3,800 per employee for a whole security
programme in small and mid-market firms. `cert_budget_sweep.py` therefore reports the whole range, and coverage
saturates (everyone covered) well below $100k.

## Known limitations (stated, not resolved)

- CERT r4.2 is a synthetic US corporate corpus; the weights come from 46 consistent Ghanaian HEI practitioners (of 105),
  recruited without a formal sampling frame. No empirical bridge between the populations is claimed.
- The weight contrast is small (calibrated vs equal weights differ by at most ~0.03 per criterion), so the two rankings are
  very similar; `primary_analysis_point_estimates.csv` reports how many persons and insiders actually change.
- The Confidence criterion is a classifier output trained on the same insider labels it is evaluated on; part of any
  enrichment is circular. `catalog_sensitivity.csv` reports the `drop_confidence` variant.
- Scenario 3 has 10 insiders (resolution 1/10); Scenarios 1 and 2 have 30 each.
- Training capacity (Scenario A vs B) remains an unsourced assumption.
- Whether a hyperparameter search was ever run for the classifier is recorded by the author in
  `cert_confidence_classifier_v2.HYPERPARAMETER_SEARCH_RUN` (currently `False`; author to confirm).

## Data

- **CERT Insider Threat Test Dataset r4.2** (Lindauer, 2020, https://doi.org/10.1184/R1/12841247.v1): not included (about 87 GB
  uncompressed). Aggregated, de-identified user-week features are written to `results/` and can be committed so the downstream
  analysis runs without the archive (`--from-features`). `answers/insiders.csv` is part of the CERT release; redistribute it only
  if the dataset licence allows.
- **AHP survey (n = 105): the raw export is NOT in this repository** and must not be: it was collected before ethics approval,
  contains submission timestamps, and 20 respondents are the only person with their institution-type x region x role combination.
  The AHP scripts read it from a local path set in `scripts/config.py` and are skipped when it is absent.

## Citation

Lindauer, B. (2020). Insider Threat Test Dataset [Data set]. Carnegie Mellon University, Kilthub. https://doi.org/10.1184/R1/12841247.v1
