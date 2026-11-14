# Analysis log

Every change to the analysis configuration, the defect it fixed, how the defect was found, and which results it
invalidated. Dates are given only where they are known from the supervisor reports; **the author must complete the
others from the git history** (`git log --date=iso`). The automatic record of what was actually run, with timestamps,
is `results/RUN_MANIFEST.md` (one per run). Total configurations run in the final pipeline: see that manifest;
the number of earlier, superseded runs was not logged and cannot be reconstructed, which is itself a limitation.

Known anchors: survey collected 1-9 Sep 2026 (before ethics approval); first repository commit 21 Sep 2026; pipeline
pushed in commit fd76c98, 28 Sep 2026 23:58; CHRPE application submitted (date: author to fill).

| # | Defect | How found | Fix | What it invalidated |
|---|---|---|---|---|
| 1 | Degenerate catalogue: every resource cost 1 abstract unit, so the allocation was a trivial sort | supervisor audit round 1 | Real-sourced five-resource catalogue; two Training-capacity scenarios | all allocation results |
| 2 | Allocation unit was the user-week (67,238 rows) but costs are per person per year | audit round 1 | Person-level unit (1,000 persons) | all allocation results |
| 3 | 5 SA seeds | audit round 1 | 10 seeds | SA summaries |
| 4 | Bootstrap resampled rows, ignoring that malicious rows cluster in 70 people | audit round 1 | Cluster bootstrap by person | all intervals |
| 5 | Ungrouped StratifiedKFold let a person's other weeks sit in training | audit round 1 | StratifiedGroupKFold by user | Confidence values, SHAP, confusion matrix |
| 6 | SA cooling 0.95 from T0=1000 reached the minimum after ~225 iterations, so the documented 10,000 never ran | supervisor round 2 | Schedule reaching its minimum at exactly max_iter (v6) | SA timings |
| 7 | Incident Response effectiveness 0.5991 = USD 2.66M (IBM **2022**) / USD 4.44M (2025) | supervisor round 2 (N1) | first replaced by USD 1.5M (0.3378) -- see #15 | all allocation results |
| 8 | Person-level features were SUMMED over weeks: a sum of a weekly proportion (Evasion) is meaningless, and every summed score rose with tenure, so early leavers (Scenario 1) scored lower mechanically | supervisor round 2 | Per-week MEANS, re-normalised | all person-level results |
| 9 | SA initial temperature 1000 is ~5 orders of magnitude above any single-move change in the objective; SA random-walked away from greedy and never recovered, giving "zero gain, zero variance" at any iteration budget | analysis after #6; confirmed on synthetic instances with a known gap | Temperature set from the measured |delta| distribution; 6-schedule sweep incl. legacy | every earlier "SA null result" |
| 10 | Boundary pool: the "marginally admitted" half was always EMPTY when >= 500 threats were unassigned (here 705), so 80% of proposals could only try to add assignments under an exhausted budget | found while porting SA into `pipeline_common`; regression-tested in `tests_pipeline.py` | Unassigned threats sort last, so admitted threats are included | every earlier SA result |
| 11 | After-hours window: `hour > 20` treated 20:00-20:59 as working hours (window was 06:00-20:59, documented 06:00-20:00) | supervisor round 2 | `hour >= 20`; verified on boundary timestamps | Evasion for every user-week, hence everything downstream |
| 12 | Timing script labelled all weeks of all insiders positive (about 3,100 vs 310-322 insider-window weeks) | supervisor round 2 | Timing uses the classifier's own labels and code path; regression test | classifier timing |
| 13 | Aggregation script blocked on `input()` | supervisor round 2 | Automatic archive check | reproducibility |
| 14 | Table 2 showed DLP cost 63 while code used 63.2; the mean of the four vendors is 63.25 | supervisor round 2 + recomputation | 63.25 everywhere | Scenario B DLP count unchanged (100) |
| 15 | IR effectiveness mixed measurement bases: the 2025 report has no IR entry, and USD 1.5M is the older "IR team + tested plan vs neither" contrast; the other four values are "factor in isolation" | primary IBM 2025 and 2024 report text, 5 Oct 2026 | All five on the in-isolation basis (IR = USD 248,000, IBM 2024); the contrast value kept as a sensitivity case | all allocation results; IR no longer ~22x Training's value/cost ratio |
| 16 | Training/SIEM/IAM savings were rounded secondary figures (190k/210k/190k); DLP came from a vendor-commissioned study | IBM 2025 Fig. 39 | Exact Fig. 39 values (192,266 / 212,061 / 189,838); DLP from "Data security and protection software" (157,456); Forrester kept as a sensitivity case | effectiveness inputs |
| 17 | Calibrated weights carried at 3 d.p. summed to 1.001 | code review | 4 d.p. (sum exactly 1) | negligible; ranking-level |
| 18 | SHAP global mean signed value: explained as "swamped by the majority class", then as "near zero by construction". Both partly wrong: the sum of the means equals (mean prediction - baseline); with class_weight='balanced' the baseline is ~0.5, so every global mean is negative | checked numerically, 5 Oct 2026 | v4 prints the identity | wording only |
| 19 | SHAP script returned silently if `shap` was missing, so a runner recorded success | pipeline smoke test | SystemExit | reproducibility |
| 20 | Exact-optimum v1 printed "Optimal value" for solutions HiGHS had not proved optimal; v2 carried its own copy of the old catalogue | review of first real output | v3: proven vs best-found, shared catalogue | exact-optimum CSV from v1/v2 |
| 21 | Primary outcome (coverage rate over 70 insiders) had resolution 1/70 = 0.0143, larger than the effects reported | supervisor round 2 | exact integer counts, gained/lost, continuous outcomes (AUC, effectiveness-weighted coverage) | interpretation of every "effect" |
| 22 | Raw AHP survey export committed publicly (pre-approval data, timestamps, 20 uniquely identifiable respondents) | supervisor round 2 | Removed from the repository; README corrected (author action; history rewrite is the supervisor's decision) | ethics |

## Not yet resolved (see README, "Known limitations")

Training capacity (unsourced); B = 7,000 (no independent justification; see the budget sweep); person-level application of an
organisation-level saving; survey sampling frame and response rate (invitation records needed from the author); whether the
Ghanaian HEI respondents were shown the operational definitions of the four criteria (author to confirm); CHRPE reference number.
