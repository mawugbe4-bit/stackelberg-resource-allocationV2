# Run manifest

Mode: **full**  
UTC: 2026-10-06T17:29:35.212958+00:00  
Git commit: `not a git checkout`  
Python 3.14.6 on Windows-11-10.0.26200-SP0  
CPU: Intel(R) Core(TM) i5-8600 CPU @ 3.10GHz (6 logical cores)  
Libraries: {'python': '3.14.6', 'numpy': '2.4.6', 'pandas': '3.0.3', 'scipy': '1.18.0', 'sklearn': '1.9.0', 'shap': '0.52.0', 'matplotlib': '3.11.0', 'openpyxl': '3.1.5'}

| step | module | status | seconds |
|---|---|---|---|
| aggregate | cert_local_aggregation_v3 | ok | 3865.0 |
| classifier | cert_confidence_classifier_v2 | ok | 8.6 |
| shap | cert_shap_confidence_v4 | ok | 69.7 |
| confusion | cert_confusion_matrix_v3 | ok | 0.5 |
| primary | cert_primary_analysis_v4 | ok | 17.1 |
| sa | cert_sa_greedy_pipeline_v8 | ok | 287.6 |
| exact | cert_exact_optimum_v3 | ok | 1201.0 |
| budget | cert_budget_sweep | ok | 68.8 |
| catalog | cert_catalog_sensitivity | ok | 99.9 |
| features | cert_feature_diagnostics | ok | 0.4 |
| normalisation | cert_normalisation_sensitivity | ok | 34.7 |
| ablation | cert_sa_ablation | ok | 272.4 |
| bruteforce | verify_formulation_bruteforce | ok | 25.9 |
| timing | cert_timing_v5 | ok | 19.7 |
| ahp_weights | ahp_weights_by_institution | skipped (survey file absent) | 0 |
| ahp_cr | ahp_cr_sensitivity | skipped (survey file absent) | 0 |
| ahp_split | ahp_split_seed_sensitivity | skipped (survey file absent) | 0 |

14 of 17 steps completed. Each step's console output: `results/logs/<step>.log`.
