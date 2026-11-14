"""
config.py -- the ONE place to edit paths for the whole pipeline.

Edit the three paths marked  <-- EDIT  once, then run:   python run_all.py

Layout (repo checkout):
    <repo>/scripts/            this folder (config.py, run_all.py, all analysis scripts)
    <repo>/results/            every output CSV/PNG/log is written here and committed
    <repo>/work/               extracted CERT log files (large; git-ignored)
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- <-- EDIT: location of the CERT r4.2 archive and its ground-truth answer key -------------------
CERT_ARCHIVE = Path(r"A:\PhD IT\r4.2.tar.bz2")
INSIDERS_CSV = Path(r"A:\PhD IT\results\answers\answers\insiders.csv")      # CERT 'answers/insiders.csv'

# --- <-- EDIT: the raw AHP survey export. LOCAL ONLY -- it must NOT be committed (ethics: collected before
#     approval, includes timestamps and quasi-identifiers). The three AHP scripts are skipped if it is absent.
SURVEY_XLSX = Path(r"A:\PhD IT\Cybersecurity_Threat_Prioritisation_-_AHP_Survey.xlsx")

# --- derived locations (normally leave as is) -------------------------------------------------------
RESULTS_DIR = REPO_ROOT / "results"
WORK_DIR = REPO_ROOT / "work"
FEATURES_CSV = RESULTS_DIR / "cert_threat_resource_features.csv"
CONFIDENCE_CSV = RESULTS_DIR / "cert_threat_resource_features_with_confidence.csv"

BUDGET = 7000.0     # dollars per year for the five modelled controls; see cert_budget_sweep.py for why this number
                    # has no independent justification and how conclusions vary with it
