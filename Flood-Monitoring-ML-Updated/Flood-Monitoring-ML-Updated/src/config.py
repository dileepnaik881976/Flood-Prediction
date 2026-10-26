"""Central configuration: paths, column names and constants (Windows-safe via pathlib)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
FLOOD_CSV = DATA_DIR / "flood.csv"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
VALIDATION_PREDICTIONS_PATH = REPORTS_DIR / "validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"

FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
    "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors",
]
TARGET = "FloodProbability"
ID_COL = "id"

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.2
# Random Forest / Extra Trees are fit on a subsample to keep memory and time reasonable.
TREE_ENSEMBLE_TRAIN_ROWS = 100_000
