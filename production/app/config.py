import json
import os
from pathlib import Path
from typing import Optional, Union

# base paths
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

# model paths
MODEL_DIR = ARTIFACTS_DIR / "Models"
MODEL_PATH = MODEL_DIR / "CatBoost_model.pkl"
FEATURE_MANIFEST_PATH = ARTIFACTS_DIR / "FeatureSelection" / "feature_menifest.json"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "ModelPerformance" / "FinalModel.csv"


def _resolve_path(path_value: Union[str, Path, None], default: Path) -> Path:
    if not path_value:
        return default.resolve()

    candidate = Path(path_value).expanduser()
    
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    
    return candidate


def load_environment(env_path: Optional[Union[str, Path]] = None) -> None:
    env_candidates = []
    
    if env_path is not None:
        env_candidates.append(Path(env_path))

    env_candidates.extend([BASE_DIR / ".env", PROJECT_ROOT / ".env", Path.cwd() / ".env"])

    for candidate in env_candidates:
        if candidate.exists() and candidate.is_file():
            with open(candidate, encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    
                    if not line or line.startswith("#") or "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    
                    if key:
                        os.environ[key] = value
            break

    _refresh_config()


def _refresh_config() -> None:
    global API_HOST, API_PORT, API_DEBUG, LOG_LEVEL, LOG_FORMAT, LOG_DIR, LOG_FILE
    global MODEL_PATH, FEATURE_MANIFEST_PATH, DATA_DIR, ARTIFACTS_DIR

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_DEBUG = os.getenv("API_DEBUG", "false").lower() in {"1", "true", "yes", "on"}

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DIR = (BASE_DIR / "logs").resolve()
    LOG_FILE = LOG_DIR / "application.log"
    LOG_DIR.mkdir(exist_ok=True)

    MODEL_PATH = _resolve_path(os.getenv("MODEL_PATH"), MODEL_DIR / "CatBoost_model.pkl")
    FEATURE_MANIFEST_PATH = _resolve_path(
        os.getenv("FEATURE_MANIFEST_PATH"),
        ARTIFACTS_DIR / "FeatureSelection" / "feature_menifest.json",
    )


# API configuration
load_environment()

# feature configuration
LAG_RANGE = [1, 2, 3, 4, 5]
ROLLING_WINDOW = 5          # days for rolling mean/std calculation
MOMENTUM_WINDOW = 5         # days for momentum calculation

# narket data configuration
ASSETS = ["GLD", "SPX", "USO", "SLV", "EURUSD"]
BASE_FEATURES = ["SPX_Return", "USO_Return", "SLV_Return", "EURUSD_Return"]
TARGET_COLUMN = "GLD_Return"

# feature engineering configuration
FEATURE_ORDER = [
    "SPX_Return", "USO_Return", "SLV_Return", "EURUSD_Return",
    "SPX_Return_lag1", "SPX_Return_lag2", "SPX_Return_lag3", "SPX_Return_lag4", "SPX_Return_lag5",
    "USO_Return_lag1", "USO_Return_lag2", "USO_Return_lag3", "USO_Return_lag4", "USO_Return_lag5",
    "SLV_Return_lag1", "SLV_Return_lag2", "SLV_Return_lag3", "SLV_Return_lag4", "SLV_Return_lag5",
    "EURUSD_Return_lag1", "EURUSD_Return_lag2", "EURUSD_Return_lag3", "EURUSD_Return_lag4", "EURUSD_Return_lag5",
    "GLD_Return_lag1", "GLD_Return_lag2", "GLD_Return_lag3", "GLD_Return_lag4", "GLD_Return_lag5",
    "rolling_mean", "rolling_std", "momentum"
]

# validation thresholds
FEATURE_TOLERANCE = 0.0001
PREDICTION_TOLERANCE = 0.001
MIN_RESPONSE_TIME_MS = 100
MAX_RESPONSE_TIME_MS = 500


# load feature manifest
def load_feature_manifest() -> dict:
    if FEATURE_MANIFEST_PATH.exists():
        with open(FEATURE_MANIFEST_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    
    return {
        "target_column": TARGET_COLUMN,
        "base_features": BASE_FEATURES,
        "engineered_features": FEATURE_ORDER,
        "lag_range": LAG_RANGE,
        "target_shift": -1,
    }


def load_final_model_info() -> dict:
    if FINAL_MODEL_PATH.exists():
        import pandas as pd
        df = pd.read_csv(FINAL_MODEL_PATH)
        
        if not df.empty:
            return df.set_index("Metrics")["Value"].to_dict()
    
    return {}


def get_best_model_path() -> Path:
    info = load_final_model_info()
    model_name = info.get("Model", "CatBoost (Optuna)")
    
    # map model name to file
    model_map = {
        "CatBoost (Optuna)": MODEL_DIR / "CatBoost_model.pkl",
        "CatBoost (Baseline)": MODEL_DIR / "CatBoost_model.pkl",
        "CatBoost (RandomSearchCV)": MODEL_DIR / "CatBoost_model.pkl",
        "LightGBM (Optuna)": MODEL_DIR / "LightGBM_model.pkl",
        "LightGBM (Baseline)": MODEL_DIR / "LightGBM_model.pkl",
        "LightGBM (RandomSearchCV)": MODEL_DIR / "LightGBM_model.pkl",
        "XGBoost (Optuna)": MODEL_DIR / "XGBoost_model.pkl",
        "XGBoost (Baseline)": MODEL_DIR / "XGBoost_model.pkl",
        "XGBoost (RandomSearchCV)": MODEL_DIR / "XGBoost_model.pkl",
    }
    
    return model_map.get(model_name, MODEL_PATH)


FEATURE_MANIFEST = load_feature_manifest()
FINAL_MODEL_INFO = load_final_model_info()
