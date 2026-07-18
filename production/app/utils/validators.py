from typing import List, Dict, Any
import pandas as pd
import numpy as np

try:
    from ..config import FEATURE_ORDER, FEATURE_TOLERANCE
    from .logger import get_logger

except ImportError:
    from app.config import FEATURE_ORDER, FEATURE_TOLERANCE
    from app.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    pass


def validate_features(features: pd.DataFrame) -> bool:
    logger.debug(f"Validating features with shape {features.shape}")
    
    # check if DataFrame is empty
    if features.empty:
        raise ValidationError("Features DataFrame is empty")
    
    # check for single row
    if len(features) != 1:
        raise ValidationError(f"Expected 1 row, got {len(features)}")
    
    # check for required columns
    missing_cols = set(FEATURE_ORDER) - set(features.columns)
    
    if missing_cols:
        raise ValidationError(f"Missing required columns: {missing_cols}")
    
    # check feature order
    expected_order = FEATURE_ORDER
    actual_order = features.columns.tolist()
    
    if actual_order != expected_order:
        logger.warning(f"Feature order mismatch. Expected: {expected_order[:5]}..., Got: {actual_order[:5]}...")
        raise ValidationError(
            f"Feature column order mismatch. Expected order: {expected_order}, Got: {actual_order}"
        )
    
    # check for NaN values
    nan_count = features.isna().sum().sum()
    
    if nan_count > 0:
        nan_cols = features.columns[features.isna().any()].tolist()
        raise ValidationError(f"Found {nan_count} NaN values in columns: {nan_cols}")
    
    # check for infinite values
    inf_mask = np.isinf(features.select_dtypes(include=[np.number]))
    inf_count = inf_mask.sum().sum()
    
    if inf_count > 0:
        inf_cols = inf_mask.columns[inf_mask.any()].tolist()
        raise ValidationError(f"Found {inf_count} infinite values in columns: {inf_cols}")
    
    logger.debug("Feature validation passed")
    
    return True


def validate_market_data(data: Dict[str, float], assets: List[str]) -> bool:
    logger.debug(f"Validating market data for assets: {assets}")
    
    if not isinstance(data, dict):
        raise ValidationError(f"Market data must be dict, got {type(data)}")
    
    missing_assets = set(assets) - set(data.keys())
    
    if missing_assets:
        raise ValidationError(f"Missing assets in market data: {missing_assets}")
    
    for asset, value in data.items():
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Asset {asset} value must be numeric, got {type(value)}")
        
        if np.isnan(value) or np.isinf(value):
            raise ValidationError(f"Asset {asset} has invalid value: {value}")
    
    logger.debug("Market data validation passed")
    
    return True


def validate_prediction_output(prediction: Dict[str, Any]) -> bool:

    logger.debug("Validating prediction output")
    
    required_keys = {"predicted_return", "direction", "confidence", "timestamp"}
    missing_keys = required_keys - set(prediction.keys())
    
    if missing_keys:
        raise ValidationError(f"Missing required keys in prediction: {missing_keys}")
    
    # validate predicted_return is numeric
    if not isinstance(prediction["predicted_return"], (int, float)):
        raise ValidationError(f"predicted_return must be numeric, got {type(prediction['predicted_return'])}")
    
    # validate direction is UP or DOWN
    if prediction["direction"] not in ["UP", "DOWN"]:
        raise ValidationError(f"direction must be 'UP' or 'DOWN', got {prediction['direction']}")
    
    # validate confidence is between 0 and 1
    if not (0 <= prediction["confidence"] <= 1):
        raise ValidationError(f"confidence must be between 0 and 1, got {prediction['confidence']}")
    
    # validate timestamp is ISO format string
    if not isinstance(prediction["timestamp"], str):
        raise ValidationError(f"timestamp must be string, got {type(prediction['timestamp'])}")
    
    logger.debug("Prediction output validation passed")
    
    return True


def compare_features_to_baseline(calculated: pd.DataFrame, expected: pd.DataFrame, tolerance: float = FEATURE_TOLERANCE) -> bool:
    logger.debug(f"Comparing features with tolerance {tolerance}")
    
    if not validate_features(calculated):
        return False
    
    # select only numeric columns
    numeric_cols = calculated.select_dtypes(include=[np.number]).columns
    
    # compare each feature
    differences = {}
    
    for col in numeric_cols:
        if col in expected.columns:
            diff = abs(calculated[col].values[0] - expected[col].values[0])
            
            if diff > tolerance:
                differences[col] = {
                    "calculated": calculated[col].values[0],
                    "expected": expected[col].values[0],
                    "difference": diff
                }
    
    if differences:
        logger.error(f"Feature mismatches found: {differences}")
        raise ValidationError(f"Features don't match within tolerance: {differences}")
    
    logger.debug("All features match within tolerance")
    
    return True