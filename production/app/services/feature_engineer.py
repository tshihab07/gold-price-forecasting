from typing import Dict, Deque, Optional
from collections import deque
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from ..config import (
        LAG_RANGE, ROLLING_WINDOW, MOMENTUM_WINDOW, BASE_FEATURES, TARGET_COLUMN,
        FEATURE_ORDER
    )
    from ..utils.logger import get_logger
    from ..utils.validators import validate_features, ValidationError

except ImportError:
    from app.config import LAG_RANGE, ROLLING_WINDOW, MOMENTUM_WINDOW, BASE_FEATURES, TARGET_COLUMN, FEATURE_ORDER
    from app.utils.logger import get_logger
    from app.utils.validators import validate_features, ValidationError

logger = get_logger(__name__)


class GoldFeatureEngineer:
    
    def __init__(self, max_history: int = 10):
        self.max_history = max(max_history, max(LAG_RANGE) + ROLLING_WINDOW + MOMENTUM_WINDOW)
        
        # rolling windows for each base feature
        self.spx_returns: Deque[float] = deque(maxlen=self.max_history)
        self.uso_returns: Deque[float] = deque(maxlen=self.max_history)
        self.slv_returns: Deque[float] = deque(maxlen=self.max_history)
        self.eurusd_returns: Deque[float] = deque(maxlen=self.max_history)
        self.gld_returns: Deque[float] = deque(maxlen=self.max_history)
        
        # track timestamps for logging
        self.timestamps: Deque[datetime] = deque(maxlen=self.max_history)
        
        logger.info(f"GoldFeatureEngineer initialized with max_history={self.max_history}")
    

    def update(self, new_data: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        # validate input has all required columns
        required_keys = set(BASE_FEATURES + [TARGET_COLUMN])
        provided_keys = set(new_data.keys())
        missing = required_keys - provided_keys
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        try:
            # add to rolling windows in order
            self.spx_returns.append(float(new_data["SPX_Return"]))
            self.uso_returns.append(float(new_data["USO_Return"]))
            self.slv_returns.append(float(new_data["SLV_Return"]))
            self.eurusd_returns.append(float(new_data["EURUSD_Return"]))
            self.gld_returns.append(float(new_data[TARGET_COLUMN]))
            
            if timestamp:
                self.timestamps.append(timestamp)
            
            else:
                self.timestamps.append(datetime.now())
            
            logger.debug(
                f"Updated feature engineer. History size: {len(self.spx_returns)}. "
                f"Latest GLD_Return: {new_data[TARGET_COLUMN]:.6f}"
            )
        
        except (TypeError, ValueError) as e:
            logger.error(f"Error updating feature engineer: {e}")
            raise ValueError(f"Invalid data types in new_data: {e}")
    

    def is_ready(self) -> bool:
        min_required = max(LAG_RANGE) + ROLLING_WINDOW
        ready = len(self.spx_returns) >= min_required
        
        if not ready:
            logger.debug(
                f"Feature engineer not ready. Current history: {len(self.spx_returns)}, "
                f"Required: {min_required}"
            )
        
        return ready
    

    def _get_lag_features(self, returns: Deque[float], prefix: str) -> Dict[str, float]:
        features = {}
        
        # convert deque to list for indexing
        returns_list = list(returns)
        
        for lag in LAG_RANGE:
            if lag <= len(returns_list) - 1:
                # get value from lag days ago
                lag_value = returns_list[-(lag + 1)]
                features[f"{prefix}_lag{lag}"] = float(lag_value)
            
            else:
                features[f"{prefix}_lag{lag}"] = np.nan
        
        return features
    

    def _get_rolling_statistics(self) -> Dict[str, float]:
        gld_list = list(self.gld_returns)
        
        if len(gld_list) < ROLLING_WINDOW:
            return {
                "rolling_mean": np.nan,
                "rolling_std": np.nan
            }
        
        # calculate over the most recent ROLLING_WINDOW values
        recent_returns = gld_list[-ROLLING_WINDOW:]
        
        return {
            "rolling_mean": float(np.mean(recent_returns)),
            "rolling_std": float(np.std(recent_returns))
        }
    

    def _get_momentum(self) -> float:
        gld_list = list(self.gld_returns)
        
        if len(gld_list) < MOMENTUM_WINDOW:
            return np.nan
        
        # get most recent MOMENTUM_WINDOW returns
        recent_returns = gld_list[-MOMENTUM_WINDOW:]
        
        # calculate cumulative return over the window
        # momentum ≈ product((1 + r_i) for r_i in returns) - 1
        momentum = np.prod([1 + r for r in recent_returns]) - 1
        
        return float(momentum)
    
    def extract_features(self) -> pd.DataFrame:
        if not self.is_ready():
            raise ValueError("Not enough historical data to extract features")
        
        logger.debug("Extracting features")
        
        # start with current base features
        features = {
            "SPX_Return": float(list(self.spx_returns)[-1]),
            "USO_Return": float(list(self.uso_returns)[-1]),
            "SLV_Return": float(list(self.slv_returns)[-1]),
            "EURUSD_Return": float(list(self.eurusd_returns)[-1])
        }
        
        # add lag features
        features.update(self._get_lag_features(self.spx_returns, "SPX_Return"))
        features.update(self._get_lag_features(self.uso_returns, "USO_Return"))
        features.update(self._get_lag_features(self.slv_returns, "SLV_Return"))
        features.update(self._get_lag_features(self.eurusd_returns, "EURUSD_Return"))
        features.update(self._get_lag_features(self.gld_returns, "GLD_Return"))
        
        # add rolling statistics
        features.update(self._get_rolling_statistics())
        
        # add momentum
        features["momentum"] = self._get_momentum()
        
        # create DataFrame with single row, in exact feature order
        df = pd.DataFrame([features])
        df = df[FEATURE_ORDER]  # Reorder columns to match training
        
        # Validate features
        try:
            validate_features(df)
        
        except ValidationError as e:
            logger.error(f"Feature extraction validation failed: {e}")
            raise
        
        logger.debug(f"Successfully extracted features. Shape: {df.shape}")
        
        return df
    

    def get_history(self) -> Dict[str, list]:
        return {
            "SPX_Return": list(self.spx_returns),
            "USO_Return": list(self.uso_returns),
            "SLV_Return": list(self.slv_returns),
            "EURUSD_Return": list(self.eurusd_returns),
            "GLD_Return": list(self.gld_returns),
            "timestamps": [ts.isoformat() for ts in self.timestamps]
        }
    
    def clear_history(self) -> None:
        self.spx_returns.clear()
        self.uso_returns.clear()
        self.slv_returns.clear()
        self.eurusd_returns.clear()
        self.gld_returns.clear()
        self.timestamps.clear()
        
        logger.info("Cleared feature engineer history")