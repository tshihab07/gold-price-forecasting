from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

try:
    from ..config import MODEL_PATH, FEATURE_ORDER, get_best_model_path
    from ..utils.logger import get_logger
    from ..utils.validators import validate_features, validate_prediction_output, ValidationError

except ImportError:
    from app.config import MODEL_PATH, FEATURE_ORDER, get_best_model_path
    from app.utils.logger import get_logger
    from app.utils.validators import validate_features, validate_prediction_output, ValidationError

logger = get_logger(__name__)


class GoldPricePredictor:
    def __init__(self, model_path: Optional[Path] = None):

        if model_path is None:
            model_path = get_best_model_path()
        
        self.model_path = Path(model_path)
        self.model = None
        self.feature_order = FEATURE_ORDER
        self.prediction_count = 0
        self.error_count = 0
        
        self._load_model()
    

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Successfully loaded model from {self.model_path}")
            
            # Validate model has predict method
            if not hasattr(self.model, 'predict'):
                raise RuntimeError("Model doesn't have predict method")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Error loading model: {e}")
    

    def predict(self, features: pd.DataFrame, confidence_method: str = "std") -> Dict[str, Any]:
        logger.debug("Making prediction")
        
        try:
            # Validate features
            validate_features(features)
            
            # ensure features are in correct order
            features_ordered = features[self.feature_order]
            
            # make prediction
            predicted_return = float(self.model.predict(features_ordered)[0])
            
            # determine direction
            direction = "UP" if predicted_return >= 0 else "DOWN"
            
            # calculate confidence based on absolute return magnitude
            # normalize to 0-1 range using typical return magnitudes
            confidence = min(1.0, max(0.0, abs(predicted_return) / 0.05))
            
            # get current timestamp
            timestamp = datetime.utcnow().isoformat()
            
            # construct result dictionary
            result = {
                "predicted_return": predicted_return,
                "direction": direction,
                "confidence": confidence,
                "timestamp": timestamp,
                "model_version": self._get_model_version(),
                "feature_count": len(self.feature_order)
            }
            
            # validate output
            validate_prediction_output(result)
            
            self.prediction_count += 1
            
            logger.info(
                f"Prediction {self.prediction_count}: "
                f"return={predicted_return:.6f}, "
                f"direction={direction}, "
                f"confidence={confidence:.4f}"
            )
            
            return result
            
        except ValidationError as e:
            self.error_count += 1
            logger.error(f"Prediction validation error: {e}")
            raise
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    

    def batch_predict(self, features_list: list) -> list:
        logger.debug(f"Making batch prediction for {len(features_list)} samples")
        
        results = []
        for i, features in enumerate(features_list):
            try:
                result = self.predict(features)
                results.append(result)
            
            except Exception as e:
                logger.error(f"Error on batch sample {i}: {e}")
                results.append({"error": str(e)})
        
        return results


    def _get_model_version(self) -> str:
        try:
            from ..config import FINAL_MODEL_INFO
        
        except ImportError:
            from app.config import FINAL_MODEL_INFO
        
        return FINAL_MODEL_INFO.get("Model", "catboost_optuna_v1")
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get predictor statistics."""
        return {
            "model_path": str(self.model_path),
            "model_loaded": self.model is not None,
            "prediction_count": self.prediction_count,
            "error_count": self.error_count,
            "success_rate": (
                self.prediction_count / (self.prediction_count + self.error_count) * 100
                if (self.prediction_count + self.error_count) > 0 else 0
            ),
            "feature_count": len(self.feature_order),
            "features": self.feature_order
        }


class EnsemblePredictor:
    def __init__(self):
        self.predictors = {}
        
        # Load primary model
        try:
            self.predictors["catboost"] = GoldPricePredictor()
        except Exception as e:
            logger.error(f"Failed to load CatBoost model: {e}")
        
        if not self.predictors:
            raise RuntimeError("Failed to load any models for ensemble")
        
        logger.info(f"Ensemble initialized with {len(self.predictors)} models")
    

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        logger.debug("Making ensemble prediction")
        
        predictions = {}
        directions = []
        confidences = []
        
        for name, predictor in self.predictors.items():
            try:
                pred = predictor.predict(features)
                predictions[name] = pred
                directions.append(1 if pred["direction"] == "UP" else -1)
                confidences.append(pred["confidence"])
            
            except Exception as e:
                logger.warning(f"Predictor {name} failed: {e}")
        
        if not predictions:
            raise RuntimeError("All predictors failed")
        
        # ensemble decision
        avg_direction = np.mean(directions)
        ensemble_direction = "UP" if avg_direction > 0 else "DOWN"
        
        # average confidence
        ensemble_confidence = np.mean(confidences)
        
        # average predicted return
        avg_return = np.mean([
            p.get("predicted_return", 0) for p in predictions.values() 
            if "predicted_return" in p
        ])
        
        result = {
            "predicted_return": avg_return,
            "direction": ensemble_direction,
            "confidence": ensemble_confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": "ensemble_v1",
            "member_predictions": predictions,
            "feature_count": len(FEATURE_ORDER)
        }
        
        logger.info(
            f"Ensemble prediction: "
            f"return={avg_return:.6f}, "
            f"direction={ensemble_direction}, "
            f"confidence={ensemble_confidence:.4f}"
        )
        
        return result