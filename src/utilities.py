import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")

import optuna
import catboost as cb


# unified metric calculation, train/test/CV comparison, and overfitting diagnosis
class Evaluator:
    
    @staticmethod
    def safe_mape(y_true, y_pred, epsilon=1e-8):
        """Robust MAPE: avoids division by zero."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        
        return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Compute [MSE, MAE, RMSE, R2, MAPE] for regression."""
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Empty arrays passed to calculate_metrics.")
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = Evaluator.safe_mape(y_true, y_pred)
        
        return [mse, mae, rmse, r2, mape]
    

    @staticmethod
    def directional_accuracy(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # if positive return, 0 if negative
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        
        # accuracy
        accuracy = np.mean(true_direction == pred_direction)
        
        # return as percentage
        return accuracy * 100
    

    @staticmethod
    def financial_metrics(model_name, y_true, y_pred, risk_free_rate=0.02/252):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # feature engineering already aligns target (T+1) with features (T). Shifting again would compare prediction (T+1) vs actual (T+2).
        positions = np.sign(y_pred)
        y_true = y_true[:len(positions)]        # enforce alignment explicitly
        
        # apply transaction costs based on position changes to simulate realistic trading returns
        transaction_cost = 0.0005

        position_change = np.abs(np.diff(positions, prepend=0))
        costs = position_change * transaction_cost

        strategy_returns = positions * y_true - costs
        
        # remove NaN
        strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
        
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return pd.DataFrame({
                'Model': [model_name],
                'Sharpe Ratio': [0.0],
                'Sortino Ratio': [0.0],
                'Max Drawdown': [0.0],
                'Total Return': [0.0]
            })
        
        # sharpe ratio
        excess_returns = strategy_returns - risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(strategy_returns) * np.sqrt(252)
        
        # sortino ratio (downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        
        else:
            sortino = 0.0
        
        # maximum Drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdown)
        
        # total Return
        total_return = (cumulative_returns[-1] - 1) * 100
        
        return pd.DataFrame({
            'Model': [model_name],
            'Sharpe Ratio': [round(sharpe, 4)],
            'Sortino Ratio': [round(sortino, 4)],
            'Max Drawdown': [round(max_drawdown * 100, 4)],
            'Total Return (%)': [round(total_return, 4)]
        })
    

    @staticmethod
    def performance_table(train_metrics, test_metrics):
        """Return DataFrame: Metrics | Training | Test """
        perf_df = pd.DataFrame({
            'Metrics': ['MSE', 'MAE', 'RMSE', 'R2 Score', 'MAPE', 'Directional Accuracy (%)'],
            'Training': train_metrics,
            'Test': test_metrics
        }).round(4)

        return perf_df
    

    @staticmethod
    def summary_builder(model_names, cv_df, test_metrics, test_dir_acc=None):
        test_df = pd.DataFrame({
            "Model": model_names,
            "Test MSE": [m[0] for m in test_metrics],
            "Test MAE": [m[1] for m in test_metrics],
            "Test RMSE": [m[2] for m in test_metrics],
            "Test R2": [m[3] for m in test_metrics],
            "Test MAPE": [m[4] for m in test_metrics],
            "Test Directional Accuracy (%)": test_dir_acc if test_dir_acc else [0.0] * len(model_names)
        })

        merged = pd.merge(cv_df, test_df, on="Model", how="inner")

        # select columns in logical order
        return merged[[
            "Model",
            "CV MSE", "CV MAE", "CV RMSE", "CV R2", "CV MAPE", "CV Directional Accuracy (%)",
            "Test MSE", "Test MAE", "Test RMSE", "Test R2", "Test MAPE", "Test Directional Accuracy (%)"
        ]].round(4)


    @staticmethod
    def cv_evaluate(model, x, y, cv, scoring=None):
        """Run cross-validation and return dict of average CV metrics including Directional Accuracy."""
        if scoring is None:
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

        # standard metrics via cross_validate
        cv_results = cross_validate(
            model, x, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
        )

        cv_mse = -cv_results['test_neg_mean_squared_error'].mean()
        cv_mae = -cv_results['test_neg_mean_absolute_error'].mean()
        cv_rmse = np.sqrt(cv_mse)
        cv_r2 = cv_results['test_r2'].mean()

        # MAPE and directional accuracy
        mape_scores = []
        dir_acc_scores = []

        for train_idx, val_idx in cv.split(x, y):
            model_clone = clone(model)
            try:
                model_clone.fit(x.iloc[train_idx], y.iloc[train_idx])
                y_pred = model_clone.predict(x.iloc[val_idx])

            except Exception as e:
                raise RuntimeError(f"Model failed during CV: {e}")

            mape_scores.append(Evaluator.safe_mape(y.iloc[val_idx], y_pred))
            dir_acc_scores.append(Evaluator.directional_accuracy(y.iloc[val_idx], y_pred))

        cv_mape = np.mean(mape_scores)
        cv_dir_acc = np.mean(dir_acc_scores)
        
        return {
            'CV MSE': cv_mse,
            'CV MAE': cv_mae,
            'CV RMSE': cv_rmse,
            'CV R2': cv_r2,
            'CV MAPE': cv_mape,
            'CV Directional Accuracy (%)': cv_dir_acc
        }


    @staticmethod
    def assess_overfitting(cv_r2, test_r2, cv_rmse, test_rmse, tolerance=0.05):
        """Determine overfitting status and generalization quality."""
        r2_gap = cv_r2 - test_r2
        rmse_ratio = test_rmse / cv_rmse if cv_rmse > 0 else np.inf
        
        # overfitting logic
        if r2_gap > tolerance or rmse_ratio > 1.05:
            overfit_status = "High"
        
        elif abs(r2_gap) <= tolerance and 0.95 <= rmse_ratio <= 1.05:
            overfit_status = "Low"
        
        else:
            overfit_status = "Mild"
        
        # generalization status
        if test_r2 > 0.85:
            gen_status = "Excellent"
        
        elif test_r2 > 0.70:
            gen_status = "Good"
        
        elif test_r2 > 0.50:
            gen_status = "Fair"
        
        else:
            gen_status = "Poor"
        
        return r2_gap, rmse_ratio, overfit_status, gen_status


# Handles saving models, performance summaries, and aggregated performance results to organized directories
class ModelPersister:
    
    def __init__(self, model_name, model_root, performance_root):
        self.model_name = model_name
        self.MODEL_DIR = model_root
        self.PERFORMANCE_DIR = performance_root
        
        # Create directories
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    

    # save the trained model in appropriate format
    def save_model(self, model):

        if self.model_name.lower() == "lstm":
            model.save(self.MODEL_DIR / f"{self.model_name}_model.keras")
        
        else:
            joblib.dump(model, self.MODEL_DIR / f"{self.model_name}_model.pkl")
        
        print(f"Model saved: {self.model_name}_model.pkl")

    # save full train/test/CV metrics
    def save_performance(self, performance_df, tag=""):
        if tag:
            filename = f"{self.model_name}_{tag}.csv"
        
        else:
            filename = f"{self.model_name}_OverallPerformance.csv"
        
        path = self.PERFORMANCE_DIR / filename
        performance_df.to_csv(path, index=False)
        print(f"performance saved: {self.model_name}")
    

    # append model's summary metrics to the shared performance file
    def aggregated_performance(self, df, name):
        path = self.PERFORMANCE_DIR / f"{name}.csv"
        
        # append or create
        if path.exists():
            model_perf = pd.read_csv(path)                          # open previous loaded data
            df = pd.concat([model_perf, df], ignore_index=True)     # append new data
            df.to_csv(path, index=False)
        
        else:
            df.to_csv(path, index=False)
        
        print(f"Appended to aggregated performance!")
    

    # append model's overfitting metrics to the shared overfitting file
    def append_overfitting(self, df):
        path = self.PERFORMANCE_DIR / "AllModel_OverfittingAnalysis.csv"
        
        if path.exists():
            overfit_df = pd.read_csv(path)                          # open previous loaded data
            df = pd.concat([overfit_df, df], ignore_index=True)     # append new data
            df.to_csv(path, index=False)
        
        else:
            df.to_csv(path, index=False)
        
        print(f"Appended to overfitting analysis!")


# loading/splitting workflow
class DataHandler:
    
    @staticmethod
    def load_dataset(file_path, target_col):
        """Load CSV and return df, x, y"""
        df = pd.read_csv(file_path)
        x = df.drop(columns=[target_col], errors='ignore')
        y = df[target_col]
        
        return df, x, y
    

    @staticmethod
    def load_artifacts(artifacts_root, cv_method, test_size=None, gap=0):
        """Load x_train, x_test, y_train, y_test, [cv]."""
        ARTIFACTS_DIR = Path(artifacts_root)
        artifacts = {
            'x_train': joblib.load(ARTIFACTS_DIR / "x_train.pkl"),
            'x_test': joblib.load(ARTIFACTS_DIR / "x_test.pkl"),
            'y_train': joblib.load(ARTIFACTS_DIR / "y_train.pkl"),
            'y_test': joblib.load(ARTIFACTS_DIR / "y_test.pkl")
        }

        if cv_method == 'kfcv':
            try:
                artifacts['cv'] = joblib.load(ARTIFACTS_DIR / "cv.pkl")
            
            except FileNotFoundError:
                artifacts['cv'] = KFold(n_splits=5, shuffle=True, random_state=42)
        
        elif cv_method == "tscv":
            try:
                artifacts['cv'] = joblib.load(ARTIFACTS_DIR / "cv.pkl")
            
            except FileNotFoundError:
                artifacts['cv'] = TimeSeriesSplit(n_splits=5, test_size=test_size, gap=gap)


        return artifacts
    

    @staticmethod
    def prepare_for_catboost(x_train, x_test):
        """Convert data to CatBoost-safe format (int/str categoricals)."""
        x_train_cb = x_train.copy()
        x_test_cb = x_test.copy()
        cat_features = []
        
        for col in x_train.columns:
            col_train = x_train[col]
            col_test = x_test[col]
            
            # case 1: Object (strings) to categorical
            if col_train.dtype == 'object':
                cat_features.append(col)

                # ensure test set has same categories
                x_train_cb[col] = col_train.astype('category')
                x_test_cb[col] = col_test.astype('category')
            
            # case 2: integer + low cardinality to categorical
            elif np.issubdtype(col_train.dtype, np.integer) and col_train.nunique() < 50:
                cat_features.append(col)
            
            # case 3: float to check if it's actually integer-encoded
            elif np.issubdtype(col_train.dtype, np.floating):
                
                # check if all values are whole numbers
                if (np.all(np.isclose(col_train, col_train.astype(int))) and np.all(np.isclose(col_test, col_test.astype(int)))):
                    x_train_cb[col] = col_train.astype(int)
                    x_test_cb[col] = col_test.astype(int)
                    
                    if x_train_cb[col].nunique() < 50:
                        cat_features.append(col)
        
        cat_indices = [x_train_cb.columns.get_loc(c) for c in cat_features] if cat_features else None
        
        return x_train_cb, x_test_cb, cat_features, cat_indices


# Optuna Pruning Callback for CatBoost
class CatBoostPruningCallback:
    """Reports intermediate RMSE to Optuna for early trial pruning."""
    
    def __init__(self, trial):
        self.trial = trial
    
    def after_iteration(self, info):
        try:
            # info.eval_results contains validation metrics per iteration
            evals = info.eval_results
            
            if evals:
                # get the last validation metric value
                val_data = evals.get('validation', evals.get('learn', {}))
                if val_data:
                    metric_name = list(val_data.keys())[0]
                    current_score = val_data[metric_name][-1]
                    
                    self.trial.report(current_score, info.iteration)
                    
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
        
        except optuna.TrialPruned:
            raise  # re-raise TrialPruned — Optuna needs to catch this
        
        except Exception:
            pass  # silently skip if eval data isn't available yet
        
        return True