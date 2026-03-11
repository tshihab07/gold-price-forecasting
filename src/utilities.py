import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore")


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
    def performance_table(train_metrics, test_metrics):
        """Return DataFrame: Metrics | Training | Test """
        perf_df = pd.DataFrame({
            'Metrics': ['MSE', 'MAE', 'RMSE', 'R2 Score', 'MAPE'],
            'Training': train_metrics,
            'Test': test_metrics
        }).round(4)

        return perf_df
    

    @staticmethod
    def summary_builder(model_names, cv_df, test_metrics):
        """ Overall Model Performance (CV + Test) — Merged """
        test_df = pd.DataFrame({
            "Model": model_names,
            "Test MSE": [m[0] for m in test_metrics],
            "Test MAE": [m[1] for m in test_metrics],
            "Test RMSE": [m[2] for m in test_metrics],
            "Test R2": [m[3] for m in test_metrics],
            "Test MAPE": [m[4] for m in test_metrics]
        })

        merged = pd.merge(cv_df, test_df, on="Model", how="inner")
        return merged[[
            "Model",
            "CV MSE", "CV MAE", "CV RMSE", "CV R2", "CV MAPE",
            "Test MSE", "Test MAE", "Test RMSE", "Test R2", "Test MAPE"
        ]].round(4)


    @staticmethod
    def cv_evaluate(model, X, y, cv, scoring=None):
        """Run cross-validation and return dict of average CV metrics (MSE, MAE, RMSE, R2, MAPE)."""
        if scoring is None:
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        # standard metrics via cross_validate
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
        )
        
        cv_mse = -cv_results['test_neg_mean_squared_error'].mean()
        cv_mae = -cv_results['test_neg_mean_absolute_error'].mean()
        cv_rmse = np.sqrt(cv_mse)
        cv_r2 = cv_results['test_r2'].mean()
        
        # MAPE: custom loop
        mape_scores = []
        for train_idx, val_idx in cv.split(X, y):
            model_clone = model
            try:
                model_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
                y_pred = model_clone.predict(X.iloc[val_idx])
            
            except Exception:
                # fallback: use original model fit if stateful
                y_pred = model.predict(X.iloc[val_idx])
           
            mape_scores.append(Evaluator.safe_mape(y.iloc[val_idx], y_pred))
        
        cv_mape = np.mean(mape_scores)
        
        return {
            'CV MSE': cv_mse,
            'CV MAE': cv_mae,
            'CV RMSE': cv_rmse,
            'CV R2': cv_r2,
            'CV MAPE': cv_mape
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
    
    def __init__(self, model_name, artifacts_root="../artifacts"):
        self.model_name = model_name
        self.artifacts_root = Path(artifacts_root)
        self.model_dir = self.artifacts_root / "models"
        self.performance_dir = self.artifacts_root / "model-performance"
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.performance_dir.mkdir(parents=True, exist_ok=True)
    

    # save the trained model in appropriate format
    def save_model(self, model):
        joblib.dump(model, self.model_dir / f"model_{self.model_name.title()}.pkl")
        
        print(f"Model saved: {self.model_dir}/{self.model_name.lower()}.pkl")

    # save full train/test/CV metrics
    def save_performance(self, performance_df, tag=""):
        if tag:
            filename = f"{self.model_name.lower()}{tag}.csv"
        else:
            filename = f"{self.model_name.lower()}Performance.csv"
        
        path = self.performance_dir / filename
        performance_df.to_csv(path, index=False)
        print(f"{self.model_name} performance saved: {path}")
    

    # append model's summary metrics to the shared performance file
    def aggregated_performance(self, df):
        path = self.performance_dir / "a_ModelPerformance.csv"
        
        # append or create
        if path.exists():
            model_perf = pd.read_csv(path)                          # open previous loaded data
            df = pd.concat([model_perf, df], ignore_index=True)     # append new data
            df.to_csv(path, index=False)
        
        else:
            df.to_csv(path, index=False)
        
        print(f"Appended to aggregated performance: {path}")
    

    # append model's overfitting metrics to the shared overfitting file
    def append_overfitting(self, df):
        path = self.performance_dir / "a_overfittingAnalysis.csv"
        
        if path.exists():
            overfit_df = pd.read_csv(path)                          # open previous loaded data
            df = pd.concat([overfit_df, df], ignore_index=True)     # append new data
            df.to_csv(path, index=False)
        
        else:
            df.to_csv(path, index=False)
        
        print(f"Appended to overfitting analysis: {path}")


# loading/splitting workflow
class DataHandler:
    
    @staticmethod
    def load_dataset(file_path):
        """Load CSV and return df, X, y (with 'OutletSales' as target)."""
        df = pd.read_csv(file_path)
        X = df.drop(columns=['OutletSales'], errors='ignore')
        y = df['OutletSales']
        return df, X, y
    

    @staticmethod
    def load_artifacts(artifacts_dir, cv_required=True):
        """Load x_train, x_test, y_train, y_test, [cv]."""
        artifacts_dir = Path(artifacts_dir)
        artifacts = {
            'x_train': joblib.load(artifacts_dir / "x_train.pkl"),
            'x_test': joblib.load(artifacts_dir / "x_test.pkl"),
            'y_train': joblib.load(artifacts_dir / "y_train.pkl"),
            'y_test': joblib.load(artifacts_dir / "y_test.pkl")
        }

        if cv_required:
            try:
                artifacts['cv'] = joblib.load(artifacts_dir / "cv.pkl")
            
            except FileNotFoundError:
                print("⚠️ Warning: cv.pkl not found. Using default 5-Fold CV.")
                from sklearn.model_selection import KFold
                artifacts['cv'] = KFold(n_splits=5, shuffle=True, random_state=42)
        
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

# Time Series Split for forecasting models and Create TimeSeriesSplit cross-validator
class TimeSeriesUtils:
    
    @staticmethod
    def create_timeseries_cv(n_splits=5, test_size=None, gap=0):
        return TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    
    
    # Create sequences for LSTM/RNN models
    @staticmethod
    def create_sequences(X, y, sequence_length):
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        X_seq, y_seq = [], []
        
        for i in range(len(X_array) - sequence_length):
            X_seq.append(X_array[i:i + sequence_length])
            y_seq.append(y_array[i + sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    
    @staticmethod
    def directional_accuracy(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # if positive return, 0 if negative
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        
        # accuracy
        accuracy = np.mean(true_direction == pred_direction)
        
        return accuracy * 100  # return as percentage
    
    # Calculate financial performance metrics.
    @staticmethod
    def calculate_financial_metrics(y_true, y_pred, risk_free_rate=0.02/252):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # strategy returns
        positions = np.sign(y_pred)
        strategy_returns = positions[:-1] * y_true[1:]  # Shift to align
        
        # remove NaN
        strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
        
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return {
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Total Return': 0.0
            }
        
        # harpe Ratio
        excess_returns = strategy_returns - risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(strategy_returns) * np.sqrt(252)
        
        # Sortino Ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        else:
            sortino = 0.0
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdown)
        
        # Total Return
        total_return = (cumulative_returns[-1] - 1) * 100
        
        return {
            'Sharpe Ratio': round(sharpe, 4),
            'Sortino Ratio': round(sortino, 4),
            'Max Drawdown': round(max_drawdown * 100, 4),
            'Total Return (%)': round(total_return, 4)
        }
    
    
    @staticmethod
    def plot_residual_diagnostics(y_true, y_pred, title="Residual Diagnostics"):
        """
        Create residual diagnostic plots.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 0].axhline(0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual Distribution
        sns.histplot(residuals, kde=True, ax=axes[0, 1], bins=30)
        axes[0, 1].axvline(0, color='red', linestyle='--')
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residual')
        
        # Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals')
        
        # Residuals Over Time (if index is datetime)
        if hasattr(y_true, 'index') and isinstance(y_true.index, pd.DatetimeIndex):
            axes[1, 1].plot(y_true.index, residuals, linewidth=0.5)
            axes[1, 1].axhline(0, color='red', linestyle='--')
            axes[1, 1].set_title('Residuals Over Time')
            axes[1, 1].set_xlabel('Date')
        else:
            axes[1, 1].plot(residuals, linewidth=0.5)
            axes[1, 1].axhline(0, color='red', linestyle='--')
            axes[1, 1].set_title('Residuals Over Index')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
        return residuals