import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from batteryPredictor.utils.common import save_json
from batteryPredictor.entity import ModelEvaluationConfig
from batteryPredictor import logger

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def save_plot(self, actual, pred, title, filename):
        """Generates Parity Plot to visualize accuracy"""
        plt.figure(figsize=(8, 8))
        plt.scatter(actual, pred, alpha=0.3, color='blue', s=10)
        
        min_val = min(actual.min(), pred.min())
        max_val = max(actual.max(), pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.title(f"{title}: Predicted vs Actual")
        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        plt.grid(True, alpha=0.3)
        
        save_path = Path(self.config.root_dir) / filename
        plt.savefig(save_path)
        plt.close()

    def evaluation(self):
        logger.info("Loading Data and Models...")
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        model_soh = joblib.load(self.config.model_soh_path)
        model_cap = joblib.load(self.config.model_capacity_path)

        # ===============================================
        # 1. EVALUATE MODEL A (SOH)
        # ===============================================
        # Prepare Features
        X_train_soh = train_data.drop(['target_SOH', 'remaining_capacity'], axis=1)
        y_train_soh = train_data['target_SOH']
        
        X_test_soh = test_data.drop(['target_SOH', 'remaining_capacity'], axis=1)
        y_test_soh = test_data['target_SOH']

        # Predict
        pred_train_soh = model_soh.predict(X_train_soh)
        pred_test_soh = model_soh.predict(X_test_soh)

        # Metrics
        (rmse_train_soh, mae_train_soh, r2_train_soh) = self.eval_metrics(y_train_soh, pred_train_soh)
        (rmse_test_soh, mae_test_soh, r2_test_soh) = self.eval_metrics(y_test_soh, pred_test_soh)

        # Save Visuals (Test Set Only)
        self.save_plot(y_test_soh, pred_test_soh, "SOH Model (Test)", "SOH_parity_plot.png")

        # ===============================================
        # 2. EVALUATE MODEL B (CAPACITY)
        # ===============================================
        # Fix Column Order for Model B
        cols = train_data.columns.tolist() # e.g. [capacity, target_SOH, imbalance...]
        # We need to drop 'remaining_capacity' but keep 'target_SOH'
        # Model B was trained on X without 'remaining_capacity'
        
        # Prepare Train
        X_train_cap = train_data.drop(['remaining_capacity'], axis=1)
        y_train_cap = train_data['remaining_capacity']
        
        # Prepare Test
        X_test_cap = test_data.drop(['remaining_capacity'], axis=1)
        y_test_cap = test_data['remaining_capacity']

        # Reorder Columns explicitly to match training signature if needed
        # (Assuming data_transformation saved them in consistent order)

        # Predict
        pred_train_cap = model_cap.predict(X_train_cap)
        pred_test_cap = model_cap.predict(X_test_cap)

        # Metrics
        (rmse_train_cap, mae_train_cap, r2_train_cap) = self.eval_metrics(y_train_cap, pred_train_cap)
        (rmse_test_cap, mae_test_cap, r2_test_cap) = self.eval_metrics(y_test_cap, pred_test_cap)

        # Save Visuals (Test Set Only)
        self.save_plot(y_test_cap, pred_test_cap, "Range Model (Test)", "Capacity_parity_plot.png")

        # ===============================================
        # 3. SAVE COMPLETE REPORT
        # ===============================================
        scores = {
            "Model_A_SOH": {
                "Train_R2": r2_train_soh,
                "Test_R2": r2_test_soh,
                "Train_MAE": mae_train_soh,
                "Test_MAE": mae_test_soh,
                "Train_RMSE": rmse_train_soh,
                "Test_RMSE": rmse_test_soh
            },
            "Model_B_Capacity": {
                "Train_R2": r2_train_cap,
                "Test_R2": r2_test_cap,
                "Train_MAE": mae_train_cap,
                "Test_MAE": mae_test_cap,
                "Train_RMSE": rmse_train_cap,
                "Test_RMSE": rmse_test_cap
            }
        }

        save_json(path=Path(self.config.metric_file_name), data=scores)
        
        logger.info("Evaluation Complete.")
        logger.info(f"SOH Train R2: {r2_train_soh:.4f} | Test R2: {r2_test_soh:.4f}")
        logger.info(f"Cap Train R2: {r2_train_cap:.4f} | Test R2: {r2_test_cap:.4f}")