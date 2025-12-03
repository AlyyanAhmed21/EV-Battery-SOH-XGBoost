import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from batteryPredictor.utils.common import save_json
from batteryPredictor.entity import ModelEvaluationConfig
from batteryPredictor import logger
import joblib
import numpy as np
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def evaluation(self):
        logger.info("Loading Test Data and Models...")
        test_data = pd.read_csv(self.config.test_data_path)
        model_soh = joblib.load(self.config.model_soh_path)
        model_cap = joblib.load(self.config.model_capacity_path)

        # -----------------------------------
        # EVALUATE MODEL A (SOH)
        # -----------------------------------
        X_test_soh = test_data.drop(['target_SOH', 'remaining_capacity'], axis=1)
        y_test_soh = test_data['target_SOH']

        predicted_soh = model_soh.predict(X_test_soh)

        (rmse_soh, mae_soh, r2_soh) = self.eval_metrics(y_test_soh, predicted_soh)

        # -----------------------------------
        # EVALUATE MODEL B (CAPACITY)
        # -----------------------------------
        # Ideally, we use the PREDICTED SOH to test Model B (Real-world scenario)
        # But for pure model validation, let's stick to X_test features
        X_test_cap = test_data.drop(['remaining_capacity'], axis=1)
        y_test_cap = test_data['remaining_capacity']

        predicted_cap = model_cap.predict(X_test_cap)

        (rmse_cap, mae_cap, r2_cap) = self.eval_metrics(y_test_cap, predicted_cap)

        # -----------------------------------
        # SAVE METRICS
        # -----------------------------------
        scores = {
            "Model_A_SOH": {
                "RMSE": rmse_soh,
                "MAE": mae_soh,
                "R2_Score": r2_soh
            },
            "Model_B_Capacity": {
                "RMSE": rmse_cap,
                "MAE": mae_cap,
                "R2_Score": r2_cap
            }
        }

        save_json(path=Path(self.config.metric_file_name), data=scores)
        logger.info(f"Evaluation Metrics saved to: {self.config.metric_file_name}")
        logger.info(f"SOH Accuracy (R2): {r2_soh:.4f}")
        logger.info(f"Capacity Accuracy (R2): {r2_cap:.4f}")