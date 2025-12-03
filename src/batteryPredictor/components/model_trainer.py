import pandas as pd
import os
from batteryPredictor import logger
from xgboost import XGBRegressor
from batteryPredictor.entity import ModelTrainerConfig
import joblib

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # ---------------------------------------------------------
        # MODEL A: SOH ESTIMATOR
        # ---------------------------------------------------------
        logger.info("🤖 Training Model A: SOH Estimator...")
        
        # Features for SOH (Everything EXCEPT target columns)
        # We drop 'remaining_capacity' because that's the answer for Model B
        X_train_soh = train_data.drop(['target_SOH', 'remaining_capacity'], axis=1)
        y_train_soh = train_data['target_SOH']
        
        # Initialize XGBoost with params from params.yaml
        xgb_soh = XGBRegressor(
            n_estimators=self.config.params['n_estimators'],
            learning_rate=self.config.params['learning_rate'],
            max_depth=self.config.params['max_depth'],
            n_jobs=self.config.params['n_jobs']
        )
        
        xgb_soh.fit(X_train_soh, y_train_soh)

        # Save Model A
        save_path_soh = os.path.join(self.config.root_dir, self.config.model_soh_name)
        joblib.dump(xgb_soh, save_path_soh)
        logger.info(f"✅ Model A saved at: {save_path_soh}")

        # ---------------------------------------------------------
        # MODEL B: REMAINING CAPACITY ESTIMATOR
        # ---------------------------------------------------------
        logger.info("🤖 Training Model B: Range Estimator...")

        # Features for Capacity (Includes SOH as a helpful feature)
        # We assume during inference, we will pass the *Predicted SOH* here.
        # During training, we use the *Real SOH* (Teacher Forcing).
        X_train_cap = train_data.drop(['remaining_capacity'], axis=1)
        y_train_cap = train_data['remaining_capacity']

        xgb_cap = XGBRegressor(
            n_estimators=self.config.params['n_estimators'],
            learning_rate=self.config.params['learning_rate'],
            max_depth=self.config.params['max_depth'],
            n_jobs=self.config.params['n_jobs']
        )

        xgb_cap.fit(X_train_cap, y_train_cap)

        # Save Model B
        save_path_cap = os.path.join(self.config.root_dir, self.config.model_capacity_name)
        joblib.dump(xgb_cap, save_path_cap)
        logger.info(f"✅ Model B saved at: {save_path_cap}")