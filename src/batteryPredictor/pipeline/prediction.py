import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from batteryPredictor import logger

class PredictionPipeline:
    def __init__(self):
        self.model_soh = joblib.load(Path('artifacts/model_trainer/model_soh.pkl'))
        self.model_capacity = joblib.load(Path('artifacts/model_trainer/model_capacity.pkl'))

    def predict_bulk(self, df: pd.DataFrame):
        """
        Process an entire CSV file (Discharge Curve).
        Returns:
            - avg_soh (float): The overall health of this battery.
            - results_df (DataFrame): The input data + predictions.
        """
        try:
            # 1. Standardize Column Names
            if 'capacityRemoved' in df.columns:
                df.rename(columns={'capacityRemoved': 'capacity'}, inplace=True)

            # 2. Feature Engineering (Batch Physics)
            cell_cols = [c for c in df.columns if c.startswith('cell_')]
            
            # Imbalance & Variance
            df['pack_imbalance'] = df[cell_cols].max(axis=1) - df[cell_cols].min(axis=1)
            df['voltage_variance'] = df[cell_cols].var(axis=1)
            df['avg_voltage'] = df[cell_cols].mean(axis=1)
            
            # Gradients (Vectorized for the whole file)
            df['slope'] = np.gradient(df['avg_voltage'], df['capacity'])
            
            # Handle potential infinity/NaNs from gradient
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)

            # Curvature
            df['curvature'] = np.gradient(df['slope'])

            # 3. Align Columns for XGBoost (Model A - SOH)
            # Model A expects: ['capacity', 'pack_imbalance', 'voltage_variance', 'avg_voltage', 'slope', 'curvature']
            feature_cols = [
                'capacity', 
                'pack_imbalance', 
                'voltage_variance', 
                'avg_voltage', 
                'slope', 
                'curvature'
            ]
            
            input_data = df[feature_cols]

            # 4. Predict SOH
            soh_predictions = self.model_soh.predict(input_data)
            df['predicted_SOH'] = soh_predictions
            final_soh = np.median(soh_predictions)

            # 5. Predict Remaining Capacity (The Trend)
            input_data_cap = input_data.copy()
            input_data_cap['target_SOH'] = df['predicted_SOH']
            
            # --- FIX: REORDER COLUMNS FOR MODEL B ---
            # Model B expects 'target_SOH' to be the 2nd column
            model_b_cols = [
                'capacity', 
                'target_SOH', 
                'pack_imbalance', 
                'voltage_variance', 
                'avg_voltage', 
                'slope', 
                'curvature'
            ]
            
            # Force the dataframe to match the training structure
            input_data_cap = input_data_cap[model_b_cols]
            
            range_predictions = self.model_capacity.predict(input_data_cap)
            df['predicted_remaining_capacity'] = range_predictions

            return final_soh, df

        except Exception as e:
            logger.error(f"Bulk Prediction failed: {e}")
            raise e