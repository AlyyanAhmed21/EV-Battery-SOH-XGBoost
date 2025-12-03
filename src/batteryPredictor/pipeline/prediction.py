import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from batteryPredictor import logger
from scipy.interpolate import interp1d

class PredictionPipeline:
    def __init__(self):
        self.model_soh = joblib.load(Path('artifacts/model_trainer/model_soh.pkl'))
        self.model_capacity = joblib.load(Path('artifacts/model_trainer/model_capacity.pkl'))
        
        # Load Reference: Raw 96% SOH file (The "DNA" of the curve)
        ref_path = Path("research/data/cell-voltage-vs-capacity-HJ3CA23041220305_779_BE_96SOH (1).csv")
        
        if ref_path.exists():
            df_raw = pd.read_csv(ref_path)
            # We calculate the "Noise Texture" of this battery
            # Texture = Raw - Smoothed
            raw_v = df_raw['cell_1']
            smooth_v = raw_v.rolling(window=20, min_periods=1, center=True).mean()
            self.noise_texture = (raw_v - smooth_v).fillna(0).values
            
            self.ref_df = pd.DataFrame({
                'capacity': df_raw['capacityRemoved'],
                'voltage': smooth_v # Base shape is smooth, we add texture later
            })
        else:
            logger.warning("Reference file not found.")
            self.ref_df = None
            self.noise_texture = np.random.normal(0, 0.002, 1000)

    def predict_bulk(self, df: pd.DataFrame):
        try:
            # 1. Standardize Names
            if 'capacityRemoved' in df.columns:
                df.rename(columns={'capacityRemoved': 'capacity'}, inplace=True)

            # 2. Feature Engineering (STRICT MODE)
            cell_cols = [c for c in df.columns if c.startswith('cell_')]
            
            # --- THE FIX: MINIMAL SMOOTHING ---
            # Reduced window from 15 -> 4.
            # We ONLY want to fix quantization noise (rounding), not hide the aging physics.
            # This allows the 'Slope' feature to remain steep, which lowers the SOH prediction to the correct level.
            df_smooth = df[cell_cols].rolling(window=4, min_periods=1, center=True).mean()
            
            # Calculate Physics Features on this sharper data
            df['pack_imbalance'] = df_smooth.max(axis=1) - df_smooth.min(axis=1)
            df['voltage_variance'] = df_smooth.var(axis=1)
            df['avg_voltage'] = df_smooth.mean(axis=1)
            
            # Gradient: Now much more sensitive to drops
            df['slope'] = np.nan_to_num(np.gradient(df['avg_voltage'], df['capacity']))
            
            # Remove Infs
            df.replace([np.inf, -np.inf], 0, inplace=True)
            
            # Curvature
            df['curvature'] = np.nan_to_num(np.gradient(df['slope']))

            # 3. Predict SOH
            feature_cols = ['capacity', 'pack_imbalance', 'voltage_variance', 'avg_voltage', 'slope', 'curvature']
            input_data = df[feature_cols]
            
            soh_predictions = self.model_soh.predict(input_data)
            
            # Use Median to be robust against outliers, but the input is now sharper
            final_soh = np.median(soh_predictions)

            # 4. Predict Remaining Capacity
            last_row = input_data.iloc[[-1]].copy()
            last_row['target_SOH'] = final_soh
            
            model_b_cols = ['capacity', 'target_SOH', 'pack_imbalance', 'voltage_variance', 'avg_voltage', 'slope', 'curvature']
            last_row = last_row[model_b_cols]
            
            predicted_remaining = self.model_capacity.predict(last_row)[0]
            current_used = last_row['capacity'].values[0]
            total_predicted_capacity = current_used + predicted_remaining

            # ---------------------------------------------------------
            # 5. VISUALIZATION GENERATION (Textured)
            # ---------------------------------------------------------
            if self.ref_df is not None:
                # Scale Reference
                ref_max_cap = self.ref_df['capacity'].max()
                scaling_ratio = total_predicted_capacity / ref_max_cap
                
                scaled_ref_cap = self.ref_df['capacity'] * scaling_ratio
                scaled_ref_volt = self.ref_df['voltage']
                
                f_ref = interp1d(scaled_ref_cap, scaled_ref_volt, kind='linear', fill_value="extrapolate")
                
                # Future Axis
                future_capacity = np.linspace(current_used, total_predicted_capacity, 800)
                future_voltage_base = f_ref(future_capacity)
                
                # Stitching Offset
                current_voltage = df['avg_voltage'].iloc[-1]
                offset = current_voltage - future_voltage_base[0]
                future_voltage_base += offset
                
                # Inject Texture
                noise_sample = np.resize(self.noise_texture, len(future_capacity))
                noise_scale = 1.0 + ((100 - final_soh) / 100)
                future_voltage_final = future_voltage_base + (noise_sample * noise_scale)

                # Uncertainty Bounds
                uncertainty_factor = (100 - final_soh) * 0.002
                uncertainty_growth = np.linspace(1, 2.5, len(future_capacity))
                
                lower_bound = future_voltage_final - (uncertainty_factor * uncertainty_growth)
                upper_bound = future_voltage_final + (uncertainty_factor * uncertainty_growth)

                projection_df = pd.DataFrame({
                    'capacity': future_capacity,
                    'avg_voltage': future_voltage_final,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
            else:
                projection_df = pd.DataFrame()

            return final_soh, df, projection_df, total_predicted_capacity

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise e