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
        
        # Load Reference
        ref_path = Path("research/data/cell-voltage-vs-capacity-HJ3CA23041220305_779_BE_96SOH (1).csv")
        if ref_path.exists():
            df_raw = pd.read_csv(ref_path)
            raw_v = df_raw['cell_1']
            smooth_v = raw_v.rolling(window=20, min_periods=1, center=True).mean()
            self.noise_texture = (raw_v - smooth_v).fillna(0).values
            self.ref_df = pd.DataFrame({'capacity': df_raw['capacityRemoved'], 'voltage': smooth_v})
        else:
            self.ref_df = None
            self.noise_texture = np.random.normal(0, 0.002, 1000)

    def predict_bulk(self, df: pd.DataFrame):
        try:
            if 'capacityRemoved' in df.columns:
                df.rename(columns={'capacityRemoved': 'capacity'}, inplace=True)

            cell_cols = [c for c in df.columns if c.startswith('cell_')]
            df_smooth = df[cell_cols].rolling(window=4, min_periods=1, center=True).mean()
            
            df['pack_imbalance'] = df_smooth.max(axis=1) - df_smooth.min(axis=1)
            df['voltage_variance'] = df_smooth.var(axis=1)
            df['avg_voltage'] = df_smooth.mean(axis=1)
            df['slope'] = np.nan_to_num(np.gradient(df['avg_voltage'], df['capacity']))
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df['curvature'] = np.nan_to_num(np.gradient(df['slope']))

            feature_cols = ['capacity', 'pack_imbalance', 'voltage_variance', 'avg_voltage', 'slope', 'curvature']
            soh_predictions = self.model_soh.predict(df[feature_cols])
            final_soh = np.median(soh_predictions)

            last_row = df[feature_cols].iloc[[-1]].copy()
            last_row['target_SOH'] = final_soh
            model_b_cols = ['capacity', 'target_SOH', 'pack_imbalance', 'voltage_variance', 'avg_voltage', 'slope', 'curvature']
            
            predicted_remaining = self.model_capacity.predict(last_row[model_b_cols])[0]
            current_used = last_row['capacity'].values[0]
            total_predicted = current_used + predicted_remaining

            projection_df = pd.DataFrame()
            if self.ref_df is not None:
                scale = total_predicted / self.ref_df['capacity'].max()
                f_ref = interp1d(self.ref_df['capacity'] * scale, self.ref_df['voltage'], kind='linear', fill_value="extrapolate")
                
                future_cap = np.linspace(current_used, total_predicted, 800)
                future_volt = f_ref(future_cap)
                
                offset = df['avg_voltage'].iloc[-1] - future_volt[0]
                noise_scale = 1.0 + ((100 - final_soh) / 100)
                texture = np.resize(self.noise_texture, len(future_cap)) * noise_scale
                future_volt = future_volt + offset + texture
                
                unc_base = min((100 - final_soh) * 0.002, 0.05)
                unc_cone = unc_base * np.linspace(0.5, 1.5, len(future_cap))
                
                projection_df = pd.DataFrame({
                    'capacity': future_cap,
                    'avg_voltage': future_volt,
                    'lower_bound': future_volt - unc_cone,
                    'upper_bound': future_volt + unc_cone
                })

            return final_soh, df, projection_df, total_predicted

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise e

    def compare_files(self, df1, df2):
        soh1, res1, proj1, cap1 = self.predict_bulk(df1)
        soh2, res2, proj2, cap2 = self.predict_bulk(df2)
        
        # --- NEW SCORING LOGIC ---
        
        # 1. Capacity Ratio (The "Veto" Factor)
        # If Cap1=20 and Cap2=200, Ratio is 0.1
        cap_ratio = min(cap1, cap2) / max(cap1, cap2)
        
        # 2. Shape Correlation (Normalized X-axis)
        norm_grid = np.linspace(0, 1, 500)
        f1 = interp1d(res1['capacity'] / res1['capacity'].max(), res1['avg_voltage'], fill_value="extrapolate")
        f2 = interp1d(res2['capacity'] / res2['capacity'].max(), res2['avg_voltage'], fill_value="extrapolate")
        y1, y2 = f1(norm_grid), f2(norm_grid)
        
        correlation = max(0, np.corrcoef(y1, y2)[0, 1])
        
        # 3. Final Calculation (Multiplicative)
        # This ensures that if EITHER shape OR capacity is bad, the score drops.
        # We weigh Capacity Ratio heavily (power of 1.5) to punish size mismatch.
        similarity = (correlation * (cap_ratio ** 1.5)) * 100
        
        return {
            "similarity": round(similarity, 2),
            "metrics": {
                "f1": {"soh": round(soh1, 2), "range": round(cap1, 2)},
                "f2": {"soh": round(soh2, 2), "range": round(cap2, 2)}
            },
            "data": {
                "f1_hist": res1, "f1_proj": proj1,
                "f2_hist": res2, "f2_proj": proj2
            }
        }