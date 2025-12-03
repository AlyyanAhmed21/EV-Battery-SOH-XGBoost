import os
import pandas as pd
import numpy as np
from batteryPredictor import logger
from sklearn.model_selection import train_test_split
from batteryPredictor.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _fill_gaps(self, df):
        """
        Applies Forward Fill and Backward Fill to remove NaNs.
        CRITICAL: Must be done per 'source_file' so data doesn't bleed 
        from one battery test to another.
        """
        logger.info("Applying Forward Fill to fix missing sensor entries...")
        
        # We group by source_file to ensure we don't fill gaps using data from a different file
        # group_keys=False prevents pandas from adding an annoying extra index
        df_filled = df.groupby('source_file', group_keys=False).apply(lambda x: x.ffill().bfill())
        
        # Double check for any remaining NaNs (e.g., if a whole column was empty)
        remaining_nans = df_filled.isna().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"Still found {remaining_nans} NaNs after filling. Filling with 0.")
            df_filled.fillna(0, inplace=True)
            
        return df_filled

    def get_derivatives(self, df):
        """
        Calculates Slope (dV/dQ) and Curvature (d2V/dQ2).
        The 'Secret Sauce' for detecting the knee.
        """
        transformed_dfs = []
        
        # We must calculate slopes separately for each file
        grouped = df.groupby('source_file')
        
        for name, group in grouped:
            # Ensure strictly sorted by capacity so derivatives make sense
            group = group.sort_values(by='capacity')
            
            # 1. Slope (dV/dQ) using Gradient
            # We take the gradient of Cell 1 (representative) or Average Voltage
            # Let's calculate an Average Voltage column first if it doesn't exist
            cell_cols = [c for c in group.columns if c.startswith('cell_')]
            avg_v = group[cell_cols].mean(axis=1)
            
            # Calculate Gradients
            dV = np.gradient(avg_v)
            dQ = np.gradient(group['capacity'])
            
            # Handle division by zero (if capacity doesn't change between rows)
            with np.errstate(divide='ignore', invalid='ignore'):
                slope = dV / dQ
            
            # 2. Curvature (Change in slope)
            curvature = np.gradient(slope)
            
            # Clean up Infs/NaNs created by gradient math
            group['slope'] = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)
            group['curvature'] = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
            
            transformed_dfs.append(group)
            
        return pd.concat(transformed_dfs)

    def feature_engineering(self):
        # 1. Load Data
        df = pd.read_csv(self.config.data_path)
        logger.info(f"Loaded Merged Data. Shape: {df.shape}")

        # 2. FILL MISSING VALUES (Your Request)
        df = self._fill_gaps(df)

        # 3. Calculate Cell Stats (Imbalance)
        logger.info("Calculating Cell Imbalance and Variance...")
        cell_cols = [c for c in df.columns if c.startswith('cell_')]
        
        # Max - Min voltage across the pack
        df['pack_imbalance'] = df[cell_cols].max(axis=1) - df[cell_cols].min(axis=1)
        # Variance
        df['voltage_variance'] = df[cell_cols].var(axis=1)
        # Avg Voltage
        df['avg_voltage'] = df[cell_cols].mean(axis=1)

        # 4. Physics Derivatives (Slope)
        logger.info("Calculating Derivatives (Slope & Curvature)...")
        df = self.get_derivatives(df)

        # 5. Calculate Target B: Remaining Capacity
        # Logic: We find the MAX capacity for each file, and subtract current capacity
        logger.info("Calculating Remaining Capacity Targets...")
        max_caps = df.groupby('source_file')['capacity'].max()
        df['max_file_capacity'] = df['source_file'].map(max_caps)
        df['remaining_capacity'] = df['max_file_capacity'] - df['capacity']

        # 6. Cleanup
        # Remove string columns and individual cells (reduce dimensionality)
        drop_cols = ['source_file', 'max_file_capacity'] + cell_cols
        df_clean = df.drop(columns=drop_cols)
        
        # Final sanity check for Infinite numbers
        df_clean.replace([np.inf, -np.inf], 0, inplace=True)

        logger.info(f"Feature Engineering Complete. Final Shape: {df_clean.shape}")
        logger.info(f"Final Columns: {df_clean.columns.tolist()}")

        # 7. Split Data
        # We split randomly. Ideally, we should group split, but random is okay for R&D POC
        train, test = train_test_split(df_clean, test_size=0.25, random_state=42)

        # 8. Save
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split into Train and Test sets.")
        logger.info(f"Train Path: {os.path.join(self.config.root_dir, 'train.csv')}")