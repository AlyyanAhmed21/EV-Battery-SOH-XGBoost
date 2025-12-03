import os
import re
import pandas as pd
from batteryPredictor import logger
from batteryPredictor.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _extract_soh_from_filename(self, filename):
        """
        Parses filenames to find the SOH value.
        Examples: 
        - "synthetic_779_BE_90SOH.csv" -> 90
        - "...99SOH (1).csv" -> 99
        """
        match = re.search(r"(\d+)SOH", filename)
        if match:
            return int(match.group(1))
        return None

    def merge_and_save_data(self):
        """
        Reads all CSVs from source_dir, adds 'SOH' column, 
        merges them, and saves to artifacts.
        """
        logger.info(f"Scanning for data in: {self.config.source_data_dir}")
        
        all_data_frames = []
        files = os.listdir(self.config.source_data_dir)
        
        # Sort files to keep log orderly
        files.sort()

        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(self.config.source_data_dir, file)
                
                # 1. Extract SOH Label
                soh_value = self._extract_soh_from_filename(file)
                
                if soh_value is None:
                    logger.warning(f"Skipping file (No SOH found in name): {file}")
                    continue

                # 2. Read CSV
                try:
                    df = pd.read_csv(file_path)
                    
                    # 3. Standardization (Ensure columns match)
                    # Some files might have 'capacity' vs 'capacityRemoved'
                    # We standardize to 'capacity' for the model
                    if 'capacityRemoved' in df.columns:
                        df.rename(columns={'capacityRemoved': 'capacity'}, inplace=True)
                    
                    # 4. Add Target Column
                    df['target_SOH'] = soh_value
                    
                    # 5. Add Source Column (Useful for debugging later)
                    df['source_file'] = file
                    
                    all_data_frames.append(df)
                    logger.info(f"Loaded: {file} | Rows: {len(df)} | SOH: {soh_value}")
                    
                except Exception as e:
                    logger.error(f"Failed to read file {file}: {e}")

        # 6. Merge All
        if all_data_frames:
            master_df = pd.concat(all_data_frames, ignore_index=True)
            
            # Save to artifacts
            master_df.to_csv(self.config.local_data_file, index=False)
            logger.info(f"✅ Merged Data Saved! Total Rows: {len(master_df)}")
            logger.info(f"Saved to: {self.config.local_data_file}")
        else:
            logger.error("No valid CSV files found to merge!")