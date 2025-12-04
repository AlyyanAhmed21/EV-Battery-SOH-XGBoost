from batteryPredictor.constants import *
from batteryPredictor.utils.common import read_yaml, create_directories
from batteryPredictor.entity import (DataIngestionConfig, 
                                     DataTransformationConfig,
                                     ModelTrainerConfig,
                                     ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_data_dir=config.source_data_dir,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config

    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            status_file=config.status_file
        )

        return data_transformation_config


    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.XGBoost

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_soh_name=config.model_soh_name,
            model_capacity_name=config.model_capacity_name,
            params=params
        )

        return model_trainer_config
    
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path, 
            test_data_path=config.test_data_path,
            model_soh_path=config.model_soh_path,
            model_capacity_path=config.model_capacity_path,
            metric_file_name=config.metric_file_name
        )

        return model_evaluation_config