from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_data_dir: Path
    local_data_file: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    status_file: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_soh_name: str
    model_capacity_name: str
    params: dict

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_soh_path: Path
    model_capacity_path: Path
    metric_file_name: Path