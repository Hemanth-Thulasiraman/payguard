# src/training/config.py

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    # Data
    train_path: str = "data/processed/train.parquet"
    val_path: str = "data/processed/val.parquet"

    # Model output
    model_path: str = "models/lgbm_fraud.pkl"

    # LightGBM hyperparameters
    n_estimators: int = 1000
    learning_rate: float = 0.05
    num_leaves: int = 64
    min_child_samples: int = 100
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1

    # Imbalance handling
    scale_pos_weight: float = 27.0

    # Training settings
    early_stopping_rounds: int = 50
    eval_metric: str = "average_precision"
    seed: int = 42
    n_jobs: int = -1

    # MLflow
    experiment_name: str = "payguard-fraud-detection"
    run_name: str = "lgbm-baseline"

    class Config:
        arbitrary_types_allowed = True