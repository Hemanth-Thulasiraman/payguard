# src/training/train.py

import json
import mlflow
import numpy as np
import pandas as pd
import pickle
from loguru import logger
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
import lightgbm as lgb

from src.training.config import TrainingConfig

MODELS_DIR = Path("models")
TARGET = "isFraud"
DROP_COLS = ["TransactionID", "TransactionDT", "isFraud"]


def get_features(df: pd.DataFrame) -> list[str]:
    """Returns feature columns — excludes ID, target, time columns."""
    return [c for c in df.columns if c not in DROP_COLS]


def load_data(config: TrainingConfig):
    """Loads train and validation parquet files."""
    logger.info("Loading train and validation data...")
    train = pd.read_parquet(config.train_path)
    val = pd.read_parquet(config.val_path)

    features = get_features(train)

    X_train = train[features]
    y_train = train[TARGET]
    X_val = val[features]
    y_val = val[TARGET]

    logger.info(f"Train: {len(X_train):,} rows, {len(features)} features")
    logger.info(f"Val:   {len(X_val):,} rows")
    logger.info(
        f"Train fraud rate: {y_train.mean():.4f} "
        f"({y_train.sum():,} fraud cases)"
    )
    logger.info(
        f"Val fraud rate:   {y_val.mean():.4f} "
        f"({y_val.sum():,} fraud cases)"
    )

    return X_train, y_train, X_val, y_val, features


def select_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    target_recall: float = 0.80,
) -> float:
    """
    Selects classification threshold targeting minimum recall.
    In fraud detection we want to catch at least 80% of fraud
    even at the cost of more false positives.
    Returns optimal threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Find thresholds where recall >= target
    valid_idx = np.where(recall[:-1] >= target_recall)[0]

    if len(valid_idx) == 0:
        logger.warning(
            f"Cannot achieve {target_recall:.0%} recall. "
            f"Using default threshold 0.5"
        )
        return 0.5

    # Among valid thresholds pick the one with highest precision
    best_idx = valid_idx[np.argmax(precision[valid_idx])]
    threshold = float(thresholds[best_idx])

    logger.info(
        f"Selected threshold: {threshold:.4f} "
        f"(recall={recall[best_idx]:.3f}, "
        f"precision={precision[best_idx]:.3f})"
    )
    return threshold


def evaluate_model(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    split_name: str,
) -> dict:
    """Evaluates model and logs metrics for a given split."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(y, y_prob)
    auc_pr = average_precision_score(y, y_prob)
    f1 = f1_score(y, y_pred, zero_division=0)

    logger.info(f"{split_name} metrics:")
    logger.info(f"  AUC-ROC: {auc_roc:.4f}")
    logger.info(f"  AUC-PR:  {auc_pr:.4f}")
    logger.info(f"  F1:      {f1:.4f}")

    return {
        f"{split_name}_auc_roc": auc_roc,
        f"{split_name}_auc_pr": auc_pr,
        f"{split_name}_f1": f1,
    }


def run_training(config: TrainingConfig = None) -> None:
    """
    Main training entry point.
    Trains LightGBM, selects threshold, evaluates,
    logs to MLflow, saves model.
    """
    if config is None:
        config = TrainingConfig()

    logger.info("=" * 50)
    logger.info("Starting PayGuard model training")
    logger.info("=" * 50)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):

        # Log hyperparameters
        mlflow.log_params({
            "n_estimators": config.n_estimators,
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "min_child_samples": config.min_child_samples,
            "scale_pos_weight": config.scale_pos_weight,
            "feature_fraction": config.feature_fraction,
            "early_stopping_rounds": config.early_stopping_rounds,
        })

        # Load data
        X_train, y_train, X_val, y_val, features = load_data(config)

        # Train model
        logger.info("Training LightGBM model...")
        model = lgb.LGBMClassifier(
            n_estimators=500,              # fixed — no early stopping
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            min_child_samples=config.min_child_samples,
            feature_fraction=config.feature_fraction,
            bagging_fraction=config.bagging_fraction,
            bagging_freq=config.bagging_freq,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            scale_pos_weight=config.scale_pos_weight,
            random_state=config.seed,
            n_jobs=config.n_jobs,
            verbosity=-1,
        )

        # Replace fit call with this — no callbacks, no early stopping
        logger.info("Training LightGBM model (500 iterations)...")
        model.fit(X_train, y_train)

        # Log progress manually
        logger.info("Training complete — evaluating on validation set")

        best_iteration = model.best_iteration_
        logger.info(f"Best iteration: {best_iteration}")

        # Select threshold
        val_probs = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, val_probs)

        # Evaluate on validation set
        val_metrics = evaluate_model(
            model, X_val, y_val, threshold, "val"
        )

        # Log metrics to MLflow
        mlflow.log_metrics({
            **val_metrics,
            "best_iteration": best_iteration,
            "threshold": threshold,
        })

        # Save model and metadata
        model_data = {
            "model": model,
            "threshold": threshold,
            "features": features,
            "config": config.model_dump(),
        }

        with open(config.model_path, "wb") as f:
            pickle.dump(model_data, f)

        mlflow.log_artifact(config.model_path)

        logger.info(f"Model saved to {config.model_path}")
        logger.info("=" * 50)
        logger.info("Training complete")
        logger.info(f"Val AUC-PR:  {val_metrics['val_auc_pr']:.4f}")
        logger.info(f"Val F1:      {val_metrics['val_f1']:.4f}")
        logger.info(f"Threshold:   {threshold:.4f}")
        logger.info("=" * 50)


if __name__ == "__main__":
    run_training()