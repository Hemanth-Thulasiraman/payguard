# src/evaluation/evaluate.py

import json
import mlflow
import numpy as np
import pandas as pd
import pickle
from loguru import logger
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

TEST_PATH = Path("data/processed/test.parquet")
MODEL_PATH = Path("models/lgbm_fraud.pkl")
REPORTS_DIR = Path("data/validation_reports")
TARGET = "isFraud"
DROP_COLS = ["TransactionID", "TransactionDT", "isFraud"]


def load_model() -> tuple:
    """
    Loads saved model, threshold, and feature list.
    Returns model, threshold, features.
    """
    logger.info(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    threshold = model_data["threshold"]
    features = model_data["features"]

    logger.info(f"Model loaded — threshold: {threshold:.4f}")
    logger.info(f"Features: {len(features)}")
    return model, threshold, features


def load_test_data(features: list[str]) -> tuple:
    """
    Loads test set and aligns features with training.
    """
    logger.info(f"Loading test data from {TEST_PATH}")
    test = pd.read_parquet(TEST_PATH)

    # Align feature columns with training
    available = [f for f in features if f in test.columns]
    missing = [f for f in features if f not in test.columns]

    if missing:
        logger.warning(f"Missing features in test set: {missing}")

    X_test = test[available]
    y_test = test[TARGET]

    logger.info(f"Test: {len(X_test):,} rows")
    logger.info(
        f"Fraud rate: {y_test.mean():.4f} "
        f"({y_test.sum():,} fraud cases)"
    )
    return X_test, y_test


def compute_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """
    Computes full evaluation metrics at given threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Precision and recall at threshold
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    idx = np.argmin(np.abs(
        np.array([
            (y_prob >= t).mean()
            for t in np.linspace(0, 1, len(precision))
        ]) - (y_pred.mean())
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fraud_caught = int(tp)
    fraud_missed = int(fn)
    false_alarms = int(fp)

    logger.info("=" * 50)
    logger.info("Test Set Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"AUC-ROC:        {auc_roc:.4f}")
    logger.info(f"AUC-PR:         {auc_pr:.4f}")
    logger.info(f"F1 @ threshold: {f1:.4f}")
    logger.info(f"Threshold:      {threshold:.4f}")
    logger.info("=" * 50)
    logger.info("Business Metrics:")
    logger.info(f"  Fraud caught:     {fraud_caught:,} / {fraud_caught+fraud_missed:,}")
    logger.info(f"  Fraud missed:     {fraud_missed:,}")
    logger.info(f"  False alarms:     {false_alarms:,}")
    logger.info(
        f"  Recall:           "
        f"{fraud_caught/(fraud_caught+fraud_missed):.1%}"
    )
    logger.info(
        f"  Precision:        "
        f"{fraud_caught/(fraud_caught+false_alarms):.1%}"
    )
    logger.info(
        f"  For every 100 alerts: "
        f"{fraud_caught/(fraud_caught+false_alarms)*100:.0f} real fraud, "
        f"{false_alarms/(fraud_caught+false_alarms)*100:.0f} false alarms"
    )

    return {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "f1": float(f1),
        "threshold": float(threshold),
        "fraud_caught": fraud_caught,
        "fraud_missed": fraud_missed,
        "false_alarms": int(false_alarms),
        "recall": float(fraud_caught / (fraud_caught + fraud_missed)),
        "precision": float(
            fraud_caught / (fraud_caught + false_alarms)
        ),
    }


def per_class_report(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> None:
    """Logs per-class classification report."""
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(
        y_true, y_pred,
        target_names=["legitimate", "fraud"],
        zero_division=0,
    )
    logger.info("Per-class report:")
    for line in report.split("\n"):
        if line.strip():
            logger.info(f"  {line}")


def run_evaluation() -> dict:
    """
    Main entry point for evaluation pipeline.
    Runs on test set only — never touches training data.
    Logs results to MLflow evaluation experiment.
    """
    logger.info("=" * 50)
    logger.info("Starting PayGuard evaluation on test set")
    logger.info("=" * 50)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model, threshold, features = load_model()
    X_test, y_test = load_test_data(features)

    logger.info("Running inference on test set...")
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_prob, threshold)
    per_class_report(y_test, y_prob, threshold)

    # Save report
    report_path = REPORTS_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    # Log to MLflow
    mlflow.set_experiment("payguard-evaluation")
    with mlflow.start_run(run_name="test-set-evaluation"):
        mlflow.log_metrics({
            "test_auc_roc": metrics["auc_roc"],
            "test_auc_pr": metrics["auc_pr"],
            "test_f1": metrics["f1"],
            "test_recall": metrics["recall"],
            "test_precision": metrics["precision"],
        })
        mlflow.log_artifact(str(report_path))

    logger.info("Evaluation complete")
    return metrics


if __name__ == "__main__":
    run_evaluation()