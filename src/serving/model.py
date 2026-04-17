# src/serving/model.py

import json
import numpy as np
import pandas as pd
import pickle
from loguru import logger
from pathlib import Path

MODEL_PATH = Path("models/lgbm_fraud.pkl")
MAPPINGS_PATH = Path("data/processed/category_mappings.json")

_model = None
_threshold = None
_features = None
_cat_mappings = {}
MOCK_MODE = not MODEL_PATH.exists()


def load_model() -> None:
    """
    Loads LightGBM model, threshold, feature list,
    and category mappings. Called once at API startup.
    """
    global _model, _threshold, _features, _cat_mappings

    if MOCK_MODE:
        logger.warning("Model not found — running in MOCK MODE")
        _model = "mock"
        _threshold = 0.145
        _features = []
        _cat_mappings = {}
        return

    logger.info(f"Loading model from {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    _model = model_data["model"]
    _threshold = model_data["threshold"]
    _features = model_data["features"]

    logger.info(f"Model loaded — threshold: {_threshold:.4f}")
    logger.info(f"Features: {len(_features)}")

    # Load category mappings saved during preprocessing
    if MAPPINGS_PATH.exists():
        with open(MAPPINGS_PATH) as f:
            _cat_mappings = json.load(f)
        logger.info(
            f"Category mappings loaded: {len(_cat_mappings)} columns"
        )
    else:
        logger.warning(
            f"No category mappings found at {MAPPINGS_PATH}. "
            f"Categorical features will be NaN at inference."
        )


def is_model_loaded() -> bool:
    return _model is not None


def get_threshold() -> float:
    return _threshold or 0.145


def get_n_features() -> int:
    return len(_features) if _features else 0


def build_feature_row(transaction: dict) -> pd.DataFrame:
    """
    Builds a single-row DataFrame from transaction dict.
    Uses saved category mappings — identical codes to training.
    Fills missing features with NaN — LightGBM handles nulls.
    """
    # Start with all features as NaN
    row = {f: np.nan for f in _features}

    # Fill in provided fields
    for key, value in transaction.items():
        if key in row:
            row[key] = value

    df = pd.DataFrame([row])

    # Engineer same features as preprocess.py
    if "TransactionDT" in df.columns:
        df["hour"] = (df["TransactionDT"] / 3600) % 24
        df["day_of_week"] = (df["TransactionDT"] / 86400) % 7
        df["day_of_month"] = (df["TransactionDT"] / 86400) % 30

    if "TransactionAmt" in df.columns:
        df["amount_log"] = np.log1p(df["TransactionAmt"])
        df["amount_cents"] = df["TransactionAmt"] % 1
        df["is_round_amount"] = (
            df["TransactionAmt"] % 1 == 0
        ).astype(int)

    # Apply saved category mappings
    # This is the single source of truth for categorical encoding
    for col, mapping in _cat_mappings.items():
        if col in df.columns:
            raw_val = df[col].fillna("unknown").astype(str).iloc[0]
            # -1 for unseen categories (new values not in training)
            df[col] = mapping.get(raw_val, -1)

    # Keep only training features in correct order
    available = [f for f in _features if f in df.columns]
    df = df[available]

    return df


def score_transaction(transaction: dict) -> dict:
    """
    Scores a single transaction for fraud probability.
    Returns probability, classification, and risk level.
    """
    if not is_model_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Mock mode
    if MOCK_MODE:
        return {
            "fraud_probability": 0.08,
            "is_fraud": False,
            "risk_level": "LOW",
            "threshold_used": 0.145,
        }

    df = build_feature_row(transaction)
    prob = float(_model.predict_proba(df)[:, 1][0])
    is_fraud = prob >= _threshold

    if prob >= 0.5:
        risk = "HIGH"
    elif prob >= _threshold:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": bool(is_fraud),
        "risk_level": risk,
        "threshold_used": round(_threshold, 4),
    }