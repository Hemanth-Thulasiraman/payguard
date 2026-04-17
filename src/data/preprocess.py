# src/data/preprocess.py

import json
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
NULL_THRESHOLD = 0.95
CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28",
    "id_29", "id_30", "id_31", "id_33", "id_34", "id_35",
    "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo",
]


def drop_high_null_features(
    df: pd.DataFrame,
    threshold: float = NULL_THRESHOLD,
) -> pd.DataFrame:
    """
    Drops features where null rate exceeds threshold.
    Never drops target or key columns.
    """
    null_rates = df.isnull().mean()
    cols_to_drop = null_rates[null_rates > threshold].index.tolist()

    protected = ["isFraud", "TransactionID", "TransactionDT"]
    cols_to_drop = [c for c in cols_to_drop if c not in protected]

    logger.info(
        f"Dropping {len(cols_to_drop)} features "
        f"with >{threshold:.0%} null rate"
    )
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Remaining features: {df.shape[1]}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new features from TransactionDT and TransactionAmt.
    TransactionDT is seconds elapsed from a reference point.
    """
    logger.info("Engineering features...")

    df["hour"] = (df["TransactionDT"] / 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] / 86400) % 7
    df["day_of_month"] = (df["TransactionDT"] / 86400) % 30

    df["amount_log"] = np.log1p(df["TransactionAmt"])
    df["amount_cents"] = df["TransactionAmt"] % 1
    df["is_round_amount"] = (
        df["TransactionAmt"] % 1 == 0
    ).astype(int)

    if "card1" in df.columns and "card2" in df.columns:
        df["card1_card2"] = (
            df["card1"].astype(str) + "_" + df["card2"].astype(str)
        )
        CATEGORICAL_COLS.append("card1_card2")

    logger.info(f"Engineered 7 new features")
    logger.info(f"Total features after engineering: {df.shape[1]}")
    return df


def encode_categoricals(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Encodes categorical columns as integer codes.
    Saves mapping dict so inference uses identical codes.
    Returns DataFrame and mapping dict.
    """
    existing_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    logger.info(f"Encoding {len(existing_cats)} categorical columns")

    mappings = {}
    for col in existing_cats:
        df[col] = df[col].fillna("unknown").astype(str)
        categories = sorted(df[col].unique().tolist())
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        mappings[col] = mapping
        df[col] = df[col].map(mapping).fillna(-1).astype(int)

    logger.info(f"Category mappings created for {len(mappings)} columns")
    return df, mappings


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset by time — not randomly.
    Train: first 70%, Val: next 15%, Test: last 15%.
    Critical: never shuffle before splitting.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logger.info("Temporal split (no shuffling):")
    logger.info(
        f"  Train: {len(train):,} rows | "
        f"fraud rate: {train['isFraud'].mean():.4f}"
    )
    logger.info(
        f"  Val:   {len(val):,} rows   | "
        f"fraud rate: {val['isFraud'].mean():.4f}"
    )
    logger.info(
        f"  Test:  {len(test):,} rows  | "
        f"fraud rate: {test['isFraud'].mean():.4f}"
    )

    return train, val, test


def run_preprocessing() -> None:
    """
    Main entry point for preprocessing pipeline.
    Loads merged data, cleans, engineers features,
    encodes categoricals, splits temporally, saves splits.
    Also saves category mappings for serving layer.
    """
    logger.info("=" * 50)
    logger.info("Starting PayGuard preprocessing")
    logger.info("=" * 50)

    df = pd.read_parquet("data/processed/raw_merged.parquet")
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    df = drop_high_null_features(df)
    df = engineer_features(df)
    df, mappings = encode_categoricals(df)

    # Save category mappings for serving layer
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mappings_path = PROCESSED_DIR / "category_mappings.json"
    with open(mappings_path, "w") as f:
        json.dump(mappings, f, indent=2)
    logger.info(f"Category mappings saved to {mappings_path}")

    train, val, test = temporal_split(df)

    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    logger.info("Splits saved to data/processed/")
    logger.info("Preprocessing complete")


if __name__ == "__main__":
    run_preprocessing()