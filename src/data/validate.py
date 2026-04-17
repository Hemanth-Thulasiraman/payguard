# src/data/validate.py

import pandas as pd
from loguru import logger

REQUIRED_COLUMNS = ["TransactionID", "isFraud", "TransactionAmt", "TransactionDT"]
EXPECTED_CLASSES = {0, 1}
MIN_FRAUD_RATE = 0.01
MAX_FRAUD_RATE = 0.10
MAX_NULL_THRESHOLD = 0.95


def check_required_columns(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False
    logger.info(f"✅ Required columns check passed")
    return True


def check_no_nulls_in_key_columns(df: pd.DataFrame) -> bool:
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    has_nulls = null_counts[null_counts > 0]
    if not has_nulls.empty:
        logger.error(f"Nulls in key columns: {has_nulls.to_dict()}")
        return False
    logger.info("✅ Key column null check passed")
    return True


def check_class_values(df: pd.DataFrame) -> bool:
    actual = set(df["isFraud"].unique())
    if actual != EXPECTED_CLASSES:
        logger.error(f"Unexpected class values: {actual}")
        return False
    logger.info(f"✅ Class values check passed: {actual}")
    return True


def check_fraud_rate(df: pd.DataFrame) -> bool:
    fraud_rate = df["isFraud"].mean()
    if not (MIN_FRAUD_RATE <= fraud_rate <= MAX_FRAUD_RATE):
        logger.error(
            f"Fraud rate {fraud_rate:.4f} outside expected range "
            f"[{MIN_FRAUD_RATE}, {MAX_FRAUD_RATE}]"
        )
        return False
    logger.info(f"✅ Fraud rate check passed: {fraud_rate:.4f}")
    return True


def check_amount_range(df: pd.DataFrame) -> bool:
    if df["TransactionAmt"].min() < 0:
        logger.error(f"Negative amounts found")
        return False
    logger.info(
        f"✅ Amount range check passed: "
        f"${df['TransactionAmt'].min():.2f} "
        f"to ${df['TransactionAmt'].max():.2f}"
    )
    return True


def check_high_null_features(df: pd.DataFrame) -> bool:
    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > MAX_NULL_THRESHOLD]
    logger.info(
        f"Features above {MAX_NULL_THRESHOLD:.0%} null threshold: "
        f"{len(high_null)} — will be dropped in preprocessing"
    )
    # This is a warning not a failure — we handle it in preprocess
    return True


def check_time_ordering(df: pd.DataFrame) -> bool:
    if not df["TransactionDT"].is_monotonic_increasing:
        logger.warning(
            "TransactionDT is not monotonically increasing — "
            "data may not be sorted. Sorting before temporal split."
        )
    else:
        logger.info("✅ Time ordering check passed")
    return True


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Runs all validation checks.
    Collects all failures before raising.
    """
    logger.info("Running PayGuard dataset validation...")

    checks = {
        "required_columns": check_required_columns,
        "key_column_nulls": check_no_nulls_in_key_columns,
        "class_values": check_class_values,
        "fraud_rate": check_fraud_rate,
        "amount_range": check_amount_range,
        "high_null_features": check_high_null_features,
        "time_ordering": check_time_ordering,
    }

    failures = []
    for name, fn in checks.items():
        if not fn(df):
            failures.append(name)

    if failures:
        raise ValueError(
            f"Validation failed. Failed checks: {failures}"
        )

    logger.info("✅ All validation checks passed")
    return True


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/raw_merged.parquet")
    validate_dataset(df)