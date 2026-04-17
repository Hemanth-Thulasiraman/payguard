# src/data/ingest.py

import pandas as pd
from loguru import logger
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TRANSACTION_PATH = RAW_DIR / "train_transaction.csv"
IDENTITY_PATH = RAW_DIR / "train_identity.csv"


def load_transactions() -> pd.DataFrame:
    """
    Loads raw transaction data.
    652MB CSV — this takes 30-60 seconds.
    """
    logger.info(f"Loading transactions from {TRANSACTION_PATH}")
    df = pd.read_csv(TRANSACTION_PATH)
    logger.info(f"Transactions loaded: {len(df):,} rows, {df.shape[1]} columns")
    return df


def load_identity() -> pd.DataFrame:
    """
    Loads identity data.
    Not every transaction has an identity record.
    """
    logger.info(f"Loading identity from {IDENTITY_PATH}")
    df = pd.read_csv(IDENTITY_PATH)
    logger.info(f"Identity loaded: {len(df):,} rows, {df.shape[1]} columns")
    return df


def merge_datasets(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left joins transactions with identity on TransactionID.
    Left join preserves all transactions — identity is sparse.
    Transactions without identity get NaN in identity columns.
    """
    logger.info("Merging transactions with identity...")
    df = transactions.merge(identity, on="TransactionID", how="left")

    identity_coverage = identity["TransactionID"].nunique()
    coverage_pct = identity_coverage / len(transactions) * 100

    logger.info(f"Merged shape: {df.shape}")
    logger.info(
        f"Identity coverage: {identity_coverage:,}/{len(transactions):,} "
        f"transactions ({coverage_pct:.1f}%)"
    )
    logger.info(
        f"Transactions WITHOUT identity: "
        f"{len(transactions) - identity_coverage:,} "
        f"({100-coverage_pct:.1f}%)"
    )
    return df


def profile_dataset(df: pd.DataFrame) -> None:
    """
    Logs key dataset statistics.
    """
    total = len(df)
    fraud = df["isFraud"].sum()
    legitimate = total - fraud
    fraud_rate = fraud / total * 100

    # Missing value analysis
    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 50]

    logger.info("=" * 50)
    logger.info("PayGuard Dataset Profile")
    logger.info("=" * 50)
    logger.info(f"Total transactions:  {total:,}")
    logger.info(f"Legitimate:          {legitimate:,} ({100-fraud_rate:.2f}%)")
    logger.info(f"Fraud:               {fraud:,} ({fraud_rate:.4f}%)")
    logger.info(f"Total features:      {df.shape[1]}")
    logger.info(
        f"Features >50% null:  {len(high_missing)} "
        f"— will review in validation"
    )
    logger.info(
        f"Mean fraud amount:   "
        f"${df[df['isFraud']==1]['TransactionAmt'].mean():.2f}"
    )
    logger.info(
        f"Mean legit amount:   "
        f"${df[df['isFraud']==0]['TransactionAmt'].mean():.2f}"
    )

    if fraud_rate > 10:
        logger.warning(
            f"Fraud rate {fraud_rate:.2f}% seems high — "
            f"verify dataset is not pre-balanced"
        )


def save_merged_parquet(df: pd.DataFrame) -> None:
    """
    Saves merged dataset as parquet for fast subsequent loads.
    Parquet is 5-10x faster to load than CSV.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "raw_merged.parquet"
    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Saved merged parquet: {output_path} ({size_mb:.0f}MB)")


def run_ingestion() -> pd.DataFrame:
    """
    Main entry point for data ingestion.
    Loads, merges, profiles, and saves transaction data.
    """
    logger.info("=" * 50)
    logger.info("Starting PayGuard data ingestion")
    logger.info("=" * 50)

    transactions = load_transactions()
    identity = load_identity()
    df = merge_datasets(transactions, identity)
    profile_dataset(df)
    save_merged_parquet(df)

    logger.info("Ingestion complete")
    return df


if __name__ == "__main__":
    run_ingestion()