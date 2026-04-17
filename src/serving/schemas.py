# src/serving/schemas.py

from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """
    Incoming transaction for fraud scoring.
    Subset of features available at inference time.
    """
    TransactionAmt: float = Field(
        ...,
        gt=0,
        description="Transaction amount in USD"
    )
    ProductCD: str = Field(
        ...,
        description="Product code"
    )
    card4: str = Field(
        default="unknown",
        description="Card network (visa, mastercard, etc)"
    )
    card6: str = Field(
        default="unknown",
        description="Card type (debit, credit)"
    )
    P_emaildomain: str = Field(
        default="unknown",
        description="Purchaser email domain"
    )
    TransactionDT: float = Field(
        default=86400.0,
        description="Transaction timestamp offset in seconds"
    )

    @field_validator("TransactionAmt")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Transaction amount must be positive")
        return v


class FraudScoreResponse(BaseModel):
    """Fraud score and classification for a transaction."""
    fraud_probability: float = Field(
        description="Model's estimated probability of fraud (0-1)"
    )
    is_fraud: bool = Field(
        description="Classification at operating threshold"
    )
    risk_level: str = Field(
        description="HIGH, MEDIUM, or LOW risk"
    )
    threshold_used: float = Field(
        description="Classification threshold applied"
    )


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    threshold: float
    n_features: int