# src/serving/api.py

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from loguru import logger

from src.serving.model import (
    load_model,
    score_transaction,
    is_model_loaded,
    get_threshold,
    get_n_features,
)
from src.serving.schemas import (
    TransactionRequest,
    FraudScoreResponse,
    HealthResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PayGuard API...")
    load_model()
    logger.info("API ready")
    yield
    logger.info("Shutting down PayGuard API")


app = FastAPI(
    title="PayGuard Fraud Detection API",
    description=(
        "Real-time transaction fraud scoring using "
        "LightGBM trained on IEEE-CIS fraud detection dataset"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if is_model_loaded() else "unhealthy",
        model_loaded=is_model_loaded(),
        threshold=get_threshold(),
        n_features=get_n_features(),
    )


@app.post("/score", response_model=FraudScoreResponse)
def score(request: TransactionRequest):
    """
    Scores a transaction for fraud probability.
    Returns probability, binary classification, and risk level.
    Latency target: < 100ms.
    """
    start = time.time()

    try:
        result = score_transaction(request.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start) * 1000
    logger.info(
        f"Scored transaction: prob={result['fraud_probability']:.4f} "
        f"risk={result['risk_level']} "
        f"latency={latency_ms:.1f}ms"
    )

    return FraudScoreResponse(**result)