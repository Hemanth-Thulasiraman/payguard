# PayGuard — Real-Time Transaction Fraud Detection

PayGuard is a production-grade ML system that scores financial 
transactions for fraud probability in real time using LightGBM 
trained on the IEEE-CIS Fraud Detection dataset (590,540 transactions).

**Results:** AUC-ROC 0.888 | AUC-PR 0.514 | Recall 71.8% on 
held-out test set of 88,581 transactions.

---

## System Architecture

Transaction (JSON)
↓
FastAPI /score endpoint
↓
Feature engineering (amount, time, categoricals)
↓
Category mapping (saved from training — no skew)
↓
LightGBM (500 trees, scale_pos_weight=27)
↓
Fraud probability + risk level returned

---

## Results

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.888 |
| AUC-PR | 0.514 |
| Recall @ threshold | 71.8% |
| Precision @ threshold | 18.0% |
| Fraud caught | 2,213 / 3,083 |
| Fraud missed | 870 |
| False alarms | 10,057 |
| Threshold | 0.145 |

**Business impact:** At the operating threshold the model prevents 
approximately $330,000 in fraudulent transactions per test period 
while generating 10,057 alerts requiring investigation. The 870 
missed fraud cases represent ~$130,000 in undetected losses.

**Baseline comparison:** Random model AUC-PR = 0.035 (fraud rate). 
PayGuard is 14.7x better than random.

---

## Key Engineering Decisions

**Temporal split over random split.**
All splits are ordered by TransactionDT. Random splitting causes 
temporal leakage — the model trains on future fraud patterns and 
is tested on past ones, producing optimistic metrics that collapse 
in production. We train on the first 70%, validate on the next 15%, 
and test on the final 15% in chronological order.

**Feature threshold at 95% null.**
433 features merged from transaction and identity tables. 9 features 
exceeded 95% null rate and were dropped. Remaining features with 
partial nulls are kept — LightGBM handles nulls natively by learning 
optimal split directions for missing values.

**Saved category mappings.**
Categorical features are encoded as integer codes during preprocessing. 
The mapping dictionary is saved to disk and loaded by the serving layer. 
This guarantees identical encoding between training and inference — 
eliminating training-serving skew on categorical features.

**class_pos_weight = 27.**
Dataset has 3.5% fraud rate — 578 legitimate transactions per fraud 
case. Without class weighting LightGBM predicts legitimate for 
everything and achieves 96.5% accuracy while catching zero fraud. 
Weight of 27 (legitimate/fraud ratio) tells the model each fraud 
example is 27x more costly to misclassify.

**Threshold selection targeting business recall.**
Default threshold of 0.5 catches very little fraud on imbalanced 
data. We select the threshold that achieves 80% recall on the 
validation set — prioritizing fraud caught over false alarm rate. 
The business tradeoff: higher recall costs more investigator hours 
per fraud caught.

---

## Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| Ingestion | `src/data/ingest.py` | Loads and merges transaction + identity |
| Validation | `src/data/validate.py` | Schema, null, fraud rate checks |
| Preprocessing | `src/data/preprocess.py` | Feature engineering, encoding, splits |
| Training | `src/training/train.py` | LightGBM with class weighting |
| Evaluation | `src/evaluation/evaluate.py` | AUC-PR, recall, business metrics |
| Serving | `src/serving/api.py` | FastAPI real-time scoring |
| Monitoring | `src/monitoring/` | Drift and latency tracking |

---

## Quickstart

```bash
git clone https://github.com/Hemanth-Thulasiraman/payguard.git
cd payguard
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download IEEE-CIS dataset from Kaggle and place in data/raw/
# train_transaction.csv and train_identity.csv

python -m src.data.ingest
python -m src.data.preprocess
python -m src.training.train
python -m src.evaluation.evaluate

uvicorn src.serving.api:app --port 8000
```

---

## API Usage

**Health check:**
```bash
curl http://localhost:8000/health
```

**Score a transaction:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 299.99,
    "ProductCD": "W",
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com"
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.2959,
  "is_fraud": false,
  "risk_level": "LOW",
  "threshold_used": 0.1433
}
```

---

## Dataset

IEEE-CIS Fraud Detection — Kaggle competition dataset.

| Stat | Value |
|------|-------|
| Total transactions | 590,540 |
| Fraud cases | 20,663 (3.50%) |
| Legitimate | 569,877 (96.50%) |
| Raw features | 433 |
| After null filtering | 425 |
| After engineering | 432 |
| Identity coverage | 24.4% |

---

## Retrospective

**What worked well:**
- LightGBM handled 433 features and missing identity data cleanly
- Temporal split prevented leakage and gave realistic evaluation
- Category mapping saved at preprocessing time eliminated inference skew
- scale_pos_weight correctly shifted model focus to minority class

**What I would do differently:**
- Implement proper early stopping using AUC-PR not binary logloss
- Build a feature store for velocity features (transactions per hour per card)
- Add SHAP values for per-prediction explainability
- Save category mappings inside the model pickle to reduce deployment complexity

**Version 2 improvements:**
- Velocity features: transactions per hour, rolling average amount per card
- SHAP explainability endpoint: why was this transaction flagged?
- Retraining pipeline triggered by drift alerts
- A/B testing framework for threshold optimization
- Proper early stopping with raw LightGBM API