# Explainable Insurance Cross-Sell API

## Live Link: https://insurance-cross-sell-prediction.onrender.com/docs

A production-style REST API that predicts whether a health-insurance customer will buy vehicle insurance, and explains **every single prediction** with SHAP values. Built for regulated industries, where a model has to justify its decisions to customers and auditors.

**Stack:** Python · XGBoost · scikit-learn · FastAPI · Pydantic · Docker · GitHub Actions

```
data/train.csv ──► train.py ──► models/model.joblib ──► FastAPI ──► /predict
                   (preprocess, XGBoost,                 (validation,  { probability,
                    F1-tuned threshold)                   logging)       top_reasons[SHAP] }
```

## Results

Trained on the [Kaggle Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) data, held-out test set:

| Metric | Value |
| --- | --- |
| F1 (positive class) | 0.4576|
| ROC AUC | 0.8572 |
| PR AUC | 0.3672 |


## Example

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "Gender": "Male", "Age": 44, "Driving_License": 1, "Region_Code": 28,
  "Previously_Insured": 0, "Vehicle_Age": "> 2 Years", "Vehicle_Damage": "Yes",
  "Annual_Premium": 40454, "Policy_Sales_Channel": 26, "Vintage": 217 }'
```

```json
{
  "probability": 0.83,
  "interested": true,
  "threshold": 0.32,
  "top_reasons": [
    {"feature": "Previously_Insured", "value": "0",   "shap_value": 1.05},
    {"feature": "Vehicle_Damage",     "value": "Yes", "shap_value": 0.38},
    {"feature": "Age",                "value": "44",  "shap_value": 0.38}
  ]
}
```

Positive SHAP values push towards "interested", negative ones away from it (log-odds scale).

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | One customer, with explanation |
| `POST /predict/batch` | Up to 1,000 customers |
| `GET /model-info` | Model version, threshold, test metrics |
| `GET /health` | Liveness check |
| `GET /docs` | Interactive Swagger UI |

## Design decisions

- **Threshold tuned for F1 on a validation split**, not the default 0.5. The positive class is rare (~12–16%), so 0.5 misses most interested customers. Metrics are reported on a separate test split.
- **Exact TreeSHAP from XGBoost** (`pred_contribs=True`) instead of the `shap` package: same values, no extra dependency, a smaller image, and fast enough for real-time requests.
- **One explanation per business feature:** one-hot columns (e.g. `Vehicle_Age_< 1 Year`) are summed back into `Vehicle_Age`, so the explanation reads the way a business user thinks.
- **Rare and unseen categories are grouped** (`min_frequency`, `handle_unknown="infrequent_if_exist"`), so a new region code does not crash the API.
- **Strict input validation** with Pydantic: an invalid age or an unknown category returns HTTP 422 with a clear message.
- **Model trained during the Docker build:** the image always contains a model that matches its code versions.

## Run it

```bash
make install                        # dependencies
# put Kaggle's train.csv into data/  (or: make sample  for synthetic data)
make train                          # trains and prints metrics
make serve                          # http://localhost:8000/docs
make test                           # 13 tests, using synthetic data
make docker                         # build and run the container
```

## CI

On every push, GitHub Actions lints with ruff, runs the tests, builds the Docker image and sends a real prediction to the running container.

## Deploy (free)

1. Push this repo to GitHub.
2. On [Render](https://render.com), create a **Web Service** from the repo; it detects the Dockerfile.
3. Use `/health` as the health-check path.

Without `data/train.csv` in the repo, the build trains on synthetic data, which is fine for a demo. For real numbers, train locally and describe the results here.

## Project structure

```
src/insurance_api/
  features.py   shared preprocessing (training and serving use the same code)
  train.py      training, threshold tuning, evaluation, optional MLflow logging
  model.py      loads the artifact, predicts, builds SHAP explanations
  schema.py     Pydantic request/response models
  main.py       FastAPI app
scripts/make_sample_data.py   synthetic data with the Kaggle schema (for tests and CI)
tests/                        API and feature tests
```
