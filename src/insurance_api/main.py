"""FastAPI app: explainable cross-sell predictions.

Run locally:
    uvicorn insurance_api.main:app --reload
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from .model import Predictor
from .schema import BatchPrediction, BatchRequest, Customer, ModelInfo, Prediction

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("insurance_api")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        state["predictor"] = Predictor(MODEL_PATH)
        log.info("Loaded model %s", state["predictor"].metadata["model_version"])
    else:
        log.warning("No model at %s; /predict will return 503 until one is trained", MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="Explainable Insurance Cross-Sell API",
    description="Predicts whether a health-insurance customer is interested in vehicle insurance, "
    "and explains each prediction with SHAP values.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code,
             (time.perf_counter() - start) * 1000)
    return response


def predictor() -> Predictor:
    if "predictor" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state["predictor"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> dict:
    return predictor().metadata


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> dict:
    return predictor().predict([customer.model_dump()])[0]


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(request: BatchRequest) -> dict:
    return {"predictions": predictor().predict([c.model_dump() for c in request.customers])}
