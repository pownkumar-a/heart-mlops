"""FastAPI service that serves the heart-disease model.

Endpoints:
  GET  /         - basic info
  GET  /health   - is the API up and the model loaded?
  POST /predict  - predict heart-disease risk for one patient
  GET  /metrics  - Prometheus metrics (used for monitoring in Task 8)
"""
import logging
import time
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict

from src.predict import SAMPLE_PATIENT, load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")

app = FastAPI(title="Heart Disease Risk API", version="1.0")
model = load_model()

REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions made, by label", ["label"])


class Patient(BaseModel):
    """The 13 clinical features of one patient."""
    model_config = ConfigDict(json_schema_extra={"example": SAMPLE_PATIENT})

    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: Optional[float] = None
    thal: Optional[float] = None


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    if path != "/metrics":
        REQUESTS.labels(path, str(response.status_code)).inc()
        LATENCY.labels(path).observe(elapsed)
    log.info("%s %s -> %s (%.1f ms)", request.method, path,
             response.status_code, elapsed * 1000)
    return response


@app.get("/")
def root():
    return {"message": "Heart Disease Risk API. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict_endpoint(patient: Patient):
    result = predict([patient.model_dump()], model=model)[0]
    PREDICTIONS.labels(result["label"]).inc()
    log.info("prediction=%s confidence=%s", result["label"], result["confidence"])
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
