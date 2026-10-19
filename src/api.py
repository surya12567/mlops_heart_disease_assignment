"""FastAPI inference service with request logging and Prometheus metrics."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-mlops-api")
REQUESTS = Counter("prediction_requests_total", "Prediction requests", ["endpoint", "status"])
LATENCY = Histogram("prediction_request_duration_seconds", "Request duration", ["endpoint"])

app = FastAPI(title="Heart Disease Classification API", version="1.0.0",
              description="Educational demo only; not for medical diagnosis.")
_model = None
_metadata = None


def get_model_and_metadata():
    global _model, _metadata
    if _model is None:
        if not MODEL_PATH.exists() or not META_PATH.exists():
            raise RuntimeError("Model files missing. Run `python -m src.data --download` and `python -m src.train`.")
        _model = joblib.load(MODEL_PATH)
        _metadata = json.loads(META_PATH.read_text())
    return _model, _metadata


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Feature name/value pairs matching model metadata")


@app.get("/health")
def health():
    ready = MODEL_PATH.exists() and META_PATH.exists()
    return {"status": "ok" if ready else "model_not_trained", "model_available": ready}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(request: PredictionRequest):
    started = time.perf_counter()
    status = "success"
    try:
        model, metadata = get_model_and_metadata()
        feature_names = metadata["feature_names"]
        missing = [name for name in feature_names if name not in request.features]
        extra = [name for name in request.features if name not in feature_names]
        if missing:
            raise HTTPException(status_code=422, detail={"missing_features": missing, "expected_features": feature_names})
        if extra:
            logger.warning("Ignoring unexpected input features: %s", extra)
        row = pd.DataFrame([{name: request.features[name] for name in feature_names}], columns=feature_names)
        row = row.apply(pd.to_numeric, errors="coerce")
        if row.isna().any().any():
            raise HTTPException(status_code=422, detail="Feature values must be numeric and non-null.")
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0].tolist()
        logger.info("prediction_request features=%s prediction=%s", row.to_dict(orient="records")[0], prediction)
        return {
            "prediction": prediction,
            "label": "presence_of_disease" if prediction == 1 else "absence_of_disease",
            "probabilities": {str(cls): float(prob) for cls, prob in zip(model.classes_, probabilities)},
            "model": metadata["selected_model"],
            "disclaimer": "Educational model only; not medical advice or a diagnostic result.",
        }
    except HTTPException:
        status = "client_error"
        raise
    except Exception:
        status = "server_error"
        logger.exception("Prediction failed")
        raise HTTPException(status_code=503, detail="Model unavailable or prediction failed.")
    finally:
        REQUESTS.labels(endpoint="/predict", status=status).inc()
        LATENCY.labels(endpoint="/predict").observe(time.perf_counter() - started)
