
"""FastAPI inference service with request logging and Prometheus metrics."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart-mlops-api")

REQUESTS = Counter(
    "prediction_requests_total",
    "Number of prediction API requests",
    ["endpoint", "status"],
)

LATENCY = Histogram(
    "prediction_request_duration_seconds",
    "Prediction API request duration in seconds",
    ["endpoint"],
)

app = FastAPI(
    title="Heart Disease Classification API",
    version="1.0.0",
    description=(
        "Educational machine learning demonstration. "
        "Not intended for medical diagnosis."
    ),
)

_model = None
_metadata = None


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def get_model_and_metadata():
    """Load the trained pipeline and metadata once."""
    global _model, _metadata

    if _model is None or _metadata is None:
        if not MODEL_PATH.exists() or not META_PATH.exists():
            raise RuntimeError(
                "Model files are missing. Run "
                "'python -m src.data --download' and "
                "'python -m src.train' before starting the API."
            )

        try:
            model = joblib.load(MODEL_PATH)
            metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

            feature_names = metadata.get("feature_names")
            if (
                not isinstance(feature_names, list)
                or not feature_names
                or not all(isinstance(name, str) for name in feature_names)
            ):
                raise ValueError(
                    "model_metadata.json must contain a non-empty "
                    "feature_names list."
                )

            if not hasattr(model, "predict"):
                raise ValueError("Loaded model does not support prediction.")

            if not hasattr(model, "predict_proba"):
                raise ValueError(
                    "Loaded model does not support probability predictions."
                )

            _model = model
            _metadata = metadata

        except Exception as exc:
            logger.exception("Could not load the trained model")
            raise RuntimeError(
                "Could not load model or metadata."
            ) from exc

    return _model, _metadata


# ---------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Feature name/value pairs required by the trained model."""

    features: dict[str, Any] = Field(
        ...,
        description=(
            "Feature name/value pairs matching model_metadata.json."
        ),
    )


# ---------------------------------------------------------------------
# Health and monitoring endpoints
# ---------------------------------------------------------------------

@app.get("/")
def root():
    """Return basic API information."""
    return {
        "message": "Heart Disease Classification API",
        "docs": "/docs",
        "health": "/health",
        "prediction": "/predict",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    """Check whether the model can be loaded and used."""
    try:
        _, metadata = get_model_and_metadata()

        return {
            "status": "ok",
            "model_available": True,
            "selected_model": metadata.get(
                "selected_model", "unknown"
            ),
        }

    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        ) from exc


@app.get("/metrics")
def metrics():
    """Expose Prometheus-compatible metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):
    """Predict the class and return class probabilities."""
    started = time.perf_counter()
    status = "success"

    try:
        model, metadata = get_model_and_metadata()
        feature_names = metadata["feature_names"]
        supplied_features = request.features

        # Validate feature names.
        missing = [
            name for name in feature_names
            if name not in supplied_features
        ]
        extra = [
            name for name in supplied_features
            if name not in feature_names
        ]

        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Required features are missing.",
                    "missing_features": missing,
                    "expected_features": feature_names,
                },
            )

        if extra:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Unexpected features supplied.",
                    "unexpected_features": extra,
                },
            )

        # Build the row in the exact order expected by the model.
        row = pd.DataFrame(
            [{name: supplied_features[name] for name in feature_names}],
            columns=feature_names,
        )

        # Reject nonnumeric, null, NaN and infinite values.
        for name in feature_names:
            value = row.iloc[0][name]

            if isinstance(value, bool):
                raise HTTPException(
                    status_code=422,
                    detail=f"Feature '{name}' must be numeric, not boolean.",
                )

        row = row.apply(pd.to_numeric, errors="coerce")

        if row.isna().any().any():
            raise HTTPException(
                status_code=422,
                detail="All feature values must be numeric and non-null.",
            )

        if not all(
            math.isfinite(float(value))
            for value in row.iloc[0].tolist()
        ):
            raise HTTPException(
                status_code=422,
                detail="Feature values must be finite numbers.",
            )

        # Predict class and probabilities.
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0].tolist()
        classes = list(model.classes_)

        probability_by_class = {
            str(cls): float(probability)
            for cls, probability in zip(classes, probabilities)
        }

        # The dataset is encoded as 0 = no disease, 1 = disease.
        disease_probability = probability_by_class.get("1")

        logger.info(
            "Prediction completed: class=%s model=%s",
            prediction,
            metadata.get("selected_model", "unknown"),
        )

        return {
            "prediction": prediction,
            "label": (
                "presence_of_disease"
                if prediction == 1
                else "absence_of_disease"
            ),
            "probabilities": probability_by_class,
            "disease_probability": disease_probability,
            "confidence_percent": (
                round(disease_probability * 100, 2)
                if disease_probability is not None
                else None
            ),
            "model": metadata.get("selected_model", "unknown"),
            "disclaimer": (
                "Educational model only; not medical advice "
                "or a diagnostic result."
            ),
        }

    except HTTPException:
        status = "client_error"
        raise

    except Exception as exc:
        status = "server_error"
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=503,
            detail="Model unavailable or prediction failed.",
        ) from exc

    finally:
        REQUESTS.labels(
            endpoint="/predict",
            status=status,
        ).inc()

        LATENCY.labels(
            endpoint="/predict"
        ).observe(time.perf_counter() - started)
