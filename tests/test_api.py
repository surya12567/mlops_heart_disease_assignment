
"""Tests for the Heart Disease Classification API."""

from fastapi.testclient import TestClient

from src import api

client = TestClient(api.app)


def test_health_endpoint_returns_status(monkeypatch):
    """Health check should succeed when the model loader succeeds."""
    monkeypatch.setattr(
        api,
        "get_model_and_metadata",
        lambda: (
            object(),
            {
                "feature_names": ["age"],
                "selected_model": "LogisticRegression",
            },
        ),
    )

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_available"] is True
    assert body["selected_model"] == "LogisticRegression"


def test_health_endpoint_returns_503_when_model_unavailable(monkeypatch):
    """Health check should report unavailable model files."""
    def raise_model_error():
        raise RuntimeError("Model files are missing.")

    monkeypatch.setattr(
        api,
        "get_model_and_metadata",
        raise_model_error,
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not ready."


def test_predict_rejects_missing_features(monkeypatch):
    """Prediction should reject incomplete feature input."""
    monkeypatch.setattr(
        api,
        "get_model_and_metadata",
        lambda: (
            object(),
            {
                "feature_names": ["age", "sex"],
                "selected_model": "LogisticRegression",
            },
        ),
    )

    response = client.post(
        "/predict",
        json={"features": {"age": 55}},
    )

    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]


def test_predict_rejects_unexpected_features(monkeypatch):
    """Prediction should reject unexpected feature names."""
    monkeypatch.setattr(
        api,
        "get_model_and_metadata",
        lambda: (
            object(),
            {
                "feature_names": ["age"],
                "selected_model": "LogisticRegression",
            },
        ),
    )

    response = client.post(
        "/predict",
        json={"features": {"age": 55, "unknown_feature": 1}},
    )

    assert response.status_code == 422
    assert "unexpected_features" in response.json()["detail"]


def test_metrics_endpoint_returns_prometheus_metrics():
    """Metrics endpoint should respond successfully."""
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "prediction_requests_total" in response.text
    assert "prediction_request_duration_seconds" in response.text
