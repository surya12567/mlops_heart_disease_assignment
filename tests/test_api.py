from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint_returns_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_rejects_missing_features_when_model_available(monkeypatch):
    # This unit test avoids requiring a trained artifact and tests validation explicitly.
    import src.api as api

    class DummyModel:
        classes_ = [0, 1]
        def predict(self, x):
            return [0]
        def predict_proba(self, x):
            return [[0.7, 0.3]]

    monkeypatch.setattr(api, "get_model_and_metadata", lambda: (DummyModel(), {
        "feature_names": ["age", "sex"], "selected_model": "dummy"
    }))
    response = client.post("/predict", json={"features": {"age": 52}})
    assert response.status_code == 422
