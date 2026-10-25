import numpy as np
from fastapi.testclient import TestClient

import app.main as main


class FakeVersion:
    version = "7"
    run_id = "run-123"


class FakeClient:
    def __init__(self, tracking_uri=None):
        self.tracking_uri = tracking_uri

    def get_model_version_by_alias(self, name, alias):
        assert name == "BrentDirectionModel"
        assert alias == "champion"
        return FakeVersion()


class FakeModel:
    classes_ = np.array([0, 1])

    def predict(self, features):
        assert len(features) == 1
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.25, 0.75]])


def sample_history():
    closes = [80.0, 80.4, 80.1, 80.8, 81.0, 80.7, 81.2, 81.4, 81.1, 81.6, 82.0]
    return {
        "candles": [
            {
                "date": f"2024-01-{index + 1:02d}",
                "open": close,
                "high": close + 0.3,
                "low": close - 0.3,
                "close": close,
                "volume": 100,
            }
            for index, close in enumerate(closes)
        ]
    }


def configure_fake_model(monkeypatch, load_model):
    monkeypatch.setattr(main, "MlflowClient", FakeClient)
    monkeypatch.setattr(main.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(main.mlflow.sklearn, "load_model", load_model)


def test_process_uses_loaded_model_and_reports_version(monkeypatch):
    load_calls = []

    def load_model(uri):
        load_calls.append(uri)
        return FakeModel()

    configure_fake_model(monkeypatch, load_model)

    with TestClient(main.app) as client:
        first = client.post("/process", json=sample_history())
        second = client.post("/process", json=sample_history())
        model_response = client.get("/model")
        health_response = client.get("/healthz")

    assert first.status_code == 200
    assert first.json()["prediction"] == 1
    assert first.json()["direction"] == "up"
    assert first.json()["probability_up"] == 0.75
    assert first.json()["model_version"] == "7"
    assert first.json()["history_rows"] == 11
    assert second.status_code == 200
    assert len(load_calls) == 1
    assert load_calls == ["models:/BrentDirectionModel/7"]
    assert model_response.json()["run_id"] == "run-123"
    assert health_response.json()["status"] == "ok"


def test_process_rejects_invalid_candle(monkeypatch):
    configure_fake_model(monkeypatch, lambda uri: FakeModel())
    payload = sample_history()
    payload["candles"][0]["high"] = payload["candles"][0]["close"] - 1

    with TestClient(main.app) as client:
        response = client.post("/process", json=payload)

    assert response.status_code == 422


def test_process_requires_enough_history(monkeypatch):
    configure_fake_model(monkeypatch, lambda uri: FakeModel())
    payload = sample_history()
    payload["candles"] = payload["candles"][:10]

    with TestClient(main.app) as client:
        response = client.post("/process", json=payload)

    assert response.status_code == 422


def test_process_rejects_mixed_date_presence(monkeypatch):
    configure_fake_model(monkeypatch, lambda uri: FakeModel())
    payload = sample_history()
    payload["candles"][0].pop("date")

    with TestClient(main.app) as client:
        response = client.post("/process", json=payload)

    assert response.status_code == 422
