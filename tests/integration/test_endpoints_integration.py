import pytest
from fastapi.testclient import TestClient
import api.api as api_mod
from api import app


class DummyModel:
    def predict(self, df):
        return [42]


@pytest.fixture(autouse=True)
def inject_dummy_model(monkeypatch):
    monkeypatch.setattr(api_mod, "load_model", lambda: None)
    monkeypatch.setattr(api_mod, "model", DummyModel())
    monkeypatch.setattr(
        api_mod,
        "model_info_data",
        {
            "model_name": "DummyModel",
            "version": "test",
            "test_prediction": 42.0,
            "metrics": {"accuracy": None, "r2_score": None},
        },
    )


@pytest.fixture
def client():
    return TestClient(app)


def test_health_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_model_info(client):
    res = client.get("/model-info")
    assert res.status_code == 200
    body = res.json()
    assert body["model_name"] == "DummyModel"
    assert body["version"] == "test"
    assert body["test_prediction"] == 42.0


def test_predict(client):
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert res.json() == {"predicted_quality": 42.0}
