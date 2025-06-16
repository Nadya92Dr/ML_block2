from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_predict_missing_field_raises_422():

    incomplete_payload = {
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
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_invalid_type_raises_422():

    bad_payload = {
        "fixed_acidity": "should_be_float",
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
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_health_when_model_not_loaded(monkeypatch):

    monkeypatch.setattr("api.model", None)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert "Модель не загружена" in response.json()["reason"]
