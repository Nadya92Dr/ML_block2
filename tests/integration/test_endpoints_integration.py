from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_ok():

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_structure():

    response = client.get("/model-info")
    assert response.status_code == 200

    json_data = response.json()

    assert "model_name" in json_data
    assert "version" in json_data
    assert "test_prediction" in json_data
    assert isinstance(json_data["metrics"], dict)


def test_predict_valid_input():

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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert "predicted_quality" in result

    assert isinstance(result["predicted_quality"], float)
