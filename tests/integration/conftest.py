import pytest
from api import app


class DummyModel:
    def predict(self, df):
        return [5]


@pytest.fixture(autouse=True)
def inject_dummy_model(monkeypatch):
    import api

    monkeypatch.setattr(api, "model", DummyModel())
    monkeypatch.setattr(
        api,
        "model_info_data",
        {
            "model_name": "DummyModel",
            "version": "test",
            "test_prediction": 5.0,
            "metrics": {"accuracy": None, "r2_score": None},
        },
    )
