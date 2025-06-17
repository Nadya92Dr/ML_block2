import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn


from mlflow.tracking import MlflowClient


app = FastAPI(title="Wine Quality Prediction API")


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


model = None
model_info_data = {}


def load_model():

    global model, model_info_data
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
    run_id = os.getenv("MLFLOW_RUN_ID", "f8d61fe1e48847b89207b2c59163af32")

    client = MlflowClient(tracking_uri=tracking_uri)
    dst = "models/mlflow_model"
    os.makedirs(dst, exist_ok=True)
    client.download_artifacts(run_id, artifact_path="model", dst_path=dst)

    model_path = os.path.join(dst, "model.pkl")
    if not os.path.exists(model_path):

        raise RuntimeError(f"Не найден файл модели по пути {model_path}")
    model = joblib.load(model_path)

    test_data = pd.DataFrame(
        [
            {
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
        ]
    )
    prediction = model.predict(test_data)[0]
    print("Модель успешно загружена. Тестовое предсказание:", prediction)

    model_info_data = {
        "model_name": "Wine Quality Prediction Model",
        "version": "latest",
        "test_prediction": float(prediction),
        "metrics": {"accuracy": None, "r2_score": None},
    }


@app.on_event("startup")
def startup_event():
    load_model()


@app.post("/predict")
def predict(wine: WineFeatures) -> dict:
    """
    Принимает входные данные о вине, проводит валидацию через Pydantic и возвращает предсказание.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    try:

        input_data = pd.DataFrame([wine.dict()])
        prediction = model.predict(input_data)[0]
        return {"predicted_quality": float(prediction)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка при получении предсказания: {str(e)}"
        )


@app.get("/health")
def health() -> dict:
    """
    Проверяет работоспособность сервиса и модели путём выполнения тестового предсказания.
    """
    if model is None:
        return {"status": "error", "reason": "Модель не загружена"}
    try:
        test_data = pd.DataFrame(
            [
                {
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
            ]
        )
        _ = model.predict(test_data)[0]
        return {"status": "ok"}
    except Exception:
        return {"status": "error", "reason": "Ошибка при выполнении предсказания"}


@app.get("/model-info")
def model_info() -> dict:
    """
    Возвращает информацию о модели и её метриках.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return model_info_data


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
