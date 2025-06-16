from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
import joblib
import os
import subprocess

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
    model_path = "models/wine_quality_model.pkl"

    try:
        subprocess.run(["dvc", "pull", model_path], check=True)
    except Exception as e:
        print("Ошибка при выполнении dvc pull:", e)

    if not os.path.exists(model_path):
        print(f"Файл модели не найден по пути {model_path}")
        model = None
        return

    try:
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
    except Exception as e:
        print("Ошибка при загрузке модели:", e)
        model = None


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
