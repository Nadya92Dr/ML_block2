from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np

from data import get_data
from config import config


def load_data():

    data = get_data()
    print("Данные о вине успешно загружены.")
    return data


def train_model(**kwargs):

    ti = kwargs["ti"]

    data = ti.xcom_pull(task_ids="load_data_task")
    if not data:
        data = get_data()

    with mlflow.start_run(run_name="LogisticRegression_Experiment"):

        model = LogisticRegression(
            max_iter=config["logistic_regression"]["max_iter"],
            penalty=config["logistic_regression"].get("penalty", "l2"),
            random_state=config["random_state"],
        )

        model.fit(data["x_train"], data["y_train"])
        y_pred = model.predict(data["x_test"])

        accuracy = accuracy_score(data["y_test"], y_pred)
        f1 = f1_score(data["y_test"], y_pred, average="weighted")
        cm = confusion_matrix(data["y_test"], y_pred)
        try:
            y_prob = model.predict_proba(data["x_test"])
            auc = roc_auc_score(
                data["y_test"], y_prob, multi_class="ovr", average="weighted"
            )
        except Exception:
            auc = None

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        if auc is not None:
            mlflow.log_metric("auc_roc", auc)

        mlflow.log_param("max_iter", config["logistic_regression"]["max_iter"])
        mlflow.log_param("random_state", config["random_state"])
        if hasattr(model, "coef_"):
            coefficients = model.coef_.tolist()
            mlflow.log_param("coefficients", coefficients)
        regularization = getattr(
            model, "penalty", config["logistic_regression"].get("penalty", "l2")
        )
        mlflow.log_param("regularization", regularization)

        np.savetxt("confusion_matrix.txt", cm, fmt="%d")
        mlflow.log_artifact("confusion_matrix.txt")
        print("Accuracy:", accuracy)

        mlflow.sklearn.log_model(model, "model")

    return {"accuracy": accuracy, "f1_score": f1, "auc_roc": auc}


def save_model(**kwargs):

    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="train_model_task")
    if metrics:
        print(f"Эксперимент завершён. Метрики: {metrics}")
    else:
        print("Модель не была обучена.")


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "wine_model_pipeline",
    default_args=default_args,
    description="Пайплайн для обучения модели качества вина с использованием MLflow",
    schedule_interval="@daily",  # Запуск ежедневно
    catchup=False,
)

load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id="train_model_task",
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

save_model_task = PythonOperator(
    task_id="save_model_task",
    python_callable=save_model,
    provide_context=True,
    dag=dag,
)

load_data_task >> train_model_task >> save_model_task
