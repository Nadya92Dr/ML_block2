import mlflow

import os
from joblib import dump

os.makedirs("models", exist_ok=True)
mlflow.set_tracking_uri("http://localhost:5000")
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np

from config import config
from data import get_data


def train(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def test(model, x_test, y_test):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    try:
        y_prob = model.predict_proba(x_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
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
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    with mlflow.start_run(run_name="LogisticRegression_Experiment"):
        logistic_regression_model = LogisticRegression(
            max_iter=config["logistic_regression"]["max_iter"],
            penalty=config["logistic_regression"].get("penalty", "l2"),
            random_state=config["random_state"],
        )
        trained_model = train(...)
        metrics = test(...)

        mlflow.sklearn.log_model(trained_model, "model")

        data = get_data()
        trained_model = train(
            logistic_regression_model, data["x_train"], data["y_train"]
        )
        metrics = test(trained_model, data["x_test"], data["y_test"])
        mlflow.sklearn.log_model(trained_model, "model")

        os.makedirs("models", exist_ok=True)
        dump(trained_model, "models/wine_quality_model.pkl")
