import mlflow
import os
from joblib import dump

os.makedirs("models", exist_ok=True)
mlflow.set_tracking_uri("http://localhost:5000")

import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from data import get_data
from config import config
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    try:
        y_prob = model.predict_proba(x_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        auc = None
    return accuracy, f1, auc, cm


data = get_data()


for max_iter_value in [100, 200, 300]:
    with mlflow.start_run(run_name=f"LogReg_max_iter_{max_iter_value}"):
        model_lr = LogisticRegression(
            max_iter=max_iter_value,
            penalty=config["logistic_regression"].get("penalty", "l2"),
            random_state=config["random_state"],
        )
        trained_model = train_model(model_lr, data["x_train"], data["y_train"])
        accuracy, f1, auc, cm = evaluate_model(
            trained_model, data["x_test"], data["y_test"]
        )

        mlflow.log_param("max_iter", max_iter_value)
        mlflow.log_param("penalty", config["logistic_regression"].get("penalty", "l2"))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        if auc is not None:
            mlflow.log_metric("auc_roc", auc)

        if hasattr(trained_model, "coef_"):
            mlflow.log_param("coefficients", trained_model.coef_.tolist())
        np.savetxt("confusion_matrix.txt", cm, fmt="%d")
        mlflow.log_artifact("confusion_matrix.txt")
        mlflow.sklearn.log_model(trained_model, "model")


for max_depth_value in [5, 10, 15]:
    for criterion in ["gini", "entropy"]:
        with mlflow.start_run(
            run_name=f"DecisionTree_depth_{max_depth_value}_crit_{criterion}"
        ):
            model_dt = DecisionTreeClassifier(
                random_state=config["random_state"],
                max_depth=max_depth_value,
                criterion=criterion,
            )
            trained_model = train_model(model_dt, data["x_train"], data["y_train"])
            accuracy, f1, auc, cm = evaluate_model(
                trained_model, data["x_test"], data["y_test"]
            )

            mlflow.log_param("max_depth", max_depth_value)
            mlflow.log_param("criterion", criterion)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            if auc is not None:
                mlflow.log_metric("auc_roc", auc)
            if hasattr(trained_model, "get_depth"):
                mlflow.log_param("tree_depth", trained_model.get_depth())
            if hasattr(trained_model, "get_n_leaves"):
                mlflow.log_param("n_leaves", trained_model.get_n_leaves())
            np.savetxt("confusion_matrix.txt", cm, fmt="%d")
            mlflow.log_artifact("confusion_matrix.txt")
            mlflow.sklearn.log_model(trained_model, "model")
