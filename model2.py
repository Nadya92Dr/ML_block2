import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
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

    mlflow.log_param("max_depth", config["decision_tree"]["max_depth"])
    mlflow.log_param("criterion", config["decision_tree"]["criterion"])
    mlflow.log_param("random_state", config["random_state"])

    if hasattr(model, "get_depth"):
        tree_depth = model.get_depth()
        mlflow.log_param("tree_depth", tree_depth)
    if hasattr(model, "get_n_leaves"):
        n_leaves = model.get_n_leaves()
        mlflow.log_param("n_leaves", n_leaves)

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

    with mlflow.start_run(run_name="DecisionTree_Experiment"):
        decision_tree_model = DecisionTreeClassifier(
            random_state=config["random_state"],
            max_depth=config["decision_tree"]["max_depth"],
            criterion=config["decision_tree"]["criterion"],
        )

        data = get_data()
        trained_model = train(decision_tree_model, data["x_train"], data["y_train"])
        metrics = test(trained_model, data["x_test"], data["y_test"])
        mlflow.sklearn.log_model(trained_model, "model")
