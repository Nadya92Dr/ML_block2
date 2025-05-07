from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from config import config


def get_data():
    data = load_wine()
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config["data"]["test_size"], random_state=config["random_state"]
    )
    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}
