import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.metrics import accuracy_score

def evaluate_classifier(model, X_test, y_test, name):

    # Safety checks
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return y_pred, y_prob, metrics


def evaluate_isolation_forest(model, X_test, y_test):

    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    y_pred = model.predict(X_test)
    y_pred = (y_pred == -1).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    with open("results/metrics_if.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return y_pred, metrics



def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
