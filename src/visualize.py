import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,confusion_matrix
import numpy as np 
import os 

os.makedirs("results/plots" , exist_ok=True)

def plot_roc(y_test,y_prob):
    fpr,tpr,_ = roc_curve(y_test,y_prob)
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Postive Rate")
    plt.title("ROC Curve")
    plt.savefig("results/plots/roc_curve.png")
    plt.close()

from sklearn.metrics import f1_score

def plot_f1_vs_threshold(y_test, y_prob):
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []

    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        f1_scores.append(f1_score(y_test, y_pred))

    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Threshold")
    plt.savefig("results/plots/f1_threshold.png")
    plt.close()


def plot_confusion(y_test,y_pred):
    cm = confusion_matrix(y_test,y_pred)
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("results/plots/confusion.png")
    plt.close()

