import matplotlib.pyplot as plt

def plot_accuracy_comparison(acc_rf, acc_if):
    models = ["Random Forest", "Isolation Forest"]
    accuracies = [acc_rf, acc_if]

    plt.figure()
    plt.bar(models, accuracies)
    plt.ylim(0.9, 1.0)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.show()
