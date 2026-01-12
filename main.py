from src.data_loader import load_data
from src.preprocessing import preprocess
from src.train_random_forest import train_rf
from src.train_isolation_forest import train_if
from src.evaluate import evaluate_classifier, evaluate_isolation_forest
from src.visualize import (
    plot_roc,
    plot_f1_vs_threshold, plot_confusion
)
from src.evaluate import compute_accuracy
from src.threshold import find_best_threshold
from src.compare import plot_accuracy_comparison


df = load_data()
X_train, X_test, y_train, y_test = preprocess(df)

# Random Forest
rf = train_rf(X_train, y_train)
y_pred_rf, y_prob_rf, rf_metrics = evaluate_classifier(
    rf, X_test, y_test, "rf"
)
best_t, best_f1 = find_best_threshold(y_test, y_prob_rf)

print(f"Best Threshold (F1-optimized): {best_t:.3f}")
print(f"Best F1 Score: {best_f1:.3f}")

y_pred_rf_opt = (y_prob_rf > best_t).astype(int)

acc_rf = compute_accuracy(y_test, y_pred_rf_opt)
print(f"Random Forest Accuracy: {acc_rf:.4f}")

plot_confusion(y_test, y_pred_rf_opt)
plot_roc(y_test, y_prob_rf)
plot_f1_vs_threshold(y_test, y_prob_rf)


# Isolation Forest
iso = train_if(X_train,y_train)
y_pred_if, if_metrics = evaluate_isolation_forest(
    iso, X_test, y_test
)
# Convert Isolation Forest output
y_pred_if = (y_pred_if == -1).astype(int)

acc_if = compute_accuracy(y_test, y_pred_if)
print(f"Isolation Forest Accuracy: {acc_if:.4f}")



plot_accuracy_comparison(acc_rf, acc_if)
