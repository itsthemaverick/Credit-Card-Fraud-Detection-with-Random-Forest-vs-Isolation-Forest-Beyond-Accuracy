# Credit Card Fraud Detection

### Random Forest vs Isolation Forest --- Beyond Accuracy

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions
using two fundamentally different machine learning paradigms:

-   **Random Forest** (supervised classification)
-   **Isolation Forest** (unsupervised anomaly detection)

The goal is **not just high accuracy**, but understanding *why* certain
metrics can be misleading in highly imbalanced, real-world fraud
datasets --- and how to evaluate models responsibly.

------------------------------------------------------------------------

## ❗ The Core Problem

Credit card fraud detection suffers from:

-   **Extreme class imbalance**
-   **Asymmetric cost of errors**
-   **Metric deception**

A naïve accuracy-driven approach can appear strong while failing
completely at detecting fraud.

------------------------------------------------------------------------

## 🧠 Models Used

### Random Forest

-   Supervised ensemble model
-   Probability outputs allow threshold optimization
-   High capacity, prone to overfitting

### Isolation Forest

-   Unsupervised anomaly detection
-   Does not rely on labels
-   Useful when fraud labels are unavailable

------------------------------------------------------------------------

## 📊 Evaluation Metrics

  Metric      Importance
  ----------- ---------------------------------
  Accuracy    Misleading under imbalance
  Precision   Cost of false positives
  Recall      Cost of false negatives
  F1 Score    Precision--Recall balance
  ROC-AUC     Ranking capability
  PR-AUC      Best metric for fraud detection

------------------------------------------------------------------------

## 🔍 Threshold Optimization

Instead of using a fixed 0.5 threshold, probabilities were evaluated
across thresholds to maximize F1 score, simulating real-world
cost-sensitive decision making.

------------------------------------------------------------------------

## ⚠️ Limitations of Random Forest

-   Risk of overfitting
-   Sensitive to threshold choice
-   Poor extrapolation to unseen fraud
-   Limited interpretability

------------------------------------------------------------------------

## ⚠️ Limitations of Isolation Forest

-   Assumes fraud is isolated
-   Cannot learn complex fraud patterns
-   Accuracy inflated by imbalance
-   No probability-based tuning

------------------------------------------------------------------------

## 🔧 Problems Solved

-   Demonstrates metric pitfalls
-   Compares supervised vs unsupervised learning
-   Applies threshold tuning
-   Uses PR-AUC appropriately
-   Encourages honest evaluation

------------------------------------------------------------------------

## 📁 Project Structure

    .
    ├── main.py
    ├── src/
    │   ├── preprocessing.py
    │   ├── train_rf.py
    │   ├── train_if.py
    │   ├── evaluate.py
    │   ├── thresholding.py
    │   ├── visualize.py
    │   └── compare.py
    └── README.md

------------------------------------------------------------------------

## 🚀 Key Takeaways

-   Accuracy alone is unreliable
-   Threshold tuning matters
-   PR-AUC is critical for fraud detection
-   Honest evaluation beats inflated metrics

------------------------------------------------------------------------

## Author : Itsthemaverick
