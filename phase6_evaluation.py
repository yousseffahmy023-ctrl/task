"""
Phase 6: Model Evaluation
Speech Emotion Recognition System

Metrics:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - Full Classification Report
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def load_artifacts():
    with open("prepared_data.pkl", "rb") as f:
        d = pickle.load(f)
    with open("best_model.pkl", "rb") as f:
        bm = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return (d["X_test"], d["y_test"],
            bm["model"], bm["name"], le)


def evaluate(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred,    average="weighted")
    f1   = f1_score(y_test, y_pred,        average="weighted")

    print("\n" + "="*50)
    print("EVALUATION RESULTS (Test Set)")
    print("="*50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=class_names,
                                digits=4))
    return y_pred, acc, prec, rec, f1


def plot_confusion_matrix(y_test, y_pred, class_names, model_name):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{model_name}\nConfusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{model_name}\nConfusion Matrix (Normalised)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("📊 Confusion matrix saved to confusion_matrix.png")


def plot_per_class_metrics(y_test, y_pred, class_names):
    report = classification_report(y_test, y_pred,
                                   target_names=class_names,
                                   output_dict=True)
    metrics_df = pd.DataFrame(report).T.iloc[:-3][["precision", "recall", "f1-score"]]

    metrics_df.plot(kind="bar", figsize=(12, 5), edgecolor="black")
    plt.title("Per-Class Precision / Recall / F1-Score")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("per_class_metrics.png", dpi=150)
    plt.show()
    print("📊 Per-class metrics chart saved to per_class_metrics.png")


# ──────────────────────────────────────
# MAIN
# ──────────────────────────────────────
if __name__ == "__main__":
    X_test, y_test, model, model_name, le = load_artifacts()
    class_names = list(le.classes_)

    print(f"🔍 Evaluating: {model_name}")
    y_pred, acc, prec, rec, f1 = evaluate(model, X_test, y_test, class_names)

    plot_confusion_matrix(y_test, y_pred, class_names, model_name)
    plot_per_class_metrics(y_test, y_pred, class_names)

    # Save metrics summary
    metrics = {
        "model":     model_name,
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1_score":  round(f1, 4),
    }
    pd.DataFrame([metrics]).to_csv("evaluation_results.csv", index=False)
    print("\n💾 Results saved to evaluation_results.csv")
