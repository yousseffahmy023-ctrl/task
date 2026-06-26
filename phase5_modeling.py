"""
Phase 5: Machine Learning Modeling
Speech Emotion Recognition System

Models trained and compared:
  1. Logistic Regression
  2. Support Vector Machine (SVM)
  3. Random Forest
  4. K-Nearest Neighbors (KNN)
  5. Gradient Boosting (optional)
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics        import accuracy_score


RANDOM_STATE = 42


def load_data(path: str = "prepared_data.pkl"):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return (d["X_train"], d["X_val"], d["X_test"],
            d["y_train"], d["y_val"], d["y_test"])


def load_encoder(path: str = "label_encoder.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────
# Model definitions + hyperparameter grids
# ──────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "saga"]
            }
        },
        "SVM": {
            "model": SVC(probability=True, random_state=RANDOM_STATE),
            "params": {
                "C":      [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma":  ["scale", "auto"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [100, 200],
                "max_depth":    [None, 20, 40],
                "min_samples_split": [2, 5]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 11],
                "weights":     ["uniform", "distance"],
                "metric":      ["euclidean", "manhattan"]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators":  [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [3, 5]
            }
        }
    }


def train_and_tune(name, cfg, X_train, y_train, X_val, y_val):
    print(f"\n🔧 Training: {name}")
    gs = GridSearchCV(
        cfg["model"], cfg["params"],
        cv=5, scoring="accuracy", n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    train_acc = accuracy_score(y_train, best.predict(X_train))
    val_acc   = accuracy_score(y_val,   best.predict(X_val))

    print(f"   Best params : {gs.best_params_}")
    print(f"   Train acc   : {train_acc:.4f}")
    print(f"   Val acc     : {val_acc:.4f}")

    return best, val_acc, gs.best_params_


def compare_models(results: dict):
    """Bar chart comparing validation accuracy across models."""
    names  = list(results.keys())
    scores = [results[n]["val_acc"] for n in names]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, scores, color="cornflowerblue", edgecolor="black")
    for bar, sc in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{sc:.3f}", ha="center", va="bottom", fontsize=10)
    plt.ylim(0, 1.05)
    plt.title("Model Comparison – Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()
    print("📊 Comparison chart saved to model_comparison.png")


# ──────────────────────────────────────
# MAIN
# ──────────────────────────────────────
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    le = load_encoder()

    models_cfg = get_models()
    results = {}

    for name, cfg in models_cfg.items():
        best_model, val_acc, best_params = train_and_tune(
            name, cfg, X_train, y_train, X_val, y_val
        )
        results[name] = {
            "model":      best_model,
            "val_acc":    val_acc,
            "best_params": best_params
        }

    # ── Summary table ──────────────────────────
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    summary = pd.DataFrame([
        {"Model": n, "Val Accuracy": f"{v['val_acc']:.4f}"}
        for n, v in results.items()
    ]).sort_values("Val Accuracy", ascending=False)
    print(summary.to_string(index=False))

    # ── Pick the best ──────────────────────────
    best_name = max(results, key=lambda n: results[n]["val_acc"])
    best_model = results[best_name]["model"]
    print(f"\n🏆 Best model: {best_name} (val acc = {results[best_name]['val_acc']:.4f})")

    # Save best model
    with open("best_model.pkl", "wb") as f:
        pickle.dump({"name": best_name, "model": best_model}, f)
    print("💾 Best model saved to best_model.pkl")

    # Save all results for Phase 6
    with open("all_models.pkl", "wb") as f:
        pickle.dump(results, f)

    compare_models(results)
