"""
Phase 4: Dataset Preparation
Speech Emotion Recognition System

- Encode emotion labels
- Handle class imbalance (SMOTE oversampling)
- Split into train / validation / test sets
- Scale features
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

# Optional – install with: pip install imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("ℹ  imbalanced-learn not installed. SMOTE will be skipped.")
    print("   Install: pip install imbalanced-learn")


RANDOM_STATE = 42


def load_features(path: str = "features.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"]


def encode_labels(y: np.ndarray):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"\n📌 Emotion classes: {list(le.classes_)}")
    return y_enc, le


def show_class_distribution(y_raw, title="Class Distribution"):
    counts = Counter(y_raw)
    plt.figure(figsize=(8, 4))
    plt.bar(counts.keys(), counts.values(), color="steelblue", edgecolor="black")
    plt.title(title)
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()


def balance_classes(X: np.ndarray, y: np.ndarray):
    """Apply SMOTE to oversample minority classes."""
    if not SMOTE_AVAILABLE:
        print("⚠  SMOTE not available – using raw (imbalanced) data")
        return X, y

    print("\n⚖ Applying SMOTE to balance classes …")
    before = Counter(y)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    after  = Counter(y_res)
    print(f"   Before: {dict(before)}")
    print(f"   After : {dict(after)}")
    return X_res, y_res


def prepare_splits(X: np.ndarray, y: np.ndarray):
    """
    Split: 70 % train | 15 % validation | 15 % test
    Then scale with StandardScaler fitted ONLY on train set.
    """
    # First split: 70 % train, 30 % temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    # Second split: 50 / 50 of the 30 % → 15 % val, 15 % test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"\n📦 Split sizes:")
    print(f"   Train : {X_train.shape[0]} samples")
    print(f"   Val   : {X_val.shape[0]} samples")
    print(f"   Test  : {X_test.shape[0]} samples")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Save scaler for Streamlit deployment
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("💾 Scaler saved to scaler.pkl")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


if __name__ == "__main__":
    # 1. Load features
    X, y_raw = load_features()

    # 2. Class distribution before balancing
    show_class_distribution(y_raw, "Class Distribution (Before Balancing)")

    # 3. Encode
    y_enc, label_encoder = encode_labels(y_raw)

    # 4. Balance
    X_bal, y_bal = balance_classes(X, y_enc)

    # 5. Distribution after balancing
    inv = label_encoder.inverse_transform(y_bal)
    show_class_distribution(inv, "Class Distribution (After SMOTE)")

    # 6. Train/Val/Test splits + scaling
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_splits(X_bal, y_bal)

    # 7. Save everything
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open("prepared_data.pkl", "wb") as f:
        pickle.dump({
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val,  "y_test": y_test,
        }, f)

    print("\n✅ Preparation complete. Files saved:")
    print("   prepared_data.pkl | scaler.pkl | label_encoder.pkl")
