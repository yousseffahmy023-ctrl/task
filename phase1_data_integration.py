"""
Phase 1: Data Understanding and Integration
Speech Emotion Recognition System
Datasets: CREMA-D, RAVDESS, SAVEE, TESS
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURE THESE PATHS AFTER DOWNLOADING DATA
# ─────────────────────────────────────────────
CREMAD_PATH  = r"C:\Datasets\CREMA-D\AudioWAV"
RAVDESS_PATH = r"C:\Datasets\RAVDESS"      # top folder containing Actor_01 .. Actor_24
SAVEE_PATH   = r"C:\Datasets\SAVEE\AudioData"
TESS_PATH    = r"C:\Datasets\TESS"         # top folder containing OAF_xxx & YAF_xxx


# ──────────────────────────────────────
# Unified emotion label mapping
# ──────────────────────────────────────
UNIFIED_EMOTIONS = {
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
}


# ──────────────────────────────────────
# CREMA-D loader
# Filename pattern: <ActorID>_<Sentence>_<Emotion>_<Intensity>.wav
# Emotion codes: ANG, DIS, FEA, HAP, NEU, SAD
# ──────────────────────────────────────
def load_cremad(path: str) -> pd.DataFrame:
    emotion_map = {
        "ANG": "angry", "DIS": "disgust", "FEA": "fear",
        "HAP": "happy",  "NEU": "neutral", "SAD": "sad"
    }
    records = []
    for fname in os.listdir(path):
        if not fname.endswith(".wav"):
            continue
        parts = fname.split("_")
        if len(parts) < 3:
            continue
        code = parts[2].upper()
        emotion = emotion_map.get(code)
        if emotion:
            records.append({
                "path": os.path.join(path, fname),
                "emotion": emotion,
                "source": "CREMA-D"
            })
    print(f"[CREMA-D]  Loaded {len(records)} samples")
    return pd.DataFrame(records)


# ──────────────────────────────────────
# RAVDESS loader
# Filename: 03-01-<emotion>-<intensity>-<stmt>-<rep>-<actor>.wav
# Emotion digit → label
# ──────────────────────────────────────
def load_ravdess(path: str) -> pd.DataFrame:
    emotion_map = {
        "01": "neutral", "02": "calm",    "03": "happy",
        "04": "sad",     "05": "angry",   "06": "fear",
        "07": "disgust", "08": "surprise"
    }
    # "calm" is RAVDESS-specific; map to neutral for unified scheme
    unified_map = {
        "neutral": "neutral", "calm": "neutral", "happy": "happy",
        "sad": "sad", "angry": "angry", "fear": "fear",
        "disgust": "disgust", "surprise": "surprise"
    }
    records = []
    for actor_dir in Path(path).rglob("*.wav"):
        parts = actor_dir.stem.split("-")
        if len(parts) < 3:
            continue
        code = parts[2]
        raw_emotion = emotion_map.get(code)
        emotion = unified_map.get(raw_emotion) if raw_emotion else None
        if emotion:
            records.append({
                "path": str(actor_dir),
                "emotion": emotion,
                "source": "RAVDESS"
            })
    print(f"[RAVDESS]  Loaded {len(records)} samples")
    return pd.DataFrame(records)


# ──────────────────────────────────────
# SAVEE loader
# Filename: <Speaker>_<emotion-code><number>.wav
# Codes: a=anger, d=disgust, f=fear, h=happiness, n=neutral, sa=sadness, su=surprise
# ──────────────────────────────────────
def load_savee(path: str) -> pd.DataFrame:
    emotion_map = {
        "a":  "angry",   "d":  "disgust", "f":  "fear",
        "h":  "happy",   "n":  "neutral", "sa": "sad",
        "su": "surprise"
    }
    records = []
    for fname in Path(path).rglob("*.wav"):
        stem = fname.stem  # e.g. DC_a01
        # strip speaker prefix
        name = stem.split("_")[-1] if "_" in stem else stem
        # try 2-char code first, then 1-char
        code = None
        for prefix in ["sa", "su", "a", "d", "f", "h", "n"]:
            if name.lower().startswith(prefix):
                code = prefix
                break
        emotion = emotion_map.get(code)
        if emotion:
            records.append({
                "path": str(fname),
                "emotion": emotion,
                "source": "SAVEE"
            })
    print(f"[SAVEE]    Loaded {len(records)} samples")
    return pd.DataFrame(records)


# ──────────────────────────────────────
# TESS loader
# Folder name: OAF_<word>_<emotion> / YAF_<word>_<emotion>
# ──────────────────────────────────────
def load_tess(path: str) -> pd.DataFrame:
    emotion_map = {
        "angry":   "angry",   "disgust": "disgust", "fear":    "fear",
        "happy":   "happy",   "neutral": "neutral", "pleasant_surprise": "surprise",
        "ps":      "surprise", "sad":    "sad"
    }
    records = []
    for folder in Path(path).iterdir():
        if not folder.is_dir():
            continue
        parts = folder.name.lower().split("_")
        emotion_key = "_".join(parts[1:]) if len(parts) > 2 else parts[-1]
        emotion = emotion_map.get(emotion_key) or emotion_map.get(parts[-1])
        if not emotion:
            continue
        for wav in folder.glob("*.wav"):
            records.append({
                "path": str(wav),
                "emotion": emotion,
                "source": "TESS"
            })
    print(f"[TESS]     Loaded {len(records)} samples")
    return pd.DataFrame(records)


# ──────────────────────────────────────
# Merge & explore
# ──────────────────────────────────────
def merge_datasets(cremad_path, ravdess_path, savee_path, tess_path) -> pd.DataFrame:
    dfs = []
    for loader, p in [
        (load_cremad,  cremad_path),
        (load_ravdess, ravdess_path),
        (load_savee,   savee_path),
        (load_tess,    tess_path),
    ]:
        try:
            df = loader(p)
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠ Skipped ({e})")

    merged = pd.concat(dfs, ignore_index=True)
    # Drop rows whose emotion is not in the unified set
    merged = merged[merged["emotion"].isin(UNIFIED_EMOTIONS)].reset_index(drop=True)
    print(f"\n✅ Total samples after merging: {len(merged)}")
    print(merged["emotion"].value_counts())
    return merged


def plot_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per emotion
    df["emotion"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("Samples per Emotion (All Datasets)")
    axes[0].set_xlabel("Emotion")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Per source
    df["source"].value_counts().plot(kind="bar", ax=axes[1], color="coral", edgecolor="black")
    axes[1].set_title("Samples per Dataset Source")
    axes[1].set_xlabel("Source")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()
    print("📊 Distribution chart saved to class_distribution.png")


# ──────────────────────────────────────
# MAIN
# ──────────────────────────────────────
if __name__ == "__main__":
    df = merge_datasets(CREMAD_PATH, RAVDESS_PATH, SAVEE_PATH, TESS_PATH)

    # Save merged metadata
    df.to_csv("merged_dataset.csv", index=False)
    print("\n💾 Saved merged_dataset.csv")
    print(df.head())

    plot_distribution(df)
