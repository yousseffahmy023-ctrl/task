"""
Phase 2: Audio Preprocessing
Speech Emotion Recognition System
"""

import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

TARGET_SR     = 22050   # target sample rate (Hz)
TARGET_DURATION = 3.0   # seconds – pad/trim all clips to this length
MONO          = True


# ──────────────────────────────────────
# Load and preprocess a single audio file
# ──────────────────────────────────────
def load_audio(path: str, sr: int = TARGET_SR,
               duration: float = TARGET_DURATION,
               mono: bool = MONO):
    """
    Load an audio file, resample, convert to mono, normalize, and
    pad/trim to a fixed length.

    Returns (signal, sr) or (None, None) on failure.
    """
    try:
        y, orig_sr = librosa.load(path, sr=sr, mono=mono, duration=None)

        # Trim silence at start/end (top_db=20 is a gentle threshold)
        y, _ = librosa.effects.trim(y, top_db=20)

        # Pad or truncate to fixed length
        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        else:
            y = y[:target_len]

        # Amplitude normalization to [-1, 1]
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        return y, sr

    except Exception as e:
        return None, None


# ──────────────────────────────────────
# Quality-check the whole merged dataset
# ──────────────────────────────────────
def check_dataset_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify every audio file is loadable; mark bad rows.
    """
    bad_paths = []
    for idx, row in df.iterrows():
        y, sr = load_audio(row["path"])
        if y is None:
            bad_paths.append(idx)
            print(f"  ⚠ Corrupt/unreadable: {row['path']}")

    if bad_paths:
        print(f"\n🗑 Removing {len(bad_paths)} bad files from dataset")
        df = df.drop(index=bad_paths).reset_index(drop=True)
    else:
        print("✅ All audio files are readable")
    return df


# ──────────────────────────────────────
# Exploratory analysis on durations
# ──────────────────────────────────────
def analyze_durations(df: pd.DataFrame, sample_size: int = 500):
    """Sample audio durations and plot distribution."""
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    durations = []
    for _, row in sample.iterrows():
        try:
            dur = librosa.get_duration(path=row["path"])
            durations.append(dur)
        except:
            pass

    durations = np.array(durations)
    print(f"\n📏 Duration stats (seconds):")
    print(f"   Mean  : {durations.mean():.2f}")
    print(f"   Median: {np.median(durations):.2f}")
    print(f"   Min   : {durations.min():.2f}")
    print(f"   Max   : {durations.max():.2f}")

    plt.figure(figsize=(8, 4))
    plt.hist(durations, bins=40, color="teal", edgecolor="white")
    plt.axvline(TARGET_DURATION, color="red", linestyle="--",
                label=f"Target {TARGET_DURATION}s")
    plt.title("Distribution of Audio Durations")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("duration_distribution.png", dpi=150)
    plt.show()
    print("📊 Duration chart saved to duration_distribution.png")


# ──────────────────────────────────────
# Waveform + spectrogram visualisation
# ──────────────────────────────────────
def visualize_sample(path: str, emotion: str):
    y, sr = load_audio(path)
    if y is None:
        print("Could not load file")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title(f"Waveform  –  Emotion: {emotion}")

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                                   y_axis="mel", ax=axes[1])
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    axes[1].set_title("Mel Spectrogram")

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img2 = librosa.display.specshow(mfccs, sr=sr, x_axis="time", ax=axes[2])
    fig.colorbar(img2, ax=axes[2])
    axes[2].set_title("MFCCs (13 coefficients)")

    plt.tight_layout()
    plt.savefig("sample_visualization.png", dpi=150)
    plt.show()


# ──────────────────────────────────────
# MAIN
# ──────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("merged_dataset.csv")
    print(f"Loaded dataset: {len(df)} samples")

    # 1. Check integrity (optional but recommended first time)
    # df = check_dataset_integrity(df)
    # df.to_csv("merged_dataset_clean.csv", index=False)

    # 2. Explore durations
    analyze_durations(df)

    # 3. Visualise one example per emotion
    for emotion in df["emotion"].unique():
        sample_path = df[df["emotion"] == emotion].iloc[0]["path"]
        print(f"\n🎵 Visualising: {emotion}")
        visualize_sample(sample_path, emotion)
        break  # remove `break` to plot all emotions
