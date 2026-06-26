"""
Phase 3: Feature Extraction
Speech Emotion Recognition System

Features extracted per audio clip:
  • MFCC (40 coefficients)  → mean + std  = 80 values
  • Chroma                  → mean + std  = 24 values
  • Mel Spectrogram         → mean + std  = 256 values
  • Spectral Centroid       → mean + std  = 2 values
  • Spectral Roll-off       → mean + std  = 2 values
  • Zero Crossing Rate      → mean + std  = 2 values
  • RMS Energy              → mean + std  = 2 values
  Total feature vector: 368 features per sample
"""

import numpy as np
import pandas as pd
import librosa
import pickle
from tqdm import tqdm
from phase2_preprocessing import load_audio   # reuse our loading function

N_MFCC     = 40
N_MELS     = 128
N_CHROMA   = 12


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a preprocessed audio signal.
    Uses mean + std across time frames for each feature.
    """
    features = []

    # ── MFCCs ──────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    # ── Chroma ─────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # ── Mel Spectrogram ─────────────────────────
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    features.extend(np.mean(mel, axis=1))
    features.extend(np.std(mel, axis=1))

    # ── Spectral Centroid ───────────────────────
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(np.mean(sc)))
    features.append(float(np.std(sc)))

    # ── Spectral Roll-off ───────────────────────
    sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(np.mean(sr_feat)))
    features.append(float(np.std(sr_feat)))

    # ── Zero Crossing Rate ───────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(float(np.mean(zcr)))
    features.append(float(np.std(zcr)))

    # ── RMS Energy ──────────────────────────────
    rms = librosa.feature.rms(y=y)
    features.append(float(np.mean(rms)))
    features.append(float(np.std(rms)))

    return np.array(features, dtype=np.float32)


def extract_all(df: pd.DataFrame, save_path: str = "features.pkl") -> tuple:
    """
    Iterate over every row of the merged dataset, load audio, and extract features.
    Returns X (feature matrix) and y (emotion labels).
    """
    X, y = [], []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio, sr = load_audio(row["path"])
        if audio is None:
            failed += 1
            continue
        feat = extract_features(audio, sr)
        X.append(feat)
        y.append(row["emotion"])

    X = np.array(X)
    y = np.array(y)
    print(f"\n✅ Feature matrix shape : {X.shape}")
    print(f"   Failed / skipped     : {failed}")

    # Save to disk
    with open(save_path, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    print(f"💾 Features saved to {save_path}")

    return X, y


if __name__ == "__main__":
    df = pd.read_csv("merged_dataset.csv")
    X, y = extract_all(df)
    print("\nFeature vector breakdown:")
    print(f"  MFCC (mean+std)        : {N_MFCC * 2}")
    print(f"  Chroma (mean+std)      : {N_CHROMA * 2}")
    print(f"  Mel Spec (mean+std)    : {N_MELS * 2}")
    print(f"  Spectral Centroid      : 2")
    print(f"  Spectral Rolloff       : 2")
    print(f"  Zero Crossing Rate     : 2")
    print(f"  RMS Energy             : 2")
    print(f"  ─────────────────────────")
    print(f"  Total                  : {X.shape[1]}")
