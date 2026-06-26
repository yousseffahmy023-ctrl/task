"""
Phase 7: Streamlit Deployment
Speech Emotion Recognition System

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
import os
import matplotlib.pyplot as plt
import librosa.display

# ── Load saved artifacts ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        bm = pickle.load(f)
    return bm["model"], bm["name"]

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)


# ── Feature extraction (same as Phase 3) ────────────────────────
def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    N_MFCC, N_MELS, N_CHROMA = 40, 128, 12
    features = []

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    features.extend(np.mean(mel, axis=1))
    features.extend(np.std(mel, axis=1))

    for fn in [librosa.feature.spectral_centroid,
               librosa.feature.spectral_rolloff]:
        f = fn(y=y, sr=sr)
        features.extend([float(np.mean(f)), float(np.std(f))])

    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend([float(np.mean(zcr)), float(np.std(zcr))])

    rms = librosa.feature.rms(y=y)
    features.extend([float(np.mean(rms)), float(np.std(rms))])

    return np.array(features, dtype=np.float32)


def preprocess(path: str, sr: int = 22050, duration: float = 3.0):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=None)
    y, _ = librosa.effects.trim(y, top_db=20)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y, sr


def make_visualisations(y, sr):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title("Waveform")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                                   y_axis="mel", ax=axes[1])
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    axes[1].set_title("Mel Spectrogram")

    plt.tight_layout()
    return fig


EMOTION_EMOJI = {
    "angry":   "😠", "disgust": "🤢", "fear":    "😨",
    "happy":   "😄", "neutral": "😐", "sad":     "😢",
    "surprise": "😲"
}


# ── Streamlit UI ─────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Speech Emotion Recognition",
        page_icon="🎙️",
        layout="centered"
    )

    st.title("🎙️ Speech Emotion Recognition")
    st.markdown(
        "Upload a WAV / MP3 audio clip and the model will predict "
        "the speaker's emotion."
    )

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            "**Datasets used:** CREMA-D, RAVDESS, SAVEE, TESS\n\n"
            "**Features:** MFCCs, Chroma, Mel Spectrogram, "
            "Spectral Centroid, Roll-off, ZCR, RMS\n\n"
            "**Emotions:** angry · disgust · fear · happy · neutral · sad · surprise"
        )
        try:
            _, model_name = load_model()
            st.success(f"✅ Model loaded: **{model_name}**")
        except Exception as e:
            st.error(f"Model load error: {e}")

    # File upload
    uploaded = st.file_uploader(
        "Choose an audio file", type=["wav", "mp3", "ogg", "flac"]
    )

    if uploaded is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded.name)[1]
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.audio(uploaded)

        with st.spinner("🔄 Processing audio …"):
            try:
                # 1. Preprocess
                y, sr = preprocess(tmp_path)

                # 2. Visualise
                fig = make_visualisations(y, sr)
                st.pyplot(fig)

                # 3. Extract features
                feat = extract_features(y, sr).reshape(1, -1)

                # 4. Scale
                scaler = load_scaler()
                feat_scaled = scaler.transform(feat)

                # 5. Predict
                model, model_name = load_model()
                le = load_encoder()

                y_pred = model.predict(feat_scaled)[0]
                emotion = le.inverse_transform([y_pred])[0]

                # Confidence (if model supports it)
                confidence = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(feat_scaled)[0]
                    confidence = proba[y_pred]

                # 6. Display result
                emoji = EMOTION_EMOJI.get(emotion, "🎭")
                st.markdown("---")
                st.markdown(f"## Predicted Emotion: {emoji} **{emotion.upper()}**")

                if confidence is not None:
                    st.progress(float(confidence))
                    st.caption(f"Confidence: {confidence * 100:.1f}%")

                # Show all class probabilities
                if hasattr(model, "predict_proba"):
                    proba_df = {
                        le.inverse_transform([i])[0]: round(float(p), 4)
                        for i, p in enumerate(proba)
                    }
                    st.bar_chart(proba_df)

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
            finally:
                os.remove(tmp_path)


if __name__ == "__main__":
    main()
