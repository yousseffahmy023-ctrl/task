# Speech Emotion Recognition System
**Datasets:** CREMA-D · RAVDESS · SAVEE · TESS

---

## Project Structure

```
SER_Project/
│
├── phase1_data_integration.py   ← Load + merge all 4 datasets
├── phase2_preprocessing.py      ← Audio loading, trimming, normalisation
├── phase3_feature_extraction.py ← MFCCs, Chroma, Mel Spec, etc.
├── phase4_dataset_preparation.py← Label encoding, SMOTE, train/val/test split
├── phase5_modeling.py           ← Train & tune 5 ML models
├── phase6_evaluation.py         ← Accuracy / F1 / Confusion Matrix
├── app.py                       ← Streamlit web app
└── requirements.txt
```

---

## Setup (run once)

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Execution

### 1. Download datasets from Kaggle

| Dataset | Kaggle Link |
|---------|------------|
| CREMA-D | https://www.kaggle.com/datasets/ejlok1/cremad |
| RAVDESS | https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio |
| SAVEE   | https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee |
| TESS    | https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess |

### 2. Update paths in phase1_data_integration.py

```python
CREMAD_PATH  = r"C:\Datasets\CREMA-D\AudioWAV"
RAVDESS_PATH = r"C:\Datasets\RAVDESS"
SAVEE_PATH   = r"C:\Datasets\SAVEE\AudioData"
TESS_PATH    = r"C:\Datasets\TESS"
```

### 3. Run phases in order

```bash
python phase1_data_integration.py    # → merged_dataset.csv
python phase2_preprocessing.py       # → EDA + visualisations
python phase3_feature_extraction.py  # → features.pkl
python phase4_dataset_preparation.py # → prepared_data.pkl, scaler.pkl, label_encoder.pkl
python phase5_modeling.py            # → best_model.pkl, all_models.pkl
python phase6_evaluation.py          # → confusion_matrix.png, evaluation_results.csv
```

### 4. Launch the web app

```bash
streamlit run app.py
```

---

## Emotion Labels (Unified)

| Emotion  | CREMA-D | RAVDESS | SAVEE | TESS |
|----------|---------|---------|-------|------|
| angry    | ANG     | 05      | a     | ✅   |
| disgust  | DIS     | 07      | d     | ✅   |
| fear     | FEA     | 06      | f     | ✅   |
| happy    | HAP     | 03      | h     | ✅   |
| neutral  | NEU     | 01,02   | n     | ✅   |
| sad      | SAD     | 04      | sa    | ✅   |
| surprise | —       | 08      | su    | ps   |

---

## Output Files Generated

| File | Description |
|------|-------------|
| `merged_dataset.csv` | All audio paths + emotion labels |
| `features.pkl` | Extracted feature matrix X + labels y |
| `prepared_data.pkl` | Scaled train/val/test splits |
| `scaler.pkl` | Fitted StandardScaler |
| `label_encoder.pkl` | Fitted LabelEncoder |
| `best_model.pkl` | Best trained classifier |
| `evaluation_results.csv` | Final metrics |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `model_comparison.png` | Bar chart of model val accuracies |
