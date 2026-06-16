# ── services/predictor.py ────────────────────────────────

import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Build paths ──────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, '..', '..', 'ml')

MODEL_PATH = os.path.join(ML_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(ML_DIR, 'vectorizer.pkl')

# ── Load model and vectorizer once at startup ────────────
print("Starting predictor....")
try:
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading vectorizer...")
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("✅ ML model loaded successfully!")

except FileNotFoundError:
    raise RuntimeError(
        "model.pkl or vectorizer.pkl not found. "
        "Please run ml/train.py first."
    )

# ── NLTK setup ───────────────────────────────────────────

stemmer = PorterStemmer()

try:
    stop_words = set(stopwords.words("english"))
except:
    stop_words = set()
# ── Text Cleaning ────────────────────────────────────────

def clean_text(text: str) -> str:

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ── Prediction Function ──────────────────────────────────

def predict_news(text: str, title: str = "") -> dict:

    combined = f"{title} {text}".strip()

    cleaned = clean_text(combined)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = round(float(probabilities.max()) * 100, 2)

    prediction_label = "FAKE" if prediction == 1 else "REAL"

    return {
        "prediction": prediction_label,
        "confidence": confidence
    }