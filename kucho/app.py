from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

app = FastAPI(
    title="Depression Detection API",
    version="1.0"
)

# Load ensemble package
package = joblib.load("models/depression_ensemble.pkl")

tfidf = package["tfidf"]
log_model = package["log_model"]
svm_model = package["svm_model"]
rf_model = package["rf_model"]
label_encoder = package["label_encoder"]

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
def predict(data: TextInput):
    text = data.text

    # Transform text
    X = tfidf.transform([text])

    # Model probabilities
    log_prob = log_model.predict_proba(X)
    svm_prob = svm_model.predict_proba(X)
    rf_prob = rf_model.predict_proba(X)

    # Ensemble average
    avg_prob = (log_prob + svm_prob + rf_prob) / 3

    pred_index = np.argmax(avg_prob, axis=1)[0]

    prediction = label_encoder.inverse_transform([pred_index])[0]

    confidence = float(np.max(avg_prob))

    return {
        "text": text,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": avg_prob.tolist()[0]
    }

# Serve static files from frontend directory (for CSS, JS, images if needed)
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")