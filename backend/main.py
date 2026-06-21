from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa
import numpy as np
import joblib
import tempfile
import os

from models.mlp_classifier import DeepfakeMLP

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + scaler at startup
model = DeepfakeMLP(input_dim=26)
model.load_state_dict(torch.load("models/mlp_best.pt", map_location="cpu"))
model.eval()

scaler = joblib.load("models/scaler.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
    
    return np.array([chroma, rms, spec_cent, spec_bw, rolloff, zcr, *mfccs])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        features = extract_features(tmp_path)
        features_scaled = scaler.transform([features])
        
        x = torch.tensor(features_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1).item()
            confidence = probs[0][pred].item()
        
        label = "FAKE" if pred == 1 else "REAL"
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "is_deepfake": pred == 1
        }
    finally:
        os.unlink(tmp_path)

@app.get("/")
def root():
    return {"status": "Voice Deepfake Detector API is running"}