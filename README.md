🎤 AI Voice Deepfake Detection System

📌 Overview

This project is an AI-powered system that detects whether a given voice recording is REAL or FAKE (deepfake). It combines machine learning with audio feature extraction to identify synthetic or manipulated speech.

The system includes:

🧠 A trained ML model for classification
⚡ FastAPI backend for prediction
🎨 Streamlit UI for user interaction
🚀 Features
🎙 Upload audio file and detect authenticity
⚡ Real-time prediction (REAL / FAKE)
📊 Confidence score output
🌐 Interactive web interface using Streamlit
🧠 ML-based feature extraction (Librosa)

🛠️ Tech Stack

💻 Languages & Libraries
Python
NumPy
Librosa
Scikit-learn / PyTorch

⚙️ Backend
FastAPI

🎨 Frontend
Streamlit


📂 Project Structure
ai-voice-authenticity-detector/
│
├── app/                  # Streamlit frontend
├── backend/              # FastAPI backend
├── features/             # Feature extraction logic
├── training/             # Model training scripts
├── utils/                # Utility functions
├── requirements.txt
└── README.md
