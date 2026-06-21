import torch
import librosa
import numpy as np
from models.mlp_classifier import DeepfakeMLP
from utils.features import extract_features

# load model
model = DeepfakeMLP(input_dim=26)
model.load_state_dict(torch.load("models/mlp_best.pt"))
model.eval()

# input audio
file_path = input("Enter audio file path: ")

# extract features
audio, sr = librosa.load(file_path, sr=None)
features = extract_features(audio,sr)
print("Feature length:", len(features))
if len(features) == 13:
    features = np.concatenate((features, features)) 
features = np.array(features).astype("float32")
features = torch.tensor(features).unsqueeze(0)

# prediction
with torch.no_grad():
    output = model(features)
    pred = torch.argmax(output, dim=1).item()

if pred == 1:
    print("🟢 REAL VOICE")
else:
    print("🔴 FAKE VOICE")