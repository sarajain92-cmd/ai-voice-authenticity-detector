print("SCRIPT STARTED 🚀")

import librosa
import os
import numpy as np

real_path = "data/KAGGLE/AUDIO/REAL"
fake_path = "data/KAGGLE/AUDIO/FAKE"

X = []   # features
y = []   # labels (0 = real, 1 = fake)

def extract_feature(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# 🔹 REAL files
print("Processing REAL...")
for file in os.listdir(real_path):
    file_path = os.path.join(real_path, file)
    feature = extract_feature(file_path)
    X.append(feature)
    y.append(0)

# 🔹 FAKE files
print("Processing FAKE...")
for file in os.listdir(fake_path):
    file_path = os.path.join(fake_path, file)
    feature = extract_feature(file_path)
    X.append(feature)
    y.append(1)

# convert to numpy
X = np.array(X)
y = np.array(y)

# save
np.save("X.npy", X)
np.save("y.npy", y)

print("✅ FEATURES SAVED!")
print("X shape:", X.shape)
print("y shape:", y.shape)