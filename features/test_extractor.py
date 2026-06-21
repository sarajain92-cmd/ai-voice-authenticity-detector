print("🧪 TESTING EXTRACTOR (KAGGLE DATASET)")

import os
import numpy as np
import matplotlib.pyplot as plt

from features.extractor import extract_mel, extract_mfcc, pad_or_trim

csv_path = "data/KAGGLE/dataset-balanced.csv"
audio_dir = "data/KAGGLE/audio"

import pandas as pd

df = pd.read_csv(csv_path)
print(df.columns.tolist())
print(df.head())


# ek real aur ek fake sample pick karo
real_file = df[df['label'] == 0].iloc[0]['file']
fake_file = df[df['label'] == 1].iloc[0]['file']

real_file = os.path.join(audio_dir, real_file)
fake_file = os.path.join(audio_dir, fake_file)

print("Real file:", real_file)
print("Fake file:", fake_file)

# 🔹 ek real aur ek fake file automatically pick karege
print("Real file:", real_file)
print("Fake file:", fake_file)


# 🔹 MEL
mel_real = extract_mel(real_file)
mel_fake = extract_mel(fake_file)

print("Mel REAL shape:", mel_real.shape)
print("Mel FAKE shape:", mel_fake.shape)


# 🔹 MFCC
mfcc_real = extract_mfcc(real_file)
mfcc_fake = extract_mfcc(fake_file)

mfcc_real = pad_or_trim(mfcc_real)
mfcc_fake = pad_or_trim(mfcc_fake)

print("MFCC REAL shape:", mfcc_real.shape)
print("MFCC FAKE shape:", mfcc_fake.shape)


# 🔹 NaN CHECK
print("NaN in MEL:", np.isnan(mel_real).any())
print("NaN in MFCC:", np.isnan(mfcc_real).any())


# 🔹 VISUALIZATION 🔥
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("REAL MEL")
plt.imshow(mel_real, aspect='auto', origin='lower')

plt.subplot(1,2,2)
plt.title("FAKE MEL")
plt.imshow(mel_fake, aspect='auto', origin='lower')

plt.tight_layout()
plt.show()