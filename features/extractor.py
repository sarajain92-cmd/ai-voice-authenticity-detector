import librosa
import numpy as np

# 🔹 FIXED SAMPLE RATE
SR = 16000

# 🔹 MEL SPECTROGRAM
def extract_mel(filepath):
    audio, _ = librosa.load(filepath, sr=SR)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


# 🔹 MFCC + DELTA + DELTA-DELTA
def extract_mfcc(filepath):
    audio, _ = librosa.load(filepath, sr=SR)

    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, delta, delta2])  # (120, time)

    return combined


# 🔹 FIX LENGTH (VERY IMPORTANT FOR DL)
def pad_or_trim(array, target_length=128):
    if array.shape[1] < target_length:
        pad_width = target_length - array.shape[1]
        array = np.pad(array, ((0, 0), (0, pad_width)), mode='constant')
    else:
        array = array[:, :target_length]

    return array