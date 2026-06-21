import librosa
import numpy as np

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta.T, axis=0)

    features = np.concatenate((mfcc_mean, delta_mean))  # 13 + 13 = 26

    return features