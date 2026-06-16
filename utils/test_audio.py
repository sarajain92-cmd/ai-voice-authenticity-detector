from utils.audio import load_audio
from utils.features import extract_features

audio, sr = load_audio("test.wav")  # ek audio file daalna folder me
features = extract_features(audio, sr)

print(features)
print(features.shape)