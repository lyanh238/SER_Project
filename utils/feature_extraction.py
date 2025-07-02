import numpy as np
import librosa

SR = 22050
N_MFCC = 40
MAX_PAD_LENGTH = 173

def extract_features(audio_path, sr=SR, n_mfcc=N_MFCC, max_pad_length=MAX_PAD_LENGTH):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception:
        return None

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)

    features = np.vstack((mfccs, mfccs_delta, mfccs_delta2, rms, zcr))

    if features.shape[1] < max_pad_length:
        pad_width = max_pad_length - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif features.shape[1] > max_pad_length:
        features = features[:, :max_pad_length]

    return features.T