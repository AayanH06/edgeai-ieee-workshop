import numpy as np

def extract_features(X):
    features = []

    for window in X:
        x = window[:, 0]
        y = window[:, 1]
        z = window[:, 2]

        feat = [
            np.mean(x), np.std(x),
            np.mean(y), np.std(y),
            np.mean(z), np.std(z),

            np.max(x), np.min(x),
            np.max(y), np.min(y),
            np.max(z), np.min(z),
        ]

        features.append(feat)

    return np.array(features)