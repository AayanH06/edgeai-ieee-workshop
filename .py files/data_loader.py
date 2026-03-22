import pandas as pd
import numpy as np

WINDOW_SIZE = 100
STEP_SIZE = 50

def load_data(filepath):
    df = pd.read_csv(filepath)

    # columns : timestamp, x, y, z, label
    data = df[['x', 'y', 'z']].values
    labels = df['label'].values

    X = []
    y = []

    for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        window = data[i:i+WINDOW_SIZE]
        window_labels = labels[i:i+WINDOW_SIZE]

        # majority label in window
        label = max(set(window_labels), key=list(window_labels).count)

        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)