import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Rename columns for consistency if needed
    df = df.rename(columns={'Class': 'Fraudulent'})
    df = df.sort_values(by='Time')
    return df

from tqdm import tqdm

def create_sequences(df, n_steps):
    sequences = []
    labels = []
    feature_cols = [col for col in df.columns if col not in ['Time', 'Fraudulent']]
    for i in tqdm(range(len(df) - n_steps), desc="⏳ Generating sequences"):
        seq = df.iloc[i:i + n_steps]
        label = df.iloc[i + n_steps]['Fraudulent']
        sequences.append(seq[feature_cols])
        labels.append(label)
    return sequences, np.array(labels)


def generate_features(sequences):
    scaler = StandardScaler()
    features = []

    for seq in sequences:
        seq_scaled = scaler.fit_transform(seq)
        features.append(seq_scaled)

    return np.array(features)

def prepare_data(df, n_steps):
    df = preprocess_data(df)
    sequences, labels = create_sequences(df, n_steps)
    features = generate_features(sequences)
    return features, labels
