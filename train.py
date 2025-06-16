import pandas as pd
import numpy as np
from data_preprocessing import prepare_data
from lstm_model import create_lstm_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping
import time
import os

CHUNKSIZE = 30000
N_STEPS = 10
EPOCHS_PER_CHUNK = 1
BATCH_SIZE = 32
CSV_PATH = 'D:/citi/CC/creditcard-sequential-modeling/data/creditcard.csv'

def train_on_chunk(model, chunk_df, chunk_num):
    features, labels = prepare_data(chunk_df, n_steps=N_STEPS)
    if len(features) == 0:
        print(f"⚠️ Chunk {chunk_num}: No valid sequences, skipping.")
        return model

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)

    # === Optional SMOTE (applied to 2D flattened data) ===
    print("⚖️ Applying SMOTE to balance classes...")
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    sm = SMOTE(random_state=42, sampling_strategy='auto')

    X_resampled, y_resampled = sm.fit_resample(X_train_2d, y_train)
    X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
    print(f"✅ Resampled to shape: {X_resampled.shape}")

    # Compute class weights
    classes = np.unique(y_resampled)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_resampled)
    class_weight_dict = dict(zip(classes, class_weights))

    # Train
    print(f"🧠 Training on chunk {chunk_num}...")
    early_stop = EarlyStopping(patience=3, restore_best_weights=True, verbose=0)
    model.fit(
        X_resampled, y_resampled,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_PER_CHUNK,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )
    return model

def main():
    print("🚀 Starting chunk-based LSTM training...")
    start_time = time.time()

    model = None
    for i, chunk_df in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNKSIZE)):
        print(f"\n📦 Processing chunk {i+1}")
        chunk_df.dropna(inplace=True)

        # If first chunk, create model with correct shape
        if model is None:
            features, labels = prepare_data(chunk_df, n_steps=N_STEPS)
            if len(features) == 0:
                print("❌ First chunk has no valid sequences, aborting.")
                return
            n_steps, n_features = features.shape[1], features.shape[2]
            model = create_lstm_model(input_shape=(n_steps, n_features))
            print(f"✅ Model initialized with input shape: ({n_steps}, {n_features})")

        # Train on current chunk
        model = train_on_chunk(model, chunk_df, i + 1)

    # Save model
    model.save("lstm_model_chunked.h5")
    print("💾 Final model saved as 'lstm_model_chunked.h5'")
    print(f"✅ Training complete in {(time.time() - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()
