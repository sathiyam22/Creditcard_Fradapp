import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from data_preprocessing import prepare_data

CHUNK_SIZE = 30000
MODEL_PATH = 'lstm_model_chunked.h5'
DATA_PATH = 'D:/citi/CC/creditcard-sequential-modeling/data/creditcard.csv'
N_STEPS = 10
THRESHOLD = 0.3  # Tune this if needed

def evaluate_all_chunks():
    df = pd.read_csv(DATA_PATH)
    total_rows = df.shape[0]
    total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE

    model = load_model(MODEL_PATH)

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for chunk_index in range(total_chunks):
        start = chunk_index * CHUNK_SIZE
        end = min((chunk_index + 1) * CHUNK_SIZE, total_rows)
        chunk_df = df.iloc[start:end].copy()

        print(f"\n🧪 Evaluating chunk {chunk_index+1}/{total_chunks} → Rows {start}:{end}")

        # Generate sequences
        features, labels = prepare_data(chunk_df, N_STEPS)

        if len(features) == 0 or len(labels) == 0:
            print("⚠️ Skipping empty chunk...")
            continue

        # Split into test set only (80% of each chunk can be used as test)
        split_idx = int(len(features) * 0.8)
        X_test = features[split_idx:]
        y_test = labels[split_idx:]

        # Predict
        y_probs = model.predict(X_test)
        y_pred = (y_probs > THRESHOLD).astype(int)

        y_true_all.extend(y_test.flatten())
        y_pred_all.extend(y_pred.flatten())
        y_prob_all.extend(y_probs.flatten())

        print(f"✅ Done with chunk {chunk_index+1}: Samples={len(y_test)}")

    # === Final Evaluation ===
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)

    print("\n📊 Final Classification Report:")
    print(classification_report(y_true_all, y_pred_all))

    print("🧮 Final Confusion Matrix:")
    print(confusion_matrix(y_true_all, y_pred_all))

    auc = roc_auc_score(y_true_all, y_prob_all)
    print(f"📈 Final ROC-AUC Score: {auc:.4f}")

    # Plot ROC
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Overall ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_all_chunks()
