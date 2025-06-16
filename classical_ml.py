import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from data_preprocessing import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def compute_per_class_accuracy(cm):
    """Returns accuracy per class from confusion matrix"""
    tn, fp, fn, tp = cm.ravel()
    fraud_acc = tp / (tp + fn + 1e-6)
    normal_acc = tn / (tn + fp + 1e-6)
    return fraud_acc, normal_acc


def get_models():
    """Returns a dictionary of initialized ML models"""
    return {
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }


def run_models(X_train, X_test, y_train, y_test):
    """Trains and evaluates models, returning trained model instances"""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    trained_models = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
            y_pred = (y_probs > 0.3).astype(int)
        else:
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        fraud_acc, normal_acc = compute_per_class_accuracy(cm)
        print(f"{name} → Fraud Accuracy: {fraud_acc:.2f} | Normal Accuracy: {normal_acc:.2f}")

    return trained_models


def main():
    df = pd.read_csv('D:/citi/CC/creditcard-sequential-modeling/data/creditcard.csv')
    chunk_size = 30000
    n_steps = 10
    total_rows = len(df)
    num_chunks = total_rows // chunk_size

    print(f"📦 Total rows: {total_rows} | Processing in {num_chunks} chunks of {chunk_size}")
    all_models = {}

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        print(f"\n🚀 Processing chunk {i+1}/{num_chunks} (rows {start}:{end})")
        chunk = df.iloc[start:end].copy()

        features, labels = prepare_data(chunk, n_steps)
        if features.shape[0] == 0:
            print("⚠️ Skipping empty chunk")
            continue

        X = features.reshape(features.shape[0], -1)
        y = labels.flatten()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ✅ Use SMOTE instead of RandomOverSampler
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        trained_models = run_models(X_train, X_test, y_train, y_test)
        all_models[f"chunk_{i+1}"] = trained_models

    # ✅ Save all models to one .pkl file
    with open("all_classical_models.pkl", "wb") as f:
        pickle.dump(all_models, f)

    print("\n✅ All models saved to 'all_classical_models.pkl'.")


if __name__ == "__main__":
    main()
