# --- Import required libraries ---
# Data manipulation
import pandas as pd
import numpy as np

# File and metadata management
from pathlib import Path
import json

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score, 
    roc_curve, 
    roc_auc_score, 
    classification_report, 
    )

# Model persistence
import joblib



# --- Helper Functions ---


def split_data(data: pd.DataFrame, target_column: str = 'RiskLevel'):
    """Split data into features and target, then into train and test sets."""
    X = data.drop(columns=[target_column], axis=1)
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
        )

    print(f"[INFO] Data split complete.")
    print(f"[INFO] Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def evaluate_and_plot(model, X_test, y_test, output_dir: str = "plots"):
    """Calculate metrics and save evaluation plots."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 1. Classification report
    print(f"\n{'='*18} Classification Report {'='*18}")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

    # 2. Overall evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("="*60)

    # 3. Plot ROC-AUC curve
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(
        fpr, tpr, color='darkorange', lw=2, 
        label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.4f})'
        )
    plt.plot(
        [0, 1], [0, 1], color='gray', lw=2, linestyle='--'
        )  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close() # Close to free up memory
    print(f"[INFO] ROC Curve saved to {output_dir}/roc_curve.png")



# --- Main training pipeline ---


def train_model(data: pd.DataFrame, model_save_path: str = "models/rf_model.joblib"):# Define features and target variable
    """Main training pipeline."""
    print(f"\n{'='*22} Training Stage {'='*22}")

    # 1. Split data
    X_train, X_test, y_train, y_test = split_data(data)

    # 2. Save feature schema
    feature_names = X_train.columns.tolist()
    save_dir = Path(model_save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    schema_path = save_dir / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(feature_names, f)
    print(f"[INFO] Feature schema saved to {schema_path}")

    # 3. Initialize and train model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    print(f"[INFO] Random Forest model trained successfully.")

    # 4. Evaluation
    evaluate_and_plot(rf_classifier, X_test, y_test)

    # 5. Save the trained model
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_classifier, model_save_path)
    print(f"[INFO] Model saved to: {model_save_path}")

    print(f"{'='*60}\n")
    return rf_classifier

if __name__ == "__main__":
     # Internal test run
    try:
        sample_data = pd.read_csv("data/processed_data.csv")
        train_model(sample_data)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")