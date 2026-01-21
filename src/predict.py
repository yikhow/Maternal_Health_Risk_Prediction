import joblib
import numpy as np
import json
from pathlib import Path

def make_prediction(model_path: str, feature_values: list):
    """
    Load model and schema, validate inputs, and return a prediction label.
    """
    model_path = Path(model_path)
    schema_path = model_path.parent / "feature_schema.json"

    # 1. Check if model and schema files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found at: {schema_path}")

    # 2. Load the trained model and the feature schema
    model = joblib.load(model_path)
    with open(schema_path, "r") as f:
        schema_data = json.load(f)

    if isinstance(schema_data, dict) and "features" in schema_data:
        feature_names = schema_data["features"]
    else:
        feature_names = schema_data

    # 3. Validation: Ensure the number of inputs matches the schema
    if len(feature_values) != len(feature_names):
        raise ValueError(
            f"Input features mismatch. Expected {len(feature_names)} features "
            f"({', '.join(feature_names)}), but got {len(feature_values)}."
        )

    # 4. Prepare data for prediction
    input_data = np.array(feature_values, dtype=float).reshape(1, -1)
    
    # 5. Execute Prediction
    try:
        prediction = model.predict(input_data)[0]
        # Map numerical result back to human-readable label
        result = "High Risk" if prediction == 1 else "Low Risk"
        return result
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return "Error in prediction"

if __name__ == "__main__":
    # Internal testing
    MODEL_FILE = "models/rf_model.joblib"
    # Example values: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate
    TEST_INPUT = (5, 120, 80, 7.5, 98.6, 70)
    
    try:
        res = make_prediction(MODEL_FILE, TEST_INPUT)
        print(f"[TEST] Input: {TEST_INPUT}")
        print(f"[TEST] Result: {res}")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")