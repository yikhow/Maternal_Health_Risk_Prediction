import gradio as gr
import json
from pathlib import Path
from src.predict import make_prediction

def launch_app(model_path="models/rf_model.joblib"):
    """
    Define and launch the Gradio web interface.
    """
    print(f"\n{'='*18} Starting Web Application {'='*18}")
    
    # 1. Load Feature Schema to ensure UI matches the model
    schema_path = Path(model_path).parent / "feature_schema.json"
    with open(schema_path, "r") as f:
        schema_data = json.load(f)
    
    if isinstance(schema_data, dict) and "features" in schema_data:
        trained_feature_names = schema_data["features"]
    else:
        trained_feature_names = schema_data

    print(f"[INFO] Successfully loaded schema with {len(trained_feature_names)} features.")

    # 2. Define UI Configs with internal names for validation
    ui_configs = [
        {"internal_name": "Age", "label": "Age", "value": 29},
        {"internal_name": "SystolicBP", "label": "Systolic Blood Pressure", "value": 90},
        {"internal_name": "DiastolicBP", "label": "Diastolic Blood Pressure", "value": 70},
        {"internal_name": "BS", "label": "Blood Sugar level (mmol/L)", "value": 8.0},
        {"internal_name": "BodyTemp", "label": "Body Temperature (°F)", "value": 100.0},
        {"internal_name": "HeartRate", "label": "Heart Rate (bpm)", "value": 80.0}
    ]

    current_ui_names = [conf["internal_name"] for conf in ui_configs]
    
    if current_ui_names != trained_feature_names:
        print("[ERROR] UI sequence does NOT match the trained model schema!")
        print(f"Expected: {trained_feature_names}")
        print(f"Actual UI: {current_ui_names}")
        raise ValueError("Feature sequence mismatch. Check your ui_configs in deploy_app.py.")
    else:
        print("[INFO] Sequence validation passed. UI is synchronized with the model.")

    # 3. Define the Prediction Wrapper
    def predict_interface(*args):
        """Wrapper to bridge Gradio inputs and our prediction logic."""
        # Convert tuple of inputs from Gradio into a list
        input_list = list(args)
        return make_prediction(model_path, input_list)

    # 4. Build Gradio Interface
    interface = gr.Interface(
        fn=predict_interface,
        inputs=[
                gr.Number(label=conf["label"], value=conf["value"]) 
                for conf in ui_configs
            ],        
        outputs=gr.Textbox(label="Maternal Health Risk Prediction"),
        title="Maternal Health Risk Assessment System",
        description=(
            "This system uses a Random Forest model to assess health risks. "
            "Please enter the clinical parameters below."
        ),
        examples=[
            [25, 120, 80, 7.5, 98.6, 70],
            [35, 140, 90, 13.0, 98.6, 85]
        ]
        )

    print("[INFO] Gradio interface is ready. Launching...")
    interface.launch(

        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Soft())

if __name__ == "__main__":
    MODEL_FILE = "models/rf_model.joblib"
    try:
        launch_app(MODEL_FILE)
    except Exception as e:
        print(f"[ERROR] Failed to start application: {e}")  