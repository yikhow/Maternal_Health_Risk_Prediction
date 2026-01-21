from src.preprocess import preprocess_data
from src.train import train_model
from deploy_app import launch_app

def main():
    # 1. Paths
    RAW_DATA = "data/Maternal_Health_Risk_Data_Set_Modified.csv"
    PROCESSED_DATA = "data/processed_data.csv"
    MODEL_PATH = "models/rf_model.joblib"

    # 2. Run Pipeline
    # Step 1: Cleaning
    data = preprocess_data(RAW_DATA, PROCESSED_DATA)
    
    # Step 2: Training & Evaluation
    train_model(data, model_save_path=MODEL_PATH)

    # Step 3: Deployment (Gradio)
    print("\n[SUCCESS] All stages completed. Opening Web Interface...")
    launch_app(MODEL_PATH)

if __name__ == "__main__":
    main()