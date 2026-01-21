# Import required libraries
import pandas as pd
import numpy as np
from pathlib import Path

# --- Helper Functions ---

def load_data(file_path: str) -> pd.DataFrame:
    """1. Load data"""
    data_path = Path(file_path)
    if not data_path.exists():
        raise FileNotFoundError(f"The data file does not exist: {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"[INFO] Initial dataset loaded. Shape: {data.shape}")
    return data

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """2. Remove duplicates"""
    initial_len = len(data)
    data = data.drop_duplicates(keep='first')
    if len(data) < initial_len:
        print(f"[INFO] Removed {initial_len - len(data)} duplicate rows.")
    return data

def remove_pii_columns(data: pd.DataFrame) -> pd.DataFrame:
    """3. Remove PII columns"""
    if 'CitizenID' in data.columns:
        data = data.drop(columns=['CitizenID'])
        print("[INFO] Removed PII column: 'CitizenID'.")
    else:
        print("[WARNING] PII column 'CitizenID' not found in the dataset.")
    return data

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """4. Handle missing values with median imputation (Numerical)"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    imputed_cols = []
    
    for col in numeric_cols:
        if data[col].isnull().any():
            median_value = data[col].median()
            data[col] = data[col].fillna(median_value)
            imputed_cols.append(col)
        
    if imputed_cols:    
        print(f"[INFO] Missing values imputed in: {', '.join(imputed_cols)}")
    else:
        print("[INFO] No missing values found in numerical columns.")
    
    return data

def encode_labels(data: pd.DataFrame) -> pd.DataFrame:
    """5. Label Encoding (RiskLevel)"""
    if 'RiskLevel' in data.columns:
        # Standardize categorical variable
        data['RiskLevel'] = data['RiskLevel'].astype(str).str.strip().str.lower()
        
        # Define mapping
        risk_mapping = {'low risk': 0, 'high risk': 1}
        data['RiskLevel'] = data['RiskLevel'].map(risk_mapping)
        
        print("[INFO] Applied label encoding to 'RiskLevel'.")
    else:
        print("[WARNING] 'RiskLevel' column not found in the dataset.")
    
    return data

# --- Main preprocessing function ---

def preprocess_data(input_path: str, output_path: str = None):
    
    print(f"\n{'='*20} Preprocessing Stage {'='*20}")
    
    data = load_data(input_path)

    # Execute sub-functions
    data = (
        data.pipe(remove_duplicates)
            .pipe(remove_pii_columns)
            .pipe(handle_missing_values)
            .pipe(encode_labels)
            )

    # 6. Save and Return
    print(f"[INFO] Preprocessing completed. Final dataset shape: {data.shape}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"[INFO] Preprocessed data saved to: {output_path}")

    return data

if __name__ == "__main__":
    # Internal test run
    SAMPLE_INPUT = "data/Maternal_Health_Risk_Data_Set_Modified.csv"
    SAMPLE_OUTPUT = "data/processed_data.csv"
    try:
        processed_df = preprocess_data(SAMPLE_INPUT, SAMPLE_OUTPUT)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")