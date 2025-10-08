import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from joblib import dump
import os
import re
from sklearn.impute import SimpleImputer
import xgboost as xgb

# --- Configuration ---
PROCESSED_DATA_PATH = "data/mimic_iv_processed/length_of_stay_features.csv"
MODEL_OUTPUT_DIR = "models"
MODEL_NAME = "specialist_los_predictor.joblib" # Save to a different file

# --- Diagnosis to Category Mapping (must be identical to the main trainer) ---
def map_diagnosis_to_category(diagnosis):
    """Maps a detailed diagnosis string to a high-level category."""
    if not isinstance(diagnosis, str):
        return 'Other'
    diagnosis = diagnosis.lower()
    if any(term in diagnosis for term in ['sepsis', 'septicemia', 'bacteremia', 'infection', 'pneumonia']):
        return 'Infection'
    if any(term in diagnosis for term in ['failure', 'infarction', 'atrial fibrillation', 'hypertension', 'cardiac']):
        return 'Cardiovascular'
    if any(term in diagnosis for term in ['respiratory', 'copd', 'asthma', 'embolism']):
        return 'Respiratory'
    if any(term in diagnosis for term in ['fracture', 'dislocation', 'injury', 'trauma', 'hemorrhage']):
        return 'Injury'
    if any(term in diagnosis for term in ['cancer', 'carcinoma', 'leukemia', 'lymphoma', 'neoplasm']):
        return 'Cancer'
    return 'Other'

def train_specialist():
    """Trains a model ONLY on the high-risk 'hard cases'."""
    
    print("Loading processed data for SPECIALIST model...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    
    # --- THIS IS THE KEY STEP: Filter for the "hard cases" ---
    # We define a "hard case" as any patient where a vital sign abnormality was measured.
    specialist_df = df[df['age_hr_interaction'] > 0].copy()
    
    print(f"Filtered down to {len(specialist_df)} high-risk samples for specialist training.")

    # --- Defensive Check ---
    if len(specialist_df) < 100: # Need a minimum number of samples to train
        print("\nWARNING: Not enough high-risk samples with measured vital signs were found.")
        print("The specialist model cannot be trained. This is likely a data issue.")
        print("Please check the 'chartevents' table and the SQL query in 'feature_engineering.sql'.")
        return # Exit gracefully

    specialist_df['diagnosis_category'] = specialist_df['primary_diagnosis'].apply(map_diagnosis_to_category)

    # --- Feature Definition ---
    categorical_features = ['gender', 'admission_type', 'insurance', 'diagnosis_category']
    numerical_features = ['anchor_age', 'procedure_count', 'max_creatinine', 'min_hemoglobin', 'age_hr_interaction']
    
    X = specialist_df.drop(columns=['subject_id', 'hadm_id', 'length_of_stay_days', 'primary_diagnosis'])
    y = specialist_df['length_of_stay_days']

    # --- Preprocessing & Model Pipeline (Identical structure to main model) ---
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough')
    
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42, n_jobs=-1)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgboost_model)])
    
    # --- Training & Evaluation ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training specialist model on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)
    print("Specialist model training complete.\n")

    print("Evaluating specialist model performance...")
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print("\n--- Specialist Model Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} days")
    print("----------------------------------\n")

    # --- Save Model ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME)
    dump(pipeline, model_path)
    print(f"Trained specialist model saved successfully to: {model_path}")

if __name__ == "__main__":
    train_specialist()

