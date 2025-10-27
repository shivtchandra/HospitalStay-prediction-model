import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from joblib import dump
import os
import xgboost as xgb
import re

# --- Configuration ---
PROCESSED_DATA_PATH = "data/mimic_iv_processed/advanced_features.csv"
MODEL_OUTPUT_DIR = "models/advanced" # Save to a new sub-directory

# --- Clinical Thresholds ---
SAPS_II_HIGH_RISK_THRESHOLD = 40 # Patients with a score > 40 are high risk

def map_diagnosis_to_category(diagnosis):
    """Maps a detailed diagnosis string to a high-level clinical category."""
    if not isinstance(diagnosis, str): return 'Other'
    diagnosis = diagnosis.lower()
    if any(keyword in diagnosis for keyword in ['sepsis', 'infection', 'pneumonia']): return 'Infection'
    if any(keyword in diagnosis for keyword in ['fracture', 'injury', 'trauma']): return 'Injury'
    if any(keyword in diagnosis for keyword in ['cancer', 'carcinoma', 'leukemia']): return 'Cancer'
    if any(keyword in diagnosis for keyword in ['heart failure', 'cardiac', 'myocardial']): return 'Cardiovascular'
    if any(keyword in diagnosis for keyword in ['respiratory', 'copd', 'asthma']): return 'Respiratory'
    return 'Other'

def train_and_save_model(X, y, model, model_name):
    """A helper function to train and save a single model pipeline."""
    categorical_features = ['gender', 'admission_type', 'insurance', 'diagnosis_category']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # A simple preprocessor for demonstration
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough')

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    print(f"Training {model_name} on {len(X)} samples...")
    pipeline.fit(X, y)
    
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, f"{model_name}.joblib")
    dump(pipeline, model_path)
    print(f"{model_name} saved successfully to {model_path}\n")

def main():
    """Loads data and trains all necessary models for the advanced system."""
    df = pd.read_csv(PROCESSED_DATA_PATH).dropna(subset=['length_of_stay_days'])
    df['diagnosis_category'] = df['primary_diagnosis'].apply(map_diagnosis_to_category)
    
    features = [col for col in df.columns if col not in ['subject_id', 'hadm_id', 'length_of_stay_days', 'primary_diagnosis']]
    X = df[features]
    y = df['length_of_stay_days']

    # --- 1. Train the Generalist Model (on all data) ---
    generalist_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    train_and_save_model(X, y, generalist_model, "generalist_model")

    # --- 2. Train the Specialist Model (on high-risk data) ---
    specialist_df = df[df['saps_ii_score'] > SAPS_II_HIGH_RISK_THRESHOLD]
    if len(specialist_df) > 50: # Only train if we have enough high-risk samples
        X_spec = specialist_df[features]
        y_spec = specialist_df['length_of_stay_days']
        specialist_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        train_and_save_model(X_spec, y_spec, specialist_model, "specialist_model")
    else:
        print("WARNING: Not enough high-risk samples to train a specialist model.")

    # --- 3. Train the Quantile Models for Uncertainty (on all data) ---
    p10_model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.1, random_state=42, n_jobs=-1)
    train_and_save_model(X, y, p10_model, "los_p10_model")
    
    p50_model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, random_state=42, n_jobs=-1)
    train_and_save_model(X, y, p50_model, "los_p50_model")

    p90_model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.9, random_state=42, n_jobs=-1)
    train_and_save_model(X, y, p90_model, "los_p90_model")

if __name__ == "__main__":
    main()

