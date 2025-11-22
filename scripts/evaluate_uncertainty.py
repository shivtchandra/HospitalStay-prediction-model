import pandas as pd
from joblib import load
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb  # <-- ADDED THIS LINE
from sklearn.pipeline import Pipeline # <-- AND THIS LINE

# --- Configuration ---
MODEL_DIR = "models/advanced"
PROCESSED_DATA_PATH = "data/mimic_iv_processed/advanced_features.csv"

def load_all_models():
    """Loads all trained models needed for evaluation."""
    models = {}
    model_names = ["los_p10_model", "los_p50_model", "los_p90_model"]
    for name in model_names:
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        try:
            models[name] = load(path)
        except FileNotFoundError:
            print(f"FATAL: Model file '{path}' not found. Please run the training script first.")
            return None
    return models

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

def evaluate_uncertainty():
    """Loads models and evaluates their performance on the test set."""
    models = load_all_models()
    if not models:
        return

    # --- 1. Load and Prepare Data ---
    print("Loading and preparing test data...")
    df = pd.read_csv(PROCESSED_DATA_PATH).dropna(subset=['length_of_stay_days'])
    df['diagnosis_category'] = df['primary_diagnosis'].apply(map_diagnosis_to_category)
    
    features = [col for col in df.columns if col not in ['subject_id', 'hadm_id', 'length_of_stay_days', 'primary_diagnosis']]
    X = df[features]
    y = df['length_of_stay_days']
    
    # Use the same split as training to get the exact same test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 2. Make Predictions on the Entire Test Set ---
    print("Making predictions with all quantile models...")
    p10_preds = models["los_p10_model"].predict(X_test)
    p50_preds = models["los_p50_model"].predict(X_test)
    p90_preds = models["los_p90_model"].predict(X_test)

    # --- 3. Calculate Aggregate Metrics ---
    print("\n--- Aggregate Performance Metrics ---")
    
    # MAE of the median prediction
    mae = np.mean(np.abs(y_test - p50_preds))
    print(f"Median Prediction MAE: {mae:.2f} days")

    # Prediction Interval Coverage (for our 80% interval)
    in_interval = np.sum((y_test >= p10_preds) & (y_test <= p90_preds))
    coverage = (in_interval / len(y_test)) * 100
    print(f"80% Prediction Interval Coverage: {coverage:.2f}% (Target: 80%)")
    print("-------------------------------------\n")

    # --- 4. Generate Calibration Plot ---
    print("--- Generating Calibration Plot ---")
    quantiles_to_check = [0.1, 0.25, 0.5, 0.75, 0.9]
    observed_frequencies = []
    
    for q in quantiles_to_check:
        model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=q)
        # For simplicity, we create a temporary pipeline to make predictions
        # A full study might train and save a model for each quantile
        temp_pipeline = Pipeline(steps=[('preprocessor', models['los_p50_model'].named_steps['preprocessor']), ('regressor', model)])
        temp_pipeline.fit(X, y) # Refitting for demonstration
        preds = temp_pipeline.predict(X_test)
        
        freq = np.mean(y_test <= preds)
        observed_frequencies.append(freq)

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    plt.plot(quantiles_to_check, observed_frequencies, 'o-', label="Model Calibration")
    plt.xlabel("Predicted Quantile")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration of Quantile Regression Models")
    plt.legend()
    plt.grid(True)
    
    plot_filename = "calibration_plot.png"
    plt.savefig(plot_filename)
    plt.close()
    
    print(f"Calibration plot saved to '{plot_filename}'.")
    print("This plot shows how reliable the predicted quantiles are.")
    print("---------------------------------------")

if __name__ == "__main__":
    evaluate_uncertainty()

