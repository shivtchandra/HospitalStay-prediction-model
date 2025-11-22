import pandas as pd
from joblib import load
import os
import re

# --- Configuration ---
MODEL_DIR = "models"
MODEL_NAME = "readmission_risk_predictor.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# --- THE FINAL FIX: A Smart Clinical Threshold ---
# Instead of a default 50% cutoff, we'll use a lower threshold to
# better identify at-risk patients, balancing precision and recall.
PREDICTION_THRESHOLD = 0.35

def map_diagnosis_to_category(diagnosis):
    """Maps a detailed diagnosis string to a high-level clinical category."""
    if not isinstance(diagnosis, str):
        return 'Other'
    diagnosis = diagnosis.lower()
    if any(keyword in diagnosis for keyword in ['sepsis', 'septicemia', 'bacteremia', 'infection', 'pneumonia']):
        return 'Infection'
    if any(keyword in diagnosis for keyword in ['fracture', 'dislocation', 'injury', 'trauma']):
        return 'Injury'
    if any(keyword in diagnosis for keyword in ['cancer', 'carcinoma', 'leukemia', 'lymphoma', 'metastasis']):
        return 'Cancer'
    if any(keyword in diagnosis for keyword in ['heart failure', 'cardiac', 'atrial', 'myocardial infarction', 'coronary']):
        return 'Cardiovascular'
    if any(keyword in diagnosis for keyword in ['respiratory', 'copd', 'asthma', 'pulmonary']):
        return 'Respiratory'
    if any(keyword in diagnosis for keyword in ['renal', 'kidney']):
        return 'Renal'
    if any(keyword in diagnosis for keyword in ['stroke', 'cerebral', 'hemorrhage']):
        return 'Neurological'
    return 'Other'

def predict_readmission():
    """Loads the classifier and makes predictions on new patient scenarios."""
    try:
        pipeline = load(MODEL_PATH)
        print("Readmission risk model loaded successfully.\n")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # Define test cases
    patient_data = {
        'high_risk_patient': {
            'gender': 'M', 'anchor_age': 75, 'admission_type': 'EMERGENCY',
            'insurance': 'Medicare', 'primary_diagnosis': 'Congestive Heart Failure',
            'procedure_count': 2, 'max_creatinine': 2.5, 'min_hemoglobin': 9.0,
            'age_hr_interaction': 2250.0
        },
        'low_risk_patient': {
            'gender': 'F', 'anchor_age': 60, 'admission_type': 'ELECTIVE',
            'insurance': 'Other', 'primary_diagnosis': 'Cholecystitis',
            'procedure_count': 1, 'max_creatinine': 1.0, 'min_hemoglobin': 13.5,
            'age_hr_interaction': 300.0
        }
    }
    
    new_patients_df = pd.DataFrame.from_dict(patient_data, orient='index')
    new_patients_df['diagnosis_category'] = new_patients_df['primary_diagnosis'].apply(map_diagnosis_to_category)

    # --- Use the Smart Threshold for Prediction ---
    pred_proba = pipeline.predict_proba(new_patients_df)[:, 1]
    
    # Apply the custom threshold to the probabilities
    final_predictions = ["Likely Readmit" if prob >= PREDICTION_THRESHOLD else "No Readmit" for prob in pred_proba]
    
    # --- Display Results ---
    results_df = new_patients_df[['primary_diagnosis']].copy()
    results_df['predicted_outcome'] = final_predictions
    results_df['readmission_probability'] = [f"{prob:.0%}" for prob in pred_proba]

    print("\n--- Readmission Risk Prediction Results ---")
    print(results_df)
    print("-------------------------------------------\n")

if __name__ == "__main__":
    predict_readmission()

