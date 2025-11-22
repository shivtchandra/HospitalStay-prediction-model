import pandas as pd
from joblib import load
import os
import re

# --- Configuration ---
MODEL_DIR = "models"
GENERALIST_MODEL_NAME = "length_of_stay_predictor.joblib"
SPECIALIST_MODEL_NAME = "specialist_los_predictor.joblib"

# --- Diagnosis Mapping (must be identical to trainers) ---
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

def main():
    # --- Load Both Models ---
    print("Loading trained model pipelines...")
    try:
        generalist_pipeline = load(os.path.join(MODEL_DIR, GENERALIST_MODEL_NAME))
        specialist_pipeline = load(os.path.join(MODEL_DIR, SPECIALIST_MODEL_NAME))
        print("Both models loaded successfully.\n")
    except FileNotFoundError:
        print(f"ERROR: Could not find one or both model files in the '{MODEL_DIR}' directory.")
        print("Please ensure you have successfully run both 'train_model.py' and 'train_specialist_model.py' first.")
        return

    # --- Define New Patient Scenarios ---
    patient_data = {
        'patient_1': { # High-risk patient with MEASURED abnormal heart rate
            'primary_diagnosis': 'Pneumonia with sepsis',
            'gender': 'M', 'anchor_age': 70, 'admission_type': 'EMERGENCY', 'insurance': 'Medicare',
            'procedure_count': 1, 'max_creatinine': 1.2, 'min_hemoglobin': 11.5,
            'age_hr_interaction': 2100.0 # High value (70 * abs(105-75))
        },
        'patient_2': { # Similar patient but with NO measured abnormality
            'primary_diagnosis': 'Pneumonia with sepsis',
            'gender': 'M', 'anchor_age': 70, 'admission_type': 'EMERGENCY', 'insurance': 'Medicare',
            'procedure_count': 1, 'max_creatinine': 1.2, 'min_hemoglobin': 11.5,
            'age_hr_interaction': 0.0 # Zero indicates no measured abnormality
        },
        'patient_3': { # Standard elective case, some measured abnormality
            'primary_diagnosis': 'Degenerative disc disease',
            'gender': 'F', 'anchor_age': 55, 'admission_type': 'ELECTIVE', 'insurance': 'Other',
            'procedure_count': 2, 'max_creatinine': 0.8, 'min_hemoglobin': 13.0,
            'age_hr_interaction': 275.0 # Lower value (55 * abs(80-75))
        }
    }
    new_patients_df = pd.DataFrame.from_dict(patient_data, orient='index')

    # --- Feature Engineering for Prediction (must match training) ---
    new_patients_df['diagnosis_category'] = new_patients_df['primary_diagnosis'].apply(map_diagnosis_to_category)

    # --- Triage and Prediction Logic ---
    print("Making predictions using the two-model system...")
    final_predictions = {}
    for patient_id, patient_info in new_patients_df.iterrows():
        patient_series_df = pd.DataFrame([patient_info])
        
        # TRIAGE RULE: If the interaction term is significant, call the specialist.
        # Otherwise, trust the generalist.
        if patient_info['age_hr_interaction'] > 1000: # A reasonable threshold for a significant abnormality
            print(f"  - {patient_id}: High risk detected. Calling SPECIALIST model.")
            prediction = specialist_pipeline.predict(patient_series_df)[0]
        else:
            print(f"  - {patient_id}: Standard case. Using GENERALIST model.")
            prediction = generalist_pipeline.predict(patient_series_df)[0]
        
        final_predictions[patient_id] = f"{prediction:.2f}"

    # --- Display Results ---
    results_df = new_patients_df.copy()
    results_df['predicted_length_of_stay_days'] = pd.Series(final_predictions)
    
    print("\n--- Final Prediction Results ---")
    print(results_df.drop(columns=['primary_diagnosis'])) # Drop original diagnosis for cleaner output
    print("----------------------------------\n")
    print("This final test uses our two-model system to make more nuanced predictions.")

if __name__ == "__main__":
    main()

