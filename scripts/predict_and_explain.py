import pandas as pd
from joblib import load
import os
import shap
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
MODEL_DIR = "models/advanced"
SAPS_II_HIGH_RISK_THRESHOLD = 40 # Patients with a score > 40 are considered high-risk

def load_all_models():
    """Loads all trained models needed for prediction."""
    models = {}
    model_names = ["generalist_model", "specialist_model", "los_p10_model", "los_p50_model", "los_p90_model"]
    for name in model_names:
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        try:
            models[name] = load(path)
        except FileNotFoundError:
            print(f"Warning: Model file '{path}' not found. Some functionality may be limited.")
            models[name] = None
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

def get_feature_names(pipeline, input_df):
    """Extracts the final feature names after preprocessing."""
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        num_features = [col for col in input_df.columns if col in preprocessor.named_transformers_['num'].feature_names_in_]
        cat_features_raw = [col for col in input_df.columns if col in preprocessor.named_transformers_['cat'].feature_names_in_]
        ohe_features = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_features_raw))
        return num_features + ohe_features
    except Exception:
        return [f"Feature {i}" for i in range(input_df.shape[1])]

def predict_and_explain(patient_name, patient_data, models):
    """Performs the full prediction and explanation for a single patient."""
    print(f"\n--- Running Prediction for: {patient_name} ---")
    patient_df = pd.DataFrame([patient_data])
    patient_df['diagnosis_category'] = patient_df['primary_diagnosis'].apply(map_diagnosis_to_category)

    # 1. Intelligent Routing
    saps_score = patient_df['saps_ii_score'].iloc[0]
    if saps_score > SAPS_II_HIGH_RISK_THRESHOLD and models["specialist_model"]:
        print(f"High risk detected (SAPS-II Score: {saps_score}). Calling SPECIALIST model.")
        prediction_model = models["specialist_model"]
    else:
        print(f"Standard case (SAPS-II Score: {saps_score}). Using GENERALIST model.")
        prediction_model = models["generalist_model"]

    # 2. Predict with Uncertainty
    p10 = models["los_p10_model"].predict(patient_df)[0]
    p50_median = models["los_p50_model"].predict(patient_df)[0]
    p90 = models["los_p90_model"].predict(patient_df)[0]

    print(f"Predicted Median LOS: {p50_median:.2f} days")
    print(f"80% Confidence Range: {p10:.2f} - {p90:.2f} days")

    # 3. Explain the Prediction (only for the high-risk case for brevity)
    if saps_score > SAPS_II_HIGH_RISK_THRESHOLD:
        print("\n--- Generating Prediction Explanation ---")
        preprocessor = prediction_model.named_steps['preprocessor']
        regressor = prediction_model.named_steps['regressor']
        
        feature_names = get_feature_names(prediction_model, patient_df.drop(columns=['primary_diagnosis']))
        patient_transformed = preprocessor.transform(patient_df.drop(columns=['primary_diagnosis']))
        
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(patient_transformed)
        
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                             base_values=explainer.expected_value, 
                                             data=patient_transformed[0], 
                                             feature_names=feature_names),
                           show=False)
        
        plot_filename = f"explanation_{patient_name.lower().replace(' ', '_')}.png"
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        print(f"SHAP explanation plot saved to '{plot_filename}'.")
    print("-------------------------------------------------")


def main():
    """Main function to load models and run predictions on multiple test cases."""
    models = load_all_models()
    if not all(models.values()):
        print("\nERROR: Not all models could be loaded. Please run the training script first.")
        return

    # --- Define New Patient Scenarios ---
    test_cases = {
        "High_Risk_ICU_Patient": {
            'gender': 'F', 'anchor_age': 82, 'admission_type': 'EMERGENCY',
            'insurance': 'Medicare', 'primary_diagnosis': 'Septic shock',
            'procedure_count': 3, 'max_creatinine': 3.1, 'min_hemoglobin': 8.2,
            'age_hr_interaction': 3280, 'saps_ii_score': 55
        },
        "Moderate_Risk_Ward_Patient": {
            'gender': 'M', 'anchor_age': 58, 'admission_type': 'EMERGENCY',
            'insurance': 'Other', 'primary_diagnosis': 'Pneumonia',
            'procedure_count': 0, 'max_creatinine': 1.4, 'min_hemoglobin': 10.5,
            'age_hr_interaction': 1160, 'saps_ii_score': 35
        },
        "Low_Risk_Elective_Patient": {
            'gender': 'F', 'anchor_age': 45, 'admission_type': 'ELECTIVE',
            'insurance': 'Other', 'primary_diagnosis': 'Knee replacement',
            'procedure_count': 1, 'max_creatinine': 0.8, 'min_hemoglobin': 13.2,
            'age_hr_interaction': 225, 'saps_ii_score': 15
        },
        "Complex_Geriatric_Patient": {
            'gender': 'F', 'anchor_age': 88, 'admission_type': 'URGENT',
            'insurance': 'Medicare', 'primary_diagnosis': 'Hip fracture',
            'procedure_count': 1, 'max_creatinine': 1.1, 'min_hemoglobin': 9.8,
            'age_hr_interaction': 880, 'saps_ii_score': 28
        }
    }
    
    for name, data in test_cases.items():
        predict_and_explain(name, data, models)

if __name__ == "__main__":
    main()

