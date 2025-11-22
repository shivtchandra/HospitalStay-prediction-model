import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from joblib import dump
import os
import xgboost as xgb
import re

# --- Configuration ---
PROCESSED_DATA_PATH = "data/mimic_iv_processed/readmission_features.csv"
MODEL_OUTPUT_DIR = "models"
MODEL_NAME = "readmission_risk_predictor.joblib"

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

def train_readmission_model():
    """Loads data, trains a classifier for readmission risk, and saves it."""
    
    print("Loading processed data for readmission model...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    df = df.dropna(subset=['primary_diagnosis']) # Ensure diagnosis is not null

    print("Mapping diagnoses to categories...")
    df['diagnosis_category'] = df['primary_diagnosis'].apply(map_diagnosis_to_category)
    
    # --- Feature and Target Definition ---
    categorical_features = ['gender', 'admission_type', 'insurance', 'diagnosis_category']
    numerical_features = ['anchor_age', 'procedure_count', 'max_creatinine', 'min_hemoglobin', 'age_hr_interaction']
    
    X = df[categorical_features + numerical_features]
    y = df['was_readmitted_in_30_days']

    # --- THE FIX: Class Imbalance Handling ---
    # Calculate the scale_pos_weight for the XGBClassifier.
    # This is the ratio of negative class samples to positive class samples.
    neg_count = y.value_counts()[0]
    pos_count = y.value_counts()[1]
    scale_pos_weight = neg_count / pos_count
    print(f"\nClass imbalance detected. Using scale_pos_weight: {scale_pos_weight:.2f}\n")


    # --- Preprocessing ---
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- Model Training ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training classification model on {len(X_train)} samples...")

    # Define the XGBoost Classifier with the scale_pos_weight parameter
    classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight, # Apply the fix here
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', classifier)])

    pipeline.fit(X_train, y_train)
    print("Model training complete.\n")

    # --- Evaluation ---
    print("Evaluating classifier performance...")
    preds = pipeline.predict(X_test)
    pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Classification Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
    print(f"Precision: {precision_score(y_test, preds):.2f}")
    print(f"Recall: {recall_score(y_test, preds):.2f}")
    print(f"AUC: {roc_auc_score(y_test, pred_proba):.2f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("--------------------------------------\n")

    # --- Save Model ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME)
    dump(pipeline, model_path)
    print(f"Trained classifier pipeline saved to: {model_path}")

if __name__ == "__main__":
    train_readmission_model()

