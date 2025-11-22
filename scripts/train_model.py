import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from joblib import dump
import os
from sklearn.impute import SimpleImputer
import xgboost as xgb
import numpy as np

# --- Configuration ---
PROCESSED_DATA_PATH = "data/mimic_iv_processed/length_of_stay_features.csv"
MODEL_OUTPUT_DIR = "models"
MODEL_NAME = "length_of_stay_predictor.joblib"

# --- NEW: Function to map detailed diagnoses to broader categories ---
def map_diagnosis_to_category(diagnosis):
    """Maps a detailed diagnosis string to a high-level category."""
    if pd.isna(diagnosis):
        return 'Unknown'
    diagnosis = diagnosis.lower()
    if 'sepsis' in diagnosis or 'septicemia' in diagnosis: return 'Infection'
    if 'fracture' in diagnosis: return 'Injury'
    if 'failure' in diagnosis and ('heart' in diagnosis or 'cardiac' in diagnosis): return 'Cardiovascular'
    if 'pneumonia' in diagnosis or 'copd' in diagnosis or 'respiratory' in diagnosis: return 'Respiratory'
    if 'cancer' in diagnosis or 'neoplasm' in diagnosis or 'carcinoma' in diagnosis: return 'Cancer'
    if 'stroke' in diagnosis or 'cerebrovascular' in diagnosis: return 'Neurological'
    if 'diabetes' in diagnosis: return 'Endocrine'
    if 'kidney' in diagnosis or 'renal' in diagnosis: return 'Renal'
    if 'gastrointestinal' in diagnosis or 'intestinal' in diagnosis: return 'Gastrointestinal'
    if 'replacement' in diagnosis and ('knee' in diagnosis or 'hip' in diagnosis): return 'Orthopedic'
    return 'Other'


def train():
    """Loads data, trains the model, evaluates it, and saves the final pipeline."""
    
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df.replace([np.inf, -np.inf], pd.NA, inplace=True)

    # --- NEW: Feature Engineering Step ---
    print("Mapping diagnoses to categories...")
    df['diagnosis_category'] = df['primary_diagnosis'].apply(map_diagnosis_to_category)
    
    # --- Feature Definition ---
    # We now use our new, simplified category instead of the thousands of detailed codes
    categorical_features = ['gender', 'admission_type', 'insurance', 'diagnosis_category']
    numerical_features = ['anchor_age', 'procedure_count', 'max_creatinine', 'min_hemoglobin', 'age_hr_interaction']
    
    print(f"Using categorical features: {categorical_features}")
    print(f"Using numerical features: {numerical_features}\n")

    # Drop the original detailed diagnosis column as it's no longer needed
    X = df.drop(columns=['subject_id', 'hadm_id', 'length_of_stay_days', 'primary_diagnosis'])
    y = df['length_of_stay_days']

    # --- Preprocessing Pipeline ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough')

    # --- Model Training ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training model on {len(X_train)} samples...")
    xgboost_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=8,
        colsample_bytree=0.7, subsample=0.7, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgboost_model)])
    pipeline.fit(X_train, y_train)
    print("Model training complete.\n")

    # --- Model Evaluation ---
    print("Evaluating model performance on the test set...")
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print("\n--- Model Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} days")
    print("-------------------------\n")
    
    # --- Feature Importance Analysis ---
    print("--- Feature Importance ---")
    regressor = pipeline.named_steps['regressor']
    ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)
    importances = regressor.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    print(feature_importance_df.head(10))
    print("--------------------------\n")

    # --- Save the Model ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME)
    dump(pipeline, model_path)
    print(f"Trained model pipeline saved successfully to: {model_path}")

if __name__ == "__main__":
    train()

