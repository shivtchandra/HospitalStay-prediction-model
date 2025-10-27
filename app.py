#!/usr/bin/env python3
"""
app.py

Flask API for single-patient LOS + cost prediction.

Expect JSON POST to /predict_impact with required features:
 - anchor_age, gender, admission_type, insurance,
 - primary_diagnosis, procedure_count, max_creatinine,
 - min_hemoglobin, saps_ii_score
Optional:
 - avg_heart_rate
 - average_daily_patient_cost
 - currency
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load

# --- App init ---
app = Flask(__name__)
app.config["DEBUG"] = True

# Configure CORS to allow your frontend origins — adjust if needed
CORS(app, resources={
    r"/predict_impact": {
        "origins": [
            "http://localhost:3001",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173"
        ],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# --- Load hospital assumptions from config/hospital_assumptions.py ---
config_dir = os.path.join(os.path.dirname(__file__), 'config')
if config_dir not in sys.path:
    sys.path.append(config_dir)

try:
    from hospital_assumptions import (
        AVERAGE_DAILY_PATIENT_COST,
        SAPS_II_RISK_THRESHOLD
    )
    # Optional: currency constant
    try:
        from hospital_assumptions import AVERAGE_DAILY_PATIENT_COST_CURRENCY
    except ImportError:
        AVERAGE_DAILY_PATIENT_COST_CURRENCY = None
    print("Loaded hospital_assumptions.")
except Exception as e:
    print("ERROR: Could not import hospital_assumptions from config/ - please create it.")
    print("Expected variables: AVERAGE_DAILY_PATIENT_COST, SAPS_II_RISK_THRESHOLD (optionally AVERAGE_DAILY_PATIENT_COST_CURRENCY).")
    print("Exception:", e)
    # Provide defaults so API can still run for local dev (you can change as appropriate)
    AVERAGE_DAILY_PATIENT_COST = 1000.0
    SAPS_II_RISK_THRESHOLD = 29
    AVERAGE_DAILY_PATIENT_COST_CURRENCY = "USD"

# --- Models directory & filenames (adjust filenames if you use different names) ---
ADVANCED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "advanced")

MODEL_PATHS = {
    "generalist_p10": os.path.join(ADVANCED_MODEL_DIR, "los_p10_model.joblib"),
    "generalist_p50": os.path.join(ADVANCED_MODEL_DIR, "los_p50_model.joblib"),
    "generalist_p90": os.path.join(ADVANCED_MODEL_DIR, "los_p90_model.joblib"),
    "specialist_p50": os.path.join(ADVANCED_MODEL_DIR, "specialist_model.joblib"),
    # Optional specialized p10/p90 (if not present we will fallback)
    "specialist_p10": os.path.join(ADVANCED_MODEL_DIR, "specialist_p10_model.joblib"),
    "specialist_p90": os.path.join(ADVANCED_MODEL_DIR, "specialist_p90_model.joblib"),
}

models = {}

def load_all_models():
    """Load models listed in MODEL_PATHS; allow missing optional models (log them)."""
    print("--- Loading models ---")
    for key, path in MODEL_PATHS.items():
        try:
            if os.path.exists(path):
                models[key] = load(path)
                print(f"Loaded model: {path} -> key: {key}")
            else:
                models[key] = None
                print(f"Model not found (skipping): {path}")
        except Exception as e:
            models[key] = None
            print(f"ERROR loading model {key} from {path}: {e}")

    # Provide fallbacks: ensure p10/p50/p90 keys exist for generalist/specialist
    # If explicit p10/p90 missing, fallback to p50
    if not models.get("generalist_p10"):
        models["generalist_p10"] = models.get("generalist_p50")
        if models["generalist_p10"]:
            print("Fallback: generalist_p10 -> generalist_p50")
    if not models.get("generalist_p90"):
        models["generalist_p90"] = models.get("generalist_p50")
        if models["generalist_p90"]:
            print("Fallback: generalist_p90 -> generalist_p50")

    if not models.get("specialist_p10"):
        models["specialist_p10"] = models.get("specialist_p50")
        if models["specialist_p10"]:
            print("Fallback: specialist_p10 -> specialist_p50")
    if not models.get("specialist_p90"):
        models["specialist_p90"] = models.get("specialist_p50") or models.get("generalist_p90")
        if models["specialist_p90"]:
            print("Fallback: specialist_p90 -> specialist_p50 or generalist_p90")

load_all_models()

# --- Diagnosis mapping (same keywords used in training) ---
DIAGNOSIS_MAP = {
    'infection': ['sepsis', 'pneumonia', 'cellulitis', 'urinary tract infection', 'infection'],
    'cardiovascular': ['heart failure', 'atrial fibrillation', 'myocardial infarction', 'stroke', 'cardiac', 'hypertension', 'vascular', 'arrhythmia'],
    'respiratory': ['respiratory failure', 'copd', 'asthma', 'pulmonary embolism', 'respiratory'],
    'gastrointestinal': ['gastrointestinal bleed', 'pancreatitis', 'liver failure', 'bowel obstruction', 'abdominal pain'],
    'neurological': ['seizure', 'altered mental status', 'neurological', 'brain'],
    'renal': ['renal failure', 'kidney'],
    'trauma/injury': ['trauma', 'fall', 'fracture', 'injury'],
    'cancer': ['cancer', 'leukemia', 'lymphoma', 'tumor'],
    'metabolic/endocrine': ['diabetes', 'electrolyte imbalance', 'endocrine'],
    'other': []
}
DIAGNOSIS_PATTERNS = {
    c: re.compile('|'.join(re.escape(k) for k in keywords), re.IGNORECASE)
    for c, keywords in DIAGNOSIS_MAP.items() if keywords
}

def map_diagnosis_to_category(text):
    if not isinstance(text, str):
        return 'Other'
    for c, pattern in DIAGNOSIS_PATTERNS.items():
        if pattern.search(text):
            return c
    return 'Other'

# --- Utility: safe scalar prediction ---
def safe_predict_scalar(model, df, model_name="model"):
    """
    Call model.predict(df) and return a float scalar.
    Accepts sklearn-like models that return arrays or scalars.
    Raises ValueError with helpful message on unexpected outputs.
    """
    if model is None:
        raise ValueError(f"Model {model_name} is not loaded.")
    preds = model.predict(df)
    # numeric scalar
    if isinstance(preds, (int, float, np.floating, np.integer)):
        return float(preds)
    # numpy array or list-like -> take first element
    try:
        # convert to numpy array first if possible
        arr = np.asarray(preds)
        if arr.size == 0:
            raise ValueError(f"Prediction from {model_name} returned empty array.")
        return float(arr.flat[0])
    except Exception as e:
        raise ValueError(f"Prediction from {model_name} returned unexpected type/value: {preds}. Error: {e}")

# --- API endpoint ---
@app.route('/predict_impact', methods=['POST', 'OPTIONS'])
def predict_impact():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    patient_data = request.get_json()

    # required features
    required_features = [
        'anchor_age', 'gender', 'admission_type', 'insurance',
        'primary_diagnosis', 'procedure_count', 'max_creatinine',
        'min_hemoglobin', 'saps_ii_score'
    ]
    missing = [f for f in required_features if f not in patient_data]
    if missing:
        return jsonify({"error": f"Missing required features: {', '.join(missing)}"}), 400

    # prepare DataFrame (single-row)
    try:
        patient_df = pd.DataFrame([patient_data])
    except Exception as e:
        return jsonify({"error": f"Failed to convert input to DataFrame: {e}"}), 400

    # feature engineering: diagnosis category & optional interaction
    try:
        patient_df['diagnosis_category'] = patient_df['primary_diagnosis'].apply(map_diagnosis_to_category)
        # age_hr_interaction if avg_heart_rate present
        if 'avg_heart_rate' in patient_df.columns and pd.notna(patient_df['avg_heart_rate'].iloc[0]):
            try:
                avg_hr = float(patient_df['avg_heart_rate'].iloc[0])
                patient_df['age_hr_interaction'] = patient_df['anchor_age'].iloc[0] * abs(avg_hr - 75)
            except Exception:
                patient_df['age_hr_interaction'] = 0.0
        else:
            patient_df['age_hr_interaction'] = 0.0
    except Exception as e:
        return jsonify({"error": f"Feature engineering failed: {e}"}), 500

    # determine routing by SAPS-II
    try:
        risk_score = float(patient_df['saps_ii_score'].iloc[0]) if pd.notna(patient_df['saps_ii_score'].iloc[0]) else 0.0
    except Exception:
        risk_score = 0.0

    is_high_risk = risk_score > SAPS_II_RISK_THRESHOLD

    # select appropriate models (use p10/p50/p90 triples)
    try:
        if is_high_risk:
            model_p10 = models.get("specialist_p10") or models.get("specialist_p50") or models.get("generalist_p50")
            model_p50 = models.get("specialist_p50") or models.get("generalist_p50")
            model_p90 = models.get("specialist_p90") or models.get("specialist_p50") or models.get("generalist_p90") or models.get("generalist_p50")
            used_model = "Specialist"
        else:
            model_p10 = models.get("generalist_p10") or models.get("generalist_p50")
            model_p50 = models.get("generalist_p50")
            model_p90 = models.get("generalist_p90") or models.get("generalist_p50")
            used_model = "Generalist"

        if not all([model_p10, model_p50, model_p90]):
            return jsonify({"error": "One or more required models are not available on server."}), 500

        # safe scalar predictions
        pred_p10 = safe_predict_scalar(model_p10, patient_df, "model_p10")
        pred_p50 = safe_predict_scalar(model_p50, patient_df, "model_p50")
        pred_p90 = safe_predict_scalar(model_p90, patient_df, "model_p90")

        # ensure non-negative floats
        pred_p10 = float(max(0.0, pred_p10))
        pred_p50 = float(max(0.0, pred_p50))
        pred_p90 = float(max(0.0, pred_p90))
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # cost calculation - allow client override
    try:
        user_cost = patient_data.get("average_daily_patient_cost") or patient_data.get("avg_daily_cost") or None
        user_currency = patient_data.get("currency") or patient_data.get("curr") or None

        if user_cost is not None:
            try:
                daily_cost = float(user_cost)
            except Exception:
                daily_cost = float(AVERAGE_DAILY_PATIENT_COST)
        else:
            daily_cost = float(AVERAGE_DAILY_PATIENT_COST)

        currency = user_currency if user_currency else (AVERAGE_DAILY_PATIENT_COST_CURRENCY or "USD")

        estimated_cost_median = pred_p50 * daily_cost
        estimated_cost_p90 = pred_p90 * daily_cost
        estimated_cost_p10 = pred_p10 * daily_cost
    except Exception as e:
        return jsonify({"error": f"Cost computation failed: {e}"}), 500

    # Prepare response
    response = {
        "patient_details_received": patient_data,
        "saps_ii_score": float(risk_score),
        "model_used": used_model,
        "predicted_los_p10_days": round(pred_p10, 2),
        "predicted_los_median_days": round(pred_p50, 2),
        "predicted_los_p90_days": round(pred_p90, 2),
        "estimated_cost_p10": round(estimated_cost_p10, 2),
        "estimated_cost_median": round(estimated_cost_median, 2),
        "estimated_cost_p90": round(estimated_cost_p90, 2),
        "currency": currency
    }

    # add simple headers for CORS-friendly clients
    r = jsonify(response)
    r.headers.add('Access-Control-Allow-Origin', '*')
    return r, 200

# --- Run ---
if __name__ == "__main__":
    # Sanity check: ensure at least generalist_p50 is loaded
    if not models.get("generalist_p50"):
        print("WARNING: generalist_p50 not loaded - API may not be able to produce predictions.")
    print("Starting Flask server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
