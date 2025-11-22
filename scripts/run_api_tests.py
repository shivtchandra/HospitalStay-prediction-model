#!/usr/bin/env python3
"""
run_api_tests.py

Sends a set of test patient payloads to the /predict_impact endpoint and
writes all responses (and request payloads) into a single JSON file.

Usage:
  python scripts/run_api_tests.py
  API URL can be overridden with the environment variable API_URL, e.g.:
  API_URL="http://127.0.0.1:5000/predict_impact" python scripts/run_api_tests.py
"""

import os
import json
import time
import traceback
from typing import List, Dict
import requests

# Configuration
API_URL = os.environ.get("API_URL", "http://127.0.0.1:5000/predict_impact")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_results.json")
TIMEOUT = 20  # seconds per request
SLEEP_BETWEEN = 0.15  # throttle to avoid overwhelming local server

# A diverse set of test patients covering diagnosis categories and edge cases
TEST_CASES: List[Dict] = [
    # infection (sepsis)
    {"anchor_age": 70, "gender": "M", "admission_type": "EMERGENCY", "insurance": "Medicare",
     "primary_diagnosis": "Sepsis", "procedure_count": 2, "max_creatinine": 1.8, "min_hemoglobin": 9.5,
     "saps_ii_score": 55, "avg_heart_rate": 105, "average_daily_patient_cost": 2500, "currency": "INR"},

    # pneumonia (respiratory/infection)
    {"anchor_age": 58, "gender": "F", "admission_type": "URGENT", "insurance": "Private",
     "primary_diagnosis": "Pneumonia", "procedure_count": 1, "max_creatinine": 1.2, "min_hemoglobin": 11.0,
     "saps_ii_score": 28, "avg_heart_rate": 90, "average_daily_patient_cost": 3000, "currency": "INR"},

    # cardiovascular - heart failure
    {"anchor_age": 82, "gender": "M", "admission_type": "EMERGENCY", "insurance": "Medicare",
     "primary_diagnosis": "Heart failure with pulmonary edema", "procedure_count": 3, "max_creatinine": 2.0,
     "min_hemoglobin": 10.2, "saps_ii_score": 60, "avg_heart_rate": 110, "average_daily_patient_cost": 4000, "currency": "INR"},

    # stroke (neurological)
    {"anchor_age": 67, "gender": "F", "admission_type": "EMERGENCY", "insurance": "Private",
     "primary_diagnosis": "Ischemic stroke", "procedure_count": 0, "max_creatinine": 1.0, "min_hemoglobin": 12.0,
     "saps_ii_score": 40, "avg_heart_rate": 78, "average_daily_patient_cost": 3500, "currency": "INR"},

    # respiratory failure / COPD
    {"anchor_age": 73, "gender": "M", "admission_type": "EMERGENCY", "insurance": "Other",
     "primary_diagnosis": "COPD exacerbation - respiratory failure", "procedure_count": 1, "max_creatinine": 1.3,
     "min_hemoglobin": 10.9, "saps_ii_score": 35, "avg_heart_rate": 95, "average_daily_patient_cost": 2000, "currency": "INR"},

    # gastrointestinal bleed
    {"anchor_age": 49, "gender": "F", "admission_type": "URGENT", "insurance": "Private",
     "primary_diagnosis": "Upper gastrointestinal bleed", "procedure_count": 2, "max_creatinine": 0.9,
     "min_hemoglobin": 7.8, "saps_ii_score": 22, "avg_heart_rate": 115, "average_daily_patient_cost": 2200, "currency": "INR"},

    # renal failure
    {"anchor_age": 61, "gender": "M", "admission_type": "EMERGENCY", "insurance": "Medicare",
     "primary_diagnosis": "Acute renal failure", "procedure_count": 1, "max_creatinine": 4.5, "min_hemoglobin": 9.0,
     "saps_ii_score": 45, "avg_heart_rate": 88, "average_daily_patient_cost": 2800, "currency": "INR"},

    # trauma / fracture
    {"anchor_age": 34, "gender": "F", "admission_type": "ELECTIVE", "insurance": "Private",
     "primary_diagnosis": "Femur fracture - trauma", "procedure_count": 1, "max_creatinine": 0.8, "min_hemoglobin": 13.0,
     "saps_ii_score": 6, "avg_heart_rate": 72, "average_daily_patient_cost": 1500, "currency": "INR"},

    # cancer patient
    {"anchor_age": 61, "gender": "F", "admission_type": "URGENT", "insurance": "Private",
     "primary_diagnosis": "Metastatic lung cancer", "procedure_count": 0, "max_creatinine": 1.0, "min_hemoglobin": 9.3,
     "saps_ii_score": 30, "avg_heart_rate": 82, "average_daily_patient_cost": 5000, "currency": "INR"},

    # metabolic (diabetes) with electrolyte imbalance
    {"anchor_age": 55, "gender": "M", "admission_type": "EMERGENCY", "insurance": "Medicaid",
     "primary_diagnosis": "Diabetic ketoacidosis", "procedure_count": 0, "max_creatinine": 1.1, "min_hemoglobin": 12.0,
     "saps_ii_score": 33, "avg_heart_rate": 120, "average_daily_patient_cost": 1800, "currency": "INR"},

    # pediatric-like (young)
    {"anchor_age": 8, "gender": "F", "admission_type": "EMERGENCY", "insurance": "Other",
     "primary_diagnosis": "Asthma exacerbation", "procedure_count": 0, "max_creatinine": 0.4, "min_hemoglobin": 11.5,
     "saps_ii_score": 5, "avg_heart_rate": 130, "average_daily_patient_cost": 1000, "currency": "INR"},

    # borderline/low-data (missing some optional fields)
    {"anchor_age": 45, "gender": "M", "admission_type": "ELECTIVE", "insurance": "Other",
     "primary_diagnosis": "Chest pain", "procedure_count": 0, "max_creatinine": None, "min_hemoglobin": None,
     "saps_ii_score": 10, "avg_heart_rate": None, "average_daily_patient_cost": 1200, "currency": "INR"},

    # weird/unseen diagnosis to test 'other' bucket
    {"anchor_age": 59, "gender": "F", "admission_type": "URGENT", "insurance": "Private",
     "primary_diagnosis": "Unknown rare disease XZ-12", "procedure_count": 1, "max_creatinine": 1.1, "min_hemoglobin": 11.0,
     "saps_ii_score": 18, "avg_heart_rate": 85, "average_daily_patient_cost": 2000, "currency": "INR"},

    # high SAPS but low labs
    {"anchor_age": 77, "gender": "M", "admission_type": "EMERGENCY", "insurance": "Medicare",
     "primary_diagnosis": "Severe pneumonia", "procedure_count": 4, "max_creatinine": 2.3, "min_hemoglobin": 8.0,
     "saps_ii_score": 85, "avg_heart_rate": 140, "average_daily_patient_cost": 3500, "currency": "INR"},

    # boundary case: zero procedures and 0 SAPS
    {"anchor_age": 30, "gender": "M", "admission_type": "ELECTIVE", "insurance": "Private",
     "primary_diagnosis": "Routine check", "procedure_count": 0, "max_creatinine": 0.7, "min_hemoglobin": 14.2,
     "saps_ii_score": 0, "avg_heart_rate": 70, "average_daily_patient_cost": 900, "currency": "INR"},
]

def safe_post(payload: Dict) -> Dict:
    """Send POST and return dict with request, response, status, and error if any."""
    entry = {"request": payload, "response": None, "status_code": None, "error": None, "elapsed": None}
    try:
        # remove None values so server can apply defaults
        clean_payload = {k: v for k, v in payload.items() if v is not None}
        start = time.time()
        r = requests.post(API_URL, json=clean_payload, timeout=TIMEOUT)
        elapsed = time.time() - start

        entry["status_code"] = r.status_code
        entry["elapsed"] = round(elapsed, 3)
        try:
            entry["response"] = r.json()
        except ValueError:
            entry["response"] = {"text": r.text}
        if r.status_code >= 400:
            entry["error"] = f"HTTP {r.status_code}"
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {str(e)}"
        entry["response"] = None
    return entry

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "meta": {
            "api_url": API_URL,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(TEST_CASES)
        },
        "results": []
    }

    print(f"Sending {len(TEST_CASES)} test cases to {API_URL}")
    for i, tc in enumerate(TEST_CASES, start=1):
        print(f"[{i}/{len(TEST_CASES)}] -> SAPS={tc.get('saps_ii_score')} diag='{tc.get('primary_diagnosis')[:40]}' ... ", end="", flush=True)
        entry = safe_post(tc)
        results["results"].append(entry)
        if entry["error"]:
            print("ERROR:", entry["error"])
        else:
            status = entry.get("status_code")
            print(f"OK ({status}) in {entry.get('elapsed')}s")
        time.sleep(SLEEP_BETWEEN)

    # write pretty JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved all results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
