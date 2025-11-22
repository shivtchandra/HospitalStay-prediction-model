#!/bin/bash
# ==============================================================================
# Batch test script for /predict_impact endpoint
# Covers all major diagnosis categories & risk levels
# Saves responses into logs/ with timestamped JSON files
# ==============================================================================

API_URL="http://127.0.0.1:5000/predict_impact"
OUT_DIR="logs"
mkdir -p "$OUT_DIR"

timestamp=$(date +"%Y%m%d_%H%M%S")
echo "Running model test suite at $timestamp"
echo "Saving responses to $OUT_DIR/"

# Helper function
run_test() {
  local name="$1"
  local payload="$2"
  local file="${OUT_DIR}/${timestamp}_${name}.json"

  echo "→ Testing: $name"
  echo "$payload" | curl -s -H "Content-Type: application/json" -X POST "$API_URL" -d @- > "$file"
  echo "  ✅ Saved: $file"
  echo ""
}

# ==============================================================================
# TEST CASES — organized by diagnosis category
# ==============================================================================

# --- Infection ---
run_test "infection_sepsis" '{"anchor_age":72,"gender":"F","admission_type":"EMERGENCY","insurance":"Medicare","primary_diagnosis":"Sepsis","procedure_count":3,"max_creatinine":2.4,"min_hemoglobin":8.3,"saps_ii_score":65}'
run_test "infection_pneumonia" '{"anchor_age":58,"gender":"M","admission_type":"URGENT","insurance":"Private","primary_diagnosis":"Pneumonia","procedure_count":1,"max_creatinine":1.1,"min_hemoglobin":11.2,"saps_ii_score":30}'
run_test "infection_cellulitis" '{"anchor_age":46,"gender":"F","admission_type":"ELECTIVE","insurance":"Other","primary_diagnosis":"Cellulitis","procedure_count":0,"max_creatinine":0.9,"min_hemoglobin":12.8,"saps_ii_score":8}'

# --- Cardiovascular ---
run_test "cardio_hf" '{"anchor_age":80,"gender":"F","admission_type":"EMERGENCY","insurance":"Medicare","primary_diagnosis":"Heart failure","procedure_count":2,"max_creatinine":1.6,"min_hemoglobin":9.0,"saps_ii_score":58}'
run_test "cardio_mi" '{"anchor_age":64,"gender":"M","admission_type":"URGENT","insurance":"Private","primary_diagnosis":"Myocardial infarction","procedure_count":1,"max_creatinine":1.0,"min_hemoglobin":13.2,"saps_ii_score":35}'
run_test "cardio_afib" '{"anchor_age":72,"gender":"M","admission_type":"URGENT","insurance":"Private","primary_diagnosis":"Atrial fibrillation","procedure_count":0,"max_creatinine":1.0,"min_hemoglobin":11.0,"saps_ii_score":22}'

# --- Respiratory ---
run_test "resp_failure" '{"anchor_age":68,"gender":"F","admission_type":"EMERGENCY","insurance":"Medicaid","primary_diagnosis":"Respiratory failure","procedure_count":2,"max_creatinine":1.9,"min_hemoglobin":9.6,"saps_ii_score":62}'
run_test "resp_copd" '{"anchor_age":71,"gender":"M","admission_type":"URGENT","insurance":"Private","primary_diagnosis":"COPD exacerbation","procedure_count":0,"max_creatinine":1.2,"min_hemoglobin":11.5,"saps_ii_score":33}'
run_test "resp_embolism" '{"anchor_age":52,"gender":"F","admission_type":"EMERGENCY","insurance":"Private","primary_diagnosis":"Pulmonary embolism","procedure_count":1,"max_creatinine":0.9,"min_hemoglobin":12.0,"saps_ii_score":28}'

# --- Gastrointestinal ---
run_test "gi_bleed" '{"anchor_age":77,"gender":"M","admission_type":"EMERGENCY","insurance":"Medicare","primary_diagnosis":"Gastrointestinal bleed","procedure_count":2,"max_creatinine":1.7,"min_hemoglobin":7.8,"saps_ii_score":68}'
run_test "gi_pancreatitis" '{"anchor_age":45,"gender":"M","admission_type":"URGENT","insurance":"Private","primary_diagnosis":"Pancreatitis","procedure_count":0,"max_creatinine":1.0,"min_hemoglobin":13.0,"saps_ii_score":24}'
run_test "gi_obstruction" '{"anchor_age":66,"gender":"F","admission_type":"EMERGENCY","insurance":"Private","primary_diagnosis":"Bowel obstruction","procedure_count":1,"max_creatinine":1.3,"min_hemoglobin":10.1,"saps_ii_score":42}'

# --- Neurological ---
run_test "neuro_stroke" '{"anchor_age":82,"gender":"F","admission_type":"EMERGENCY","insurance":"Medicare","primary_diagnosis":"Stroke","procedure_count":1,"max_creatinine":1.4,"min_hemoglobin":10.0,"saps_ii_score":60}'
run_test "neuro_seizure" '{"anchor_age":39,"gender":"M","admission_type":"URGENT","insurance":"Private","primary_diagnosis":"Seizure","procedure_count":0,"max_creatinine":0.8,"min_hemoglobin":13.5,"saps_ii_score":18}'
run_test "neuro_ams" '{"anchor_age":74,"gender":"F","admission_type":"ELECTIVE","insurance":"Other","primary_diagnosis":"Altered mental status","procedure_count":0,"max_creatinine":1.1,"min_hemoglobin":11.0,"saps_ii_score":26}'

# --- Renal ---
run_test "renal_failure" '{"anchor_age":70,"gender":"M","admission_type":"EMERGENCY","insurance":"Medicare","primary_diagnosis":"Renal failure","procedure_count":1,"max_creatinine":4.2,"min_hemoglobin":9.2,"saps_ii_score":63}'
run_test "renal_injury" '{"anchor_age":60,"gender":"F","admission_type":"URGENT","insurance":"Private","primary_diagnosis":"Kidney injury","procedure_count":0,"max_creatinine":2.1,"min_hemoglobin":10.5,"saps_ii_score":40}'
run_test "renal_obstruction" '{"anchor_age":55,"gender":"M","admission_type":"ELECTIVE","insurance":"Private","primary_diagnosis":"Obstructive uropathy","procedure_count":1,"max_creatinine":1.3,"min_hemoglobin":12.2,"saps_ii_score":22}'

# --- Trauma / Injury ---
run_test "trauma_multiple" '{"anchor_age":48,"gender":"M","admission_type":"EMERGENCY","insurance":"Private","primary_diagnosis":"Trauma - multiple fractures","procedure_count":4,"max_creatinine":1.1,"min_hemoglobin":8.8,"saps_ii_score":50}'
run_test "trauma_hip" '{"anchor_age":86,"gender":"F","admission_type":"EMERGENCY","insurance":"Medicare","primary_diagnosis":"Hip fracture","procedure_count":2,"max_creatinine":1.0,"min_hemoglobin":9.5,"saps_ii_score":46}'
run_test "trauma_minor" '{"anchor_age":30,"gender":"M","admission_type":"ELECTIVE","insurance":"Private","primary_diagnosis":"Minor soft tissue injury","procedure_count":0,"max_creatinine":0.8,"min_hemoglobin":14.0,"saps_ii_score":5}'

# --- Cancer ---
run_test "cancer_leukemia" '{"anchor_age":63,"gender":"F","admission_type":"URGENT","insurance":"Private","primary_diagnosis":"Leukemia","procedure_count":1,"max_creatinine":1.2,"min_hemoglobin":7.5,"saps_ii_score":55}'
run_test "cancer_metastatic" '{"anchor_age":69,"gender":"M","admission_type":"EMERGENCY","insurance":"Private","primary_diagnosis":"Metastatic colon cancer","procedure_count":2,"max_creatinine":1.5,"min_hemoglobin":9.0,"saps_ii_score":48}'
run_test "cancer_breast" '{"anchor_age":52,"gender":"F","admission_type":"ELECTIVE","insurance":"Private","primary_diagnosis":"Breast cancer - planned admission","procedure_count":0,"max_creatinine":0.9,"min_hemoglobin":13.0,"saps_ii_score":12}'

# --- Metabolic / Endocrine ---
run_test "metabolic_dka" '{"anchor_age":29,"gender":"F","admission_type":"EMERGENCY","insurance":"Private","primary_diagnosis":"Diabetic ketoacidosis","procedure_count":0,"max_creatinine":1.0,"min_hemoglobin":12.5,"saps_ii_score":40}'
run_test "metabolic_electrolyte" '{"anchor_age":67,"gender":"M","admission_type":"URGENT","insurance":"Medicare","primary_diagnosis":"Electrolyte imbalance - hyponatremia","procedure_count":0,"max_creatinine":1.2,"min_hemoglobin":11.3,"saps_ii_score":34}'
run_test "metabolic_diabetes" '{"anchor_age":50,"gender":"F","admission_type":"ELECTIVE","insurance":"Private","primary_diagnosis":"Diabetes follow-up","procedure_count":0,"max_creatinine":0.9,"min_hemoglobin":13.6,"saps_ii_score":10}'

# --- Other / Edge ---
run_test "other_unspecified" '{"anchor_age":40,"gender":"M","admission_type":"URGENT","insurance":"Other","primary_diagnosis":"Unspecified illness","procedure_count":0,"max_creatinine":0.8,"min_hemoglobin":14.2,"saps_ii_score":6}'
run_test "other_elderly" '{"anchor_age":93,"gender":"F","admission_type":"EMERGENCY","insurance":"Medicare","primary_diagnosis":"Multiple comorbidities - frailty","procedure_count":5,"max_creatinine":3.8,"min_hemoglobin":7.0,"saps_ii_score":75,"average_daily_patient_cost":2500,"currency":"INR"}'
run_test "other_minimal" '{"anchor_age":18,"gender":"M","admission_type":"ELECTIVE","insurance":"Private","primary_diagnosis":"Routine observation","procedure_count":0,"max_creatinine":0.5,"min_hemoglobin":15.0,"saps_ii_score":0}'

echo "✅ All test cases executed. Check the $OUT_DIR directory for output JSON files."
