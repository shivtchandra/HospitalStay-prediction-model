# Configuration file for hospital-specific assumptions used in business impact calculations.

# --- Resource Planning Assumptions ---
TOTAL_ICU_BEDS = 50  # Example: Total number of ICU beds available in the unit being simulated
SIMULATION_HORIZON_DAYS = 7 # How many days into the future to forecast bed occupancy

# --- Cost Estimation Assumptions ---
# This is a major simplification. Real costs vary hugely by patient, unit, and interventions.
# Use values representative for your target scenario (e.g., average daily ICU cost).
AVERAGE_DAILY_PATIENT_COST = 2500 # Example: Assumed average cost per patient per day (e.g., in USD)

# --- Triage Threshold ---
# The SAPS-II score threshold used to route to the specialist model
SAPS_II_RISK_THRESHOLD = 40 # Matches the threshold used in predict_and_explain.py

