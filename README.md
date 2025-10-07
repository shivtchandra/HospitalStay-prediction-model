# Predicting Hospital Length of Stay using MIMIC-IV

A machine learning pipeline designed to predict the length of stay (LOS) for hospital patients using the MIMIC-IV clinical dataset. This project helps hospitals with resource planning and patient flow management by forecasting how long a new patient is likely to be hospitalized.

This initial version serves as a robust baseline, incorporating demographic, administrative, and key clinical variables, including lab results and vital signs from the first 24 hours of a patient's stay.

## Core Idea

The model predicts a patient's Length of Stay based on a feature set engineered from multiple clinical tables. Key features include:

- **Demographics**: Patient's age and gender
- **Admission Details**: Type of admission (e.g., EMERGENCY, URGENT) and insurance provider
- **Clinical Indicators**: Primary diagnosis, count of procedures performed, and key lab results (creatinine, hemoglobin)
- **Vital Signs**: Average heart rate during the first 24 hours of admission, a critical indicator of physiological stress

## Tech Stack

- **Language**: Python 3.9+
- **Data Manipulation**: Pandas
- **Machine Learning**: Scikit-learn, XGBoost
- **Database**: PostgreSQL (managed with Docker or Homebrew)
- **Database Connector**: psycopg2

## Project Structure

```
.
├── data/
│   ├── mimic_iv_raw/hosp/      # Raw MIMIC-IV CSV files
│   └── mimic_iv_processed/     # Final, cleaned feature set
├── models/
│   └── length_of_stay_predictor.joblib    # Saved model
└── scripts/
    ├── load_all_data_to_db.py             # Loads raw CSVs into PostgreSQL
    ├── feature_engineering.sql            # Feature creation logic
    ├── run_feature_engineering.py         # Executes SQL query and saves feature set
    ├── train_model.py                     # Trains XGBoost model
    └── predict.py                         # Prediction on new patients
```

## Getting Started

### Prerequisites

- **Docker** (or Homebrew PostgreSQL): A running PostgreSQL server
- **Python 3.9+**: Install dependencies via `pip install pandas psycopg2-binary scikit-learn xgboost`
- **MIMIC-IV Data**: Access to the MIMIC-IV dataset with `hosp` CSV files placed in `data/mimic_iv_raw/hosp/`

### Installation & Usage

#### 1. Start the Database

Start your PostgreSQL server. If using Docker:

```bash
docker run --name my-postgres-container \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=MyStrongPassword123! \
  -e POSTGRES_DB=my_postgres \
  -p 5432:5432 \
  -d postgres
```

#### 2. Load Raw Data

Execute the data loading script (one-time process):

```bash
python3 scripts/load_all_data_to_db.py
```

#### 3. Engineer Features

Run the feature engineering script:

```bash
python3 scripts/run_feature_engineering.py
```

#### 4. Train the Model

Train the XGBoost regressor:

```bash
python3 scripts/train_model.py
```

#### 5. Make Predictions

Use the trained model on new patient data:

```bash
python3 scripts/predict.py
```

## Results & Observations

This initial version achieves a **Mean Absolute Error (MAE) of approximately 2.18 days**, providing a solid baseline for length of stay prediction.

### Key Finding

Initial tests revealed that the model was not yet sensitive enough to vital signs data. When presented with two otherwise identical patients—one with a normal heart rate and one with a significantly elevated heart rate—the model produced the same prediction for both.

This finding demonstrated that while the pipeline was successful, the model was being dominated by stronger features like `admission_type` and `primary_diagnosis`. It highlighted the need for more advanced techniques, such as the "Generalist + Specialist" system and more sophisticated feature engineering, to help the model learn these subtle but clinically critical patterns. This baseline was the essential first step that motivated the project's more advanced iterations.
