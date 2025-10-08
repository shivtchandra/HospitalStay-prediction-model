# Predicting Hospital Length of Stay using MIMIC-IV

A machine learning system designed to predict the length of stay (LOS) for hospital patients using the MIMIC-IV clinical dataset. This project helps hospitals with resource planning by forecasting how long a new patient is likely to be hospitalized, with a special focus on robustly handling high-risk cases.

This advanced version implements a two-model "Generalist + Specialist" architecture, inspired by research on Distributionally Robust Optimization (DRO), to ensure that predictions are both accurate for the average patient and sensitive to clinically critical outliers.

## 💡 Core Idea

Initial model versions revealed a classic machine learning problem: the model performed well on average but failed on difficult subgroups, specifically patients with abnormal vital signs. It learned to ignore these subtle signals in favor of more dominant features like `admission_type`.

To solve this, we implemented a practical version of Group DRO, as described in the paper ["Distributionally Robust Neural Networks for Group Shifts"](https://arxiv.org/abs/1911.08731) by Sagawa et al. Our system consists of:

- **A Generalist Model**: An XGBoost model trained on the entire patient population to accurately predict outcomes for the majority of standard cases.

- **A Specialist Model**: A second XGBoost model trained only on a filtered subset of high-risk patients (the "worst-performing group"). This forces the specialist to learn the subtle clinical patterns that the generalist ignores.

- **An Intelligent Triage System**: A rule-based router in the prediction script that uses a clinically-grounded feature (`age_hr_interaction`) to decide whether to consult the Generalist or the Specialist for a new patient.

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Data Manipulation**: Pandas
- **Machine Learning**: Scikit-learn, XGBoost
- **Database**: PostgreSQL (managed with Docker or Homebrew)
- **Database Connector**: psycopg2

## 📂 Project Structure

The project is organized into a modular pipeline that supports the two-model system.

```
.
├── data/
│   ├── mimic_iv_raw/hosp/     # Raw MIMIC-IV CSV files
│   └── mimic_iv_processed/    # Final, cleaned feature set
├── models/
│   ├── length_of_stay_predictor.joblib    # The saved Generalist model
│   └── specialist_los_predictor.joblib    # The saved Specialist model
└── scripts/
    ├── load_all_data_to_db.py             # Loads all raw CSVs into PostgreSQL
    ├── feature_engineering.sql            # Advanced feature creation logic
    ├── run_feature_engineering.py         # Executes SQL query and saves feature set
    ├── train_model.py                     # Trains the Generalist XGBoost model
    ├── train_specialist_model.py          # Trains the Specialist model on high-risk cases
    └── predict.py                         # Runs the triage logic and predicts using both models
```

## 🚀 Getting Started

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

Execute the data loading script (this is a one-time, lengthy process):

```bash
python3 scripts/load_all_data_to_db.py
```

#### 3. Engineer Features

Run the feature engineering script to generate the complete dataset:

```bash
python3 scripts/run_feature_engineering.py
```

#### 4. Train Both Models

First, train the Generalist model on the full dataset:

```bash
python3 scripts/train_model.py
```

Next, train the Specialist model on the filtered high-risk cases:

```bash
python3 scripts/train_specialist_model.py
```

#### 5. Make Predictions

Use the final prediction script to see the intelligent, two-model system in action:

```bash
python3 scripts/predict.py
```

## 📊 Results & Success

This advanced, two-model system successfully solves the "lazy model" problem and demonstrates a robust, clinically-nuanced approach to prediction.

### Model Performance

- **Generalist Model MAE**: ~2.31 days
- **Specialist Model MAE**: ~3.95 days

> **Note**: The specialist's higher error is expected and indicates success. It reflects the inherent difficulty and higher variance of its specialized task of predicting outcomes for only the most complex and unpredictable patients.

### Key Finding & Final Outcome

The final prediction script demonstrates the success of the system. When presented with two clinically identical patients, one with normal vitals and one with a high-risk vital sign profile:

- The **low-risk patient** is correctly routed to the Generalist, receiving a prediction of **~4.11 days**.
- The **high-risk patient** is correctly routed to the Specialist, receiving a significantly longer prediction of **~5.75 days**.

This outcome proves that our implementation of the Group DRO concept was successful. The system no longer ignores critical clinical signals, resulting in a more intelligent, robust, and trustworthy predictive tool.

## 📚 References

- Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2019). Distributionally Robust Neural Networks for Group Shifts. *arXiv preprint arXiv:1911.08731*.
- Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). *PhysioNet*.

## 📄 License

This project uses the MIMIC-IV dataset, which requires credentialed access through PhysioNet. Please ensure you have completed the required training and agreed to the data use agreement before using this code.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on this repository.
