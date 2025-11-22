import pandas as pd
import psycopg2
import os

# --- Configuration ---
DB_NAME = "my_postgres"
DB_USER = "postgres"
DB_PASS = "MyStrongPassword123!"
DB_HOST = "localhost"
DB_PORT = "5432"

# --- NEW: Point to the readmission-specific files ---
SQL_FILE_PATH = "scripts/feature_engineering_readmission.sql"
OUTPUT_DIR = "data/mimic_iv_processed"
OUTPUT_FILENAME = "readmission_features.csv" # Save to a new file

def fetch_readmission_features():
    """Connects to the DB, runs the readmission query, and saves the result."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    conn_string = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    try:
        print("Connecting to the database for READMISSION task...")
        with open(SQL_FILE_PATH, 'r') as f:
            sql_query = f.read()
        
        print("Executing readmission feature engineering query...")
        df = pd.read_sql_query(sql_query, conn_string)
        
        print(f"Query successful. Fetched {len(df)} rows.")
        
        df.to_csv(output_path, index=False)
        print(f"Readmission feature set saved successfully to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_readmission_features()
