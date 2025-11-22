import pandas as pd
import psycopg2
import os

# --- Configuration ---
DB_NAME = "my_postgres"
DB_USER = "postgres"
DB_PASS = "MyStrongPassword123!"
DB_HOST = "localhost"
DB_PORT = "5432"

SQL_FILE_PATH = "scripts/feature_engineering.sql"
OUTPUT_DIR = "data/mimic_iv_processed"
OUTPUT_FILENAME = "length_of_stay_features.csv"

def fetch_features_and_save():
    """Connects to the DB, runs the feature engineering query, and saves the result."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    conn_string = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    try:
        if not os.path.exists(SQL_FILE_PATH):
            print(f"ERROR: SQL file not found at {SQL_FILE_PATH}")
            return
            
        print("Connecting to the database...")
        with open(SQL_FILE_PATH, 'r') as f:
            sql_query = f.read()
        
        print("Executing feature engineering query...")
        # Use pandas to directly read SQL query results into a DataFrame
        df = pd.read_sql_query(sql_query, conn_string)
        
        print(f"Query successful. Fetched {len(df)} rows and {len(df.columns)} columns.")
        
        if df.empty:
            print("WARNING: The query returned an empty DataFrame. Check your SQL query and the database tables.")
            return

        # Save the clean feature set to the processed data folder
        df.to_csv(output_path, index=False)
        print(f"Feature set saved successfully to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_features_and_save()


