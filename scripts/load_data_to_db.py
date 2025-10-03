import psycopg2
import os
import re

# --- Configuration ---
DB_NAME = "my_postgres"
DB_USER = "postgres"
DB_PASS = "MyStrongPassword123!" # Make sure to use your actual password
DB_HOST = "localhost"
DB_PORT = "5432"

DATA_DIR = "data/mimic_iv_raw/hosp"

def clean_csv_line(line):
    """Removes null characters from a line."""
    return line.replace('\x00', '')

def load_csv_to_table(conn, csv_filename, table_name, create_table_sql):
    """Loads a single CSV file into a specified database table."""
    csv_file_path = os.path.join(DATA_DIR, csv_filename)
    cur = conn.cursor()
    
    try:
        print(f"--- Processing {table_name} ---")
        cur.execute(f"DROP TABLE IF EXISTS {table_name};")
        cur.execute(create_table_sql)
        conn.commit()
        print(f"Table '{table_name}' created and ready.")

        # Special handling for chartevents to clean null bytes
        if table_name == 'chartevents':
            print(f"Performing special cleaning for '{csv_filename}'...")
            with open(csv_file_path, 'r', errors='replace') as f_in, open('temp_chartevents_cleaned.csv', 'w') as f_out:
                header = next(f_in) # Read header
                f_out.write(header) # Write header to temp file
                for line in f_in:
                    f_out.write(clean_csv_line(line))
            
            print("Cleaning complete. Starting data load...")
            with open('temp_chartevents_cleaned.csv', 'r') as f:
                 cur.copy_expert(sql=f"COPY {table_name} FROM STDIN WITH CSV HEADER", file=f)
            os.remove('temp_chartevents_cleaned.csv') # Clean up temp file
        else:
            with open(csv_file_path, 'r') as f:
                print(f"Loading data from '{csv_filename}'...")
                cur.copy_expert(sql=f"COPY {table_name} FROM STDIN WITH CSV HEADER", file=f)

        conn.commit()
        print(f"Data for '{table_name}' loaded successfully.\n")

    except Exception as e:
        print(f"Error processing {table_name}: {e}")
        conn.rollback()
    finally:
        cur.close()

def main():
    # --- Table Definitions ---
    create_admissions_sql = """CREATE TABLE admissions (subject_id INT NOT NULL, hadm_id INT NOT NULL, admit_dt TIMESTAMP(0) NOT NULL, disch_dt TIMESTAMP(0) NOT NULL, deathtime TIMESTAMP(0), admission_type VARCHAR(255) NOT NULL, admit_provider_id VARCHAR(255), admission_location VARCHAR(255), discharge_location VARCHAR(255), insurance VARCHAR(255) NOT NULL, language VARCHAR(50), marital_status VARCHAR(255), race VARCHAR(255) NOT NULL, edregtime TIMESTAMP(0), edouttime TIMESTAMP(0), hospital_expire_flag INT2 NOT NULL);"""
    create_patients_sql = """CREATE TABLE patients (subject_id INT NOT NULL, gender CHAR(1) NOT NULL, anchor_age INT NOT NULL, anchor_year INT NOT NULL, anchor_year_group VARCHAR(255) NOT NULL, dod DATE);"""
    create_diagnoses_sql = """CREATE TABLE diagnoses_icd (subject_id INT NOT NULL, hadm_id INT NOT NULL, seq_num INT, icd_code VARCHAR(20) NOT NULL, icd_version INT NOT NULL);"""
    create_labevents_sql = """CREATE TABLE labevents (labevent_id INT NOT NULL, subject_id INT NOT NULL, hadm_id INT, specimen_id INT NOT NULL, itemid INT NOT NULL, order_provider_id VARCHAR(255), charttime TIMESTAMP(0), storetime TIMESTAMP(0), value VARCHAR(255), valuenum NUMERIC, valueuom VARCHAR(50), ref_range_lower NUMERIC, ref_range_upper NUMERIC, flag VARCHAR(50), priority VARCHAR(50), comments TEXT);"""
    create_procedures_sql = """CREATE TABLE procedures_icd (subject_id INT NOT NULL, hadm_id INT NOT NULL, seq_num INT NOT NULL, chartdate DATE NOT NULL, icd_code VARCHAR(20) NOT NULL, icd_version INT NOT NULL);"""
    create_chartevents_sql = """CREATE TABLE chartevents (subject_id INT NOT NULL, hadm_id INT NOT NULL, stay_id INT NOT NULL, caregiver_id VARCHAR(255), charttime TIMESTAMP(0) NOT NULL, storetime TIMESTAMP(0), itemid INT NOT NULL, value VARCHAR(255), valuenum NUMERIC, valueuom VARCHAR(50), warning INT2);"""
    create_d_icd_diagnoses_sql = """CREATE TABLE d_icd_diagnoses (icd_code VARCHAR(20) NOT NULL, icd_version INT NOT NULL, long_title TEXT NOT NULL);"""
    
    conn = None
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
        
        load_csv_to_table(conn, "admissions.csv", "admissions", create_admissions_sql)
        load_csv_to_table(conn, "patients.csv", "patients", create_patients_sql)
        load_csv_to_table(conn, "diagnoses_icd.csv", "diagnoses_icd", create_diagnoses_sql)
        load_csv_to_table(conn, "procedures_icd.csv", "procedures_icd", create_procedures_sql)
        load_csv_to_table(conn, "labevents.csv", "labevents", create_labevents_sql)
        load_csv_to_table(conn, "chartevents.csv", "chartevents", create_chartevents_sql)
        load_csv_to_table(conn, "d_icd_diagnoses.csv", "d_icd_diagnoses", create_d_icd_diagnoses_sql)

    except Exception as e:
        print(f"Database connection error: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()

