import psycopg2

# --- Configuration ---
# Ensure these match your actual database credentials
DB_NAME = "my_postgres"
DB_USER = "postgres"
DB_PASS = "MyStrongPassword123!"
DB_HOST = "localhost"
DB_PORT = "5432"

def run_diagnostics():
    """Connects to the DB and runs diagnostic queries on the chartevents table."""
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        cur = conn.cursor()

        print("--- Running ChartEvents Diagnostics ---")

        # Query 1: Get the total number of rows in chartevents
        print("1. Checking total rows in chartevents...")
        cur.execute("SELECT COUNT(*) FROM chartevents;")
        total_rows = cur.fetchone()[0]
        print(f"   Result: Found {total_rows:,} total rows.\n")

        if total_rows == 0:
            print("   STOP: The chartevents table is empty. Please re-run the data loader.")
            return

        # Query 2: Check for our specific heart rate item IDs
        target_ids = (220045, 211)
        print(f"2. Checking for specific Heart Rate item IDs {target_ids}...")
        cur.execute("SELECT COUNT(*) FROM chartevents WHERE itemid IN %s;", (target_ids,))
        hr_rows = cur.fetchone()[0]
        print(f"   Result: Found {hr_rows:,} rows with these specific IDs.\n")

        if hr_rows > 0:
            print("   CONCLUSION: The item IDs are correct, but the join or time window logic in the main query is failing.")
        else:
            print("   CONCLUSION: The assumed Heart Rate item IDs do not exist in your data.")
            # Query 3: If no specific IDs were found, find the most common ones
            print("3. Finding the TOP 10 most common item IDs in chartevents...")
            cur.execute("""
                SELECT itemid, COUNT(*)
                FROM chartevents
                GROUP BY itemid
                ORDER BY COUNT(*) DESC
                LIMIT 10;
            """)
            top_items = cur.fetchall()
            print("   Result: The most common item IDs are:")
            for item in top_items:
                print(f"     - Item ID: {item[0]} (found {item[1]:,} times)")
            print("\n   NEXT STEP: We should use one of these top IDs in our main SQL query.")
        
        print("\n--- Diagnostics Complete ---")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    run_diagnostics()
