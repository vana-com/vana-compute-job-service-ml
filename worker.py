from pathlib import Path
import sqlite3
import json
import os

# Paths to the database and output file
DB_PATH = Path(os.getenv("INPUT_PATH", "/mnt/input")) / "query_results.db"  # Default path to the SQLite database
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "/mnt/output")) / "stats.json"  # Default output JSON path
def logDbContents(file_path: str) -> None:
    """
    Log debug information about the downloaded SQLite database file.
    
    Args:
        file_path: Path to the SQLite database file
    """
    
    try:
        # Check if file exists
        if os.path.exists(file_path):
            print(f"[DEBUG] SQLite file validation - File exists: {file_path}")
            
            # Get file size
            file_size = os.path.getsize(file_path)
            print(f"[DEBUG] SQLite file size: {file_size} bytes ({file_size / 1024:.2f} KB)")
            
            # Connect to database and get schema
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"[DEBUG] SQLite tables: {tables}")
            
            # For each table, get schema and sample rows
            for table in tables:
                table_name = table[0]
                
                # Get schema
                cursor.execute(f"PRAGMA table_info({table_name});")
                schema = cursor.fetchall()
                print(f"[DEBUG] Schema for table '{table_name}': {schema}")
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                print(f"[DEBUG] Row count for table '{table_name}': {row_count}")
                
                # Get sample rows (up to 5)
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                rows = cursor.fetchall()
                print(f"[DEBUG] Sample rows for table '{table_name}' (up to 5): {rows}")
            
            conn.close()
        else:
            print(f"[DEBUG] WARNING: SQLite file does not exist: {file_path}")
    except Exception as e:
        print(f"[DEBUG]  WARNING: Error inspecting SQLite database: {str(e)}")
        # Don't raise the exception - this is just debug logging

def get_user_locales():
    """Connects to the SQLite DB and retrieves user_id and locale."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Query user_id and locale from the results table
        cursor.execute('SELECT user_id, locale FROM results')
        
        # Create a dictionary with user_id as keys and locale as values
        user_locales = {}
        for row in cursor.fetchall():
            user_id, locale = row
            user_locales[str(user_id)] = locale
        
        conn.close()
        return user_locales
    except Exception as e:
        print(f"Error querying database: {e}")
        raise e

def save_stats_to_json(user_locales, output_path):
    """Saves the user_id: locale mapping to a JSON file."""
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(user_locales, f, indent=4)
        print(f"Stats saved to {output_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

def main():
    print(f"Processing query results from {DB_PATH}")

    logDbContents(DB_PATH)

    user_locales = get_user_locales()
    if user_locales:
        print(f"Found {len(user_locales)} users in the database")
        save_stats_to_json(user_locales, OUTPUT_PATH)
    else:
        print("No user stats found in the database")
        # Create an empty stats file to indicate processing completed
        save_stats_to_json({}, OUTPUT_PATH)

if __name__ == "__main__":
    main()