from pathlib import Path
import sqlite3
import json
import os

# Paths to the database and output file
DB_PATH = Path(os.getenv("INPUT_PATH", "/mnt/input")) / "query_results.db"  # Default path to the SQLite database
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "/mnt/output")) / "stats.json"  # Default output JSON path

def get_user_minutes():
    """Connects to the SQLite DB and retrieves user_id and total_minutes."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Query user_id and total_minutes from the results table
        cursor.execute('SELECT user_id, total_minutes FROM results')
        
        # Create a dictionary with user_id as keys and total_minutes as values
        user_minutes = {}
        for row in cursor.fetchall():
            user_id, total_minutes = row
            user_minutes[str(user_id)] = total_minutes
        
        conn.close()
        return user_minutes
    except Exception as e:
        print(f"Error querying database: {e}")
        raise e

def save_stats_to_json(user_minutes, output_path):
    """Saves the user_id: total_minutes mapping to a JSON file."""
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(user_minutes, f, indent=4)
        print(f"Stats saved to {output_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

def main():
    print(f"Processing query results from {DB_PATH}")
    user_minutes = get_user_minutes()
    if user_minutes:
        print(f"Found {len(user_minutes)} users in the database")
        save_stats_to_json(user_minutes, OUTPUT_PATH)
    else:
        print("No user stats found in the database")
        # Create an empty stats file to indicate processing completed
        save_stats_to_json({}, OUTPUT_PATH)

if __name__ == "__main__":
    main()