from pathlib import Path
import sqlite3
import json
import os

# Paths to the database and output file
DB_PATH = Path(os.getenv("INPUT_PATH", "/mnt/input")) / "query_results.db"  # Default path to the SQLite database
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "/mnt/output")) / "stats.json"  # Default output JSON path

def aggregate_user_stats():
    """Connects to the SQLite DB and performs aggregation on user listening stats."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Aggregate stats per user
        cursor.execute('''
            SELECT 
                u.user_id, 
                u.total_minutes, 
                u.track_count, 
                u.unique_artists, 
                u.activity_period_days, 
                u.first_listen, 
                u.last_listen,
                COUNT(t.id) AS total_tracks_played,
                SUM(t.duration_ms) / 60000 AS total_listening_minutes, 
                COUNT(DISTINCT t.artist_id) AS distinct_artists_played
            FROM unwrapped_user_stats u
            LEFT JOIN unwrapped_user_tracks t ON u.user_id = t.user_id
            GROUP BY u.user_id
        ''')
        
        user_stats = []
        for row in cursor.fetchall():
            user_stats.append({
                "user_id": row[0],
                "total_minutes_reported": row[1],
                "track_count": row[2],
                "unique_artists": json.loads(row[3]) if row[3] else [],  # Deserialize stored array
                "activity_period_days": row[4],
                "first_listen": row[5],
                "last_listen": row[6],
                "total_tracks_played": row[7],
                "total_listening_minutes": row[8],
                "distinct_artists_played": row[9]
            })
        
        conn.close()
        return user_stats
    except Exception as e:
        print(f"Error querying database: {e}")
        return []

def save_stats_to_json(stats, output_path):
    """Saves aggregated stats to a JSON file."""
    try:
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Aggregated stats saved to {output_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

def main():
    stats = aggregate_user_stats()
    save_stats_to_json(stats, OUTPUT_PATH)

if __name__ == "__main__":
    main()