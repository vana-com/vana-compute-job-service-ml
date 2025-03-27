from pathlib import Path
import sqlite3
import json
import os

# Paths to the database and output file
DB_PATH = Path(os.getenv("INPUT_PATH", "/mnt/input")) / "query_results.db"  # Default path to the SQLite database
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "/mnt/output")) / "stats.json"  # Default output JSON path

def get_user_stats():
    """Connects to the SQLite DB and retrieves user stats."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Enable row factory to get column names
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query all user stats from the query results
        cursor.execute('SELECT * FROM results')
        
        user_stats = []
        for row in cursor.fetchall():
            # Convert row to dictionary
            user_stats.append({
                "user_id": row['user_id'],
                "total_minutes": row['total_minutes'],
                "track_count": row['track_count'],
                "calculated_minutes": row['calculated_minutes']
            })
        
        conn.close()
        return user_stats
    except Exception as e:
        print(f"Error querying database: {e}")
        raise e

def create_buckets(user_stats):
    """Create aggregated statistics in buckets."""
    # Define the buckets
    minute_buckets = {
        "0-5000": 0,
        "5001-10000": 0,
        "10001-20000": 0,
        "20001+": 0
    }
    
    track_count_buckets = {
        "0-1": 0,
        "2-5": 0,
        "6-10": 0,
        "11+": 0
    }
    
    calculated_minutes_buckets = {
        "0-5": 0,
        "6-10": 0,
        "11-15": 0,
        "16+": 0
    }
    
    # Aggregate stats into buckets
    for user in user_stats:
        # Total minutes buckets
        total_mins = user["total_minutes"]
        if total_mins <= 5000:
            minute_buckets["0-5000"] += 1
        elif total_mins <= 10000:
            minute_buckets["5001-10000"] += 1
        elif total_mins <= 20000:
            minute_buckets["10001-20000"] += 1
        else:
            minute_buckets["20001+"] += 1
        
        # Track count buckets
        tracks = user["track_count"]
        if tracks <= 1:
            track_count_buckets["0-1"] += 1
        elif tracks <= 5:
            track_count_buckets["2-5"] += 1
        elif tracks <= 10:
            track_count_buckets["6-10"] += 1
        else:
            track_count_buckets["11+"] += 1
        
        # Calculated minutes buckets
        calc_mins = user["calculated_minutes"]
        if calc_mins <= 5:
            calculated_minutes_buckets["0-5"] += 1
        elif calc_mins <= 10:
            calculated_minutes_buckets["6-10"] += 1
        elif calc_mins <= 15:
            calculated_minutes_buckets["11-15"] += 1
        else:
            calculated_minutes_buckets["16+"] += 1
    
    return {
        "total_minutes_distribution": minute_buckets,
        "track_count_distribution": track_count_buckets,
        "calculated_minutes_distribution": calculated_minutes_buckets
    }

def save_stats_to_json(user_stats, output_path):
    """Saves individual user stats and aggregated stats to a JSON file."""
    try:
        # Generate bucketed stats
        aggregated_stats = create_buckets(user_stats)
        
        # Create final stats object with both individual and aggregated stats
        stats = {
            "users": user_stats,
            "aggregated_stats": aggregated_stats,
            "user_count": len(user_stats)
        }
        
        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Stats saved to {output_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

def main():
    print(f"Processing query results from {DB_PATH}")
    user_stats = get_user_stats()
    if user_stats:
        print(f"Found {len(user_stats)} users in the database")
        save_stats_to_json(user_stats, OUTPUT_PATH)
    else:
        print("No user stats found in the database")
        # Create an empty stats file to indicate processing completed
        save_stats_to_json([], OUTPUT_PATH)

if __name__ == "__main__":
    main()