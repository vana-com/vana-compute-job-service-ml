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
                "locale": row['locale'],
                "auth_source": row['auth_source'],
                "auth_data_type": row['auth_data_type'],
                "storage_metric_id": row['storage_metric_id']
            })
        
        conn.close()
        return user_stats
    except Exception as e:
        print(f"Error querying database: {e}")
        raise e

def create_aggregations(user_stats):
    """Create aggregated statistics for the new schema."""
    # Aggregations for locale
    locale_distribution = {}
    
    # Aggregations for auth sources
    auth_source_distribution = {}
    
    # Aggregations for auth data types
    auth_data_type_distribution = {}
    
    # Count users with storage metrics
    users_with_storage_metrics = 0
    
    # Process each user record
    for user in user_stats:
        # Aggregate locale stats
        locale = user["locale"]
        if locale in locale_distribution:
            locale_distribution[locale] += 1
        else:
            locale_distribution[locale] = 1
        
        # Aggregate auth source stats
        auth_source = user["auth_source"]
        if auth_source:  # Handle None values from LEFT JOIN
            if auth_source in auth_source_distribution:
                auth_source_distribution[auth_source] += 1
            else:
                auth_source_distribution[auth_source] = 1
        
        # Aggregate auth data type stats
        auth_data_type = user["auth_data_type"]
        if auth_data_type:  # Handle None values from LEFT JOIN
            if auth_data_type in auth_data_type_distribution:
                auth_data_type_distribution[auth_data_type] += 1
            else:
                auth_data_type_distribution[auth_data_type] = 1
        
        # Count users with storage metrics
        if user["storage_metric_id"] is not None:
            users_with_storage_metrics += 1
    
    return {
        "locale_distribution": locale_distribution,
        "auth_source_distribution": auth_source_distribution,
        "auth_data_type_distribution": auth_data_type_distribution,
        "users_with_storage_metrics": users_with_storage_metrics
    }

def save_stats_to_json(user_stats, output_path):
    """Saves individual user stats and aggregated stats to a JSON file."""
    try:
        # Generate aggregated stats
        aggregated_stats = create_aggregations(user_stats)
        
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