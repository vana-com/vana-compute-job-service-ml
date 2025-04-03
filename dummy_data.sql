-- Create the tables
CREATE TABLE unwrapped_user_stats (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_minutes INTEGER,
    track_count INTEGER,
    unique_artists TEXT,
    activity_period_days INTEGER,
    first_listen TEXT,
    last_listen TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Insert dummy users
INSERT INTO unwrapped_user_stats (user_id, total_minutes, track_count, unique_artists, activity_period_days, first_listen, last_listen) VALUES
(1, 100, 50, 'Artist1, Artist2', 30, '2023-01-15 10:30:00', '2023-02-20 14:45:00'),
(2, 150, 75, 'Artist3, Artist4', 45, '2023-02-20 14:45:00', '2023-03-10 08:15:00'),
(3, 200, 100, 'Artist5, Artist6', 60, '2023-03-10 08:15:00', '2023-04-05 16:20:00'),
(4, 250, 125, 'Artist7, Artist8', 75, '2023-04-05 16:20:00', '2023-05-12 11:10:00'),
(5, 300, 150, 'Artist9, Artist10', 90, '2023-05-12 11:10:00', '2023-06-18 09:30:00'),
(6, 350, 175, 'Artist11, Artist12', 105, '2023-06-18 09:30:00', '2023-07-22 15:40:00'),
(7, 400, 200, 'Artist13, Artist14', 120, '2023-07-22 15:40:00', '2023-08-30 13:25:00'),
(8, 450, 225, 'Artist15, Artist16', 135, '2023-08-30 13:25:00', '2023-09-05 17:50:00'),
(9, 500, 250, 'Artist17, Artist18', 150, '2023-09-05 17:50:00', '2023-10-11 12:15:00'),
(10, 550, 275, 'Artist19, Artist20', 165, '2023-10-11 12:15:00', '2023-11-19 10:05:00');

-- Create the results table to store the query result
CREATE TABLE results AS
SELECT
    u.user_id,
    u.total_minutes
FROM unwrapped_user_stats AS u; 