-- Create the tables
CREATE TABLE users (
    user_id VARCHAR NOT NULL, 
    email VARCHAR NOT NULL, 
    name VARCHAR NOT NULL, 
    locale VARCHAR NOT NULL, 
    created_at DATETIME NOT NULL, 
    PRIMARY KEY (user_id), 
    UNIQUE (email)
);

CREATE TABLE auth_sources (
    auth_id INTEGER NOT NULL, 
    user_id VARCHAR NOT NULL, 
    source VARCHAR NOT NULL, 
    collection_date DATETIME NOT NULL, 
    data_type VARCHAR NOT NULL, 
    PRIMARY KEY (auth_id), 
    FOREIGN KEY(user_id) REFERENCES users (user_id)
);

CREATE TABLE storage_metrics (
    metric_id INTEGER NOT NULL, 
    user_id VARCHAR NOT NULL, 
    percent_used FLOAT NOT NULL, 
    recorded_at DATETIME NOT NULL, 
    PRIMARY KEY (metric_id), 
    FOREIGN KEY(user_id) REFERENCES users (user_id)
);

-- Insert dummy users
INSERT INTO users (user_id, email, name, locale, created_at) VALUES
('user1', 'john.doe@example.com', 'John Doe', 'en-US', '2023-01-15 10:30:00'),
('user2', 'jane.smith@example.com', 'Jane Smith', 'en-GB', '2023-02-20 14:45:00'),
('user3', 'hans.mueller@example.com', 'Hans Mueller', 'de-DE', '2023-03-10 08:15:00'),
('user4', 'maria.garcia@example.com', 'Maria Garcia', 'es-ES', '2023-04-05 16:20:00'),
('user5', 'takashi.yamamoto@example.com', 'Takashi Yamamoto', 'ja-JP', '2023-05-12 11:10:00'),
('user6', 'liu.wei@example.com', 'Liu Wei', 'zh-CN', '2023-06-18 09:30:00'),
('user7', 'emma.wilson@example.com', 'Emma Wilson', 'en-AU', '2023-07-22 15:40:00'),
('user8', 'pierre.dupont@example.com', 'Pierre Dupont', 'fr-FR', '2023-08-30 13:25:00'),
('user9', 'sofia.rossi@example.com', 'Sofia Rossi', 'it-IT', '2023-09-05 17:50:00'),
('user10', 'carlos.silva@example.com', 'Carlos Silva', 'pt-BR', '2023-10-11 12:15:00'),
('user11', 'olga.petrov@example.com', 'Olga Petrov', 'ru-RU', '2023-11-19 10:05:00'),
('user12', 'kim.ji-woo@example.com', 'Kim Ji-woo', 'ko-KR', '2023-12-24 08:40:00'),
('user13', 'mohamed.ali@example.com', 'Mohamed Ali', 'ar-SA', '2024-01-07 14:30:00'),
('user14', 'anna.andersson@example.com', 'Anna Andersson', 'sv-SE', '2024-02-13 16:15:00'),
('user15', 'david.kumar@example.com', 'David Kumar', 'en-IN', '2024-03-20 11:20:00'),
('user16', 'fatima.hassan@example.com', 'Fatima Hassan', 'ar-EG', '2024-04-25 13:40:00'),
('user17', 'nguyen.van@example.com', 'Nguyen Van', 'vi-VN', '2024-05-30 09:25:00'),
('user18', 'alex.brown@example.com', 'Alex Brown', 'en-CA', '2024-06-08 15:50:00'),
('user19', 'lars.hansen@example.com', 'Lars Hansen', 'da-DK', '2024-07-15 10:35:00'),
('user20', 'ana.santos@example.com', 'Ana Santos', 'pt-PT', '2024-08-22 12:10:00');

-- Insert dummy auth sources (some users have multiple auth sources)
INSERT INTO auth_sources (auth_id, user_id, source, collection_date, data_type) VALUES
(1, 'user1', 'google', '2023-01-16 08:30:00', 'oauth'),
(2, 'user1', 'facebook', '2023-02-10 14:20:00', 'oauth'),
(3, 'user2', 'apple', '2023-02-21 09:15:00', 'oauth'),
(4, 'user3', 'google', '2023-03-11 10:40:00', 'oauth'),
(5, 'user4', 'email', '2023-04-06 11:25:00', 'password'),
(6, 'user5', 'twitter', '2023-05-13 13:30:00', 'oauth'),
(7, 'user6', 'github', '2023-06-19 16:45:00', 'oauth'),
(8, 'user7', 'email', '2023-07-23 08:20:00', 'password'),
(9, 'user8', 'google', '2023-08-31 12:10:00', 'oauth'),
(10, 'user9', 'email', '2023-09-06 14:35:00', 'password'),
(11, 'user10', 'facebook', '2023-10-12 09:50:00', 'oauth'),
(12, 'user11', 'apple', '2023-11-20 15:15:00', 'oauth'),
(13, 'user13', 'email', '2024-01-08 10:25:00', 'password'),
(14, 'user14', 'github', '2024-02-14 13:40:00', 'oauth'),
(15, 'user15', 'google', '2024-03-21 16:30:00', 'oauth'),
(16, 'user16', 'email', '2024-04-26 09:10:00', 'password'),
(17, 'user18', 'apple', '2024-06-09 11:45:00', 'oauth'),
(18, 'user19', 'facebook', '2024-07-16 14:05:00', 'oauth'),
(19, 'user20', 'google', '2024-08-23 15:20:00', 'oauth'),
(20, 'user2', 'github', '2023-05-15 13:30:00', 'oauth');

-- Insert dummy storage metrics (not all users have storage metrics)
INSERT INTO storage_metrics (metric_id, user_id, percent_used, recorded_at) VALUES
(1, 'user1', 65.2, '2023-06-10 09:45:00'),
(2, 'user2', 12.7, '2023-07-15 14:30:00'),
(3, 'user3', 89.5, '2023-08-20 11:20:00'),
(4, 'user5', 45.3, '2023-09-25 16:40:00'),
(5, 'user6', 72.8, '2023-10-30 13:15:00'),
(6, 'user7', 18.1, '2023-11-05 10:50:00'),
(7, 'user9', 56.4, '2023-12-10 15:25:00'),
(8, 'user10', 33.9, '2024-01-15 12:30:00'),
(9, 'user11', 92.0, '2024-02-20 09:10:00'),
(10, 'user13', 27.6, '2024-03-25 14:45:00'),
(11, 'user14', 61.3, '2024-04-30 11:35:00'),
(12, 'user16', 48.7, '2024-05-05 16:20:00'),
(13, 'user18', 75.9, '2024-06-10 13:05:00'),
(14, 'user19', 22.3, '2024-07-15 10:40:00'),
(15, 'user20', 51.2, '2024-08-20 15:15:00');

-- Create the results table to store the query result
CREATE TABLE results AS
SELECT
    u.user_id,
    u.locale,
    a.source AS auth_source,
    a.data_type AS auth_data_type,
    sm.metric_id AS storage_metric_id
FROM users AS u
LEFT JOIN auth_sources AS a ON a.user_id = u.user_id
LEFT JOIN storage_metrics AS sm ON sm.user_id = u.user_id; 