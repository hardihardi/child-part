import sqlite3
import datetime
from werkzeug.security import generate_password_hash
import os

DB_NAME = 'database.db'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Users Table (RBAC)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama TEXT,
            foto TEXT DEFAULT 'default.png',
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    
    # Auto migrate if nama doesn't exist
    try:
        cursor.execute("ALTER TABLE Users ADD COLUMN nama TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
        
    # Auto migrate if foto doesn't exist
    try:
        cursor.execute("ALTER TABLE Users ADD COLUMN foto TEXT DEFAULT 'default.png'")
    except sqlite3.OperationalError:
        pass # Column already exists
    
    # 2. Inspection Log Table (Updated to match PRD)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Inspection_Log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            raw_image_path TEXT,
            status_result TEXT NOT NULL,
            confidence_score REAL,
            operator_id INTEGER,
            FOREIGN KEY (operator_id) REFERENCES Users (id)
        )
    ''')
    
    # 3. Activity Log Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Activity_Log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id INTEGER,
            action TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES Users (id)
        )
    ''')
    
    # 4. Feature Suggestion Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Feature_Suggestion (
            suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES Users(id)
        )
    ''')

    # 5. Light Profile Table (Updated to match PRD)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Light_Profile (
            profile_id INTEGER PRIMARY KEY CHECK (profile_id = 1),
            target_klip_lh INTEGER DEFAULT 1,
            target_klip_rh INTEGER DEFAULT 1,
            log_delay_seconds INTEGER DEFAULT 5,
            lux_level INTEGER DEFAULT 50,
            exposure_setting INTEGER DEFAULT 0,
            buzzer_enabled INTEGER DEFAULT 1,
            ai_conf_threshold REAL DEFAULT 0.4,
            ai_nms_threshold REAL DEFAULT 0.4,
            model_version TEXT DEFAULT 'YOLOv5s-ONNX v1.0',
            app_name TEXT DEFAULT 'IkuyoVision',
            camera_source INTEGER DEFAULT 0
        )
    ''')
    
    # Auto migrate if camera_source doesn't exist
    try:
        cursor.execute("ALTER TABLE Light_Profile ADD COLUMN camera_source INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Column already exists
        
    # Auto migrate if buzzer_ok_enabled doesn't exist
    try:
        cursor.execute("ALTER TABLE Light_Profile ADD COLUMN buzzer_ok_enabled INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Column already exists
    
    # Insert default Admin if none exists
    cursor.execute('SELECT COUNT(*) FROM Users')
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            'INSERT INTO Users (username, password_hash, role) VALUES (?, ?, ?)',
            ('admin', generate_password_hash('admin123'), 'Admin')
        )
        cursor.execute(
            'INSERT INTO Users (username, password_hash, role) VALUES (?, ?, ?)',
            ('qc', generate_password_hash('qc123'), 'QC')
        )
        cursor.execute(
            'INSERT INTO Users (username, password_hash, role) VALUES (?, ?, ?)',
            ('operator', generate_password_hash('operator123'), 'Operator')
        )
        
    # Insert default settings if none exists
    cursor.execute('SELECT COUNT(*) FROM Light_Profile')
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO Light_Profile (profile_id, target_klip_lh, target_klip_rh, log_delay_seconds, lux_level, exposure_setting, buzzer_enabled, ai_conf_threshold, ai_nms_threshold, model_version, app_name, camera_source) 
            VALUES (1, 1, 1, 5, 60, 0, 1, 0.4, 0.4, 'YOLOv5s-ONNX v1.0', 'IkuyoVision', 0)
        ''')

    conn.commit()
    conn.close()

def log_activity(user_id, action):
    conn = get_db_connection()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute('INSERT INTO Activity_Log (timestamp, user_id, action) VALUES (?, ?, ?)', (timestamp, user_id, action))
    conn.commit()
    conn.close()

def log_inspection(status_result, confidence_score=0.0, raw_image_path=None, operator_id=None):
    conn = get_db_connection()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        'INSERT INTO Inspection_Log (timestamp, raw_image_path, status_result, confidence_score, operator_id) VALUES (?, ?, ?, ?, ?)',
        (timestamp, raw_image_path, status_result, confidence_score, operator_id)
    )
    conn.commit()
    conn.close()

def get_config():
    conn = get_db_connection()
    row = conn.execute('SELECT * FROM Light_Profile WHERE profile_id = 1').fetchone()
    conn.close()
    if row:
        return dict(row)
    return {'target_klip_lh': 1, 'target_klip_rh': 1, 'log_delay_seconds': 5, 'lux_level': 50, 'exposure_setting': 0, 'buzzer_enabled': 1, 'buzzer_ok_enabled': 0, 'ai_conf_threshold': 0.4, 'ai_nms_threshold': 0.4, 'model_version': 'YOLOv5s-ONNX v1.0', 'app_name': 'IkuyoVision'}

def clear_data_folder():
    import shutil
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'static', 'data')
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error removing {item_path}: {e}")
    else:
        os.makedirs(data_dir, exist_ok=True)

if __name__ == '__main__':
    # Force reset DB for this iteration
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    clear_data_folder()
    init_db()
    print("Database and data folder aligned with PRD Data Model.")

