"""
Database layer — SQLite via SQLAlchemy.
Tables: users, quiz_questions, quiz_attempts, learning_progress,
        certificates, resume_data, announcements
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "ailearn.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    # ── Users ──────────────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        email       TEXT    UNIQUE NOT NULL,
        password    TEXT    NOT NULL,
        role        TEXT    NOT NULL DEFAULT 'student',
        avatar      TEXT    DEFAULT '',
        is_active   INTEGER DEFAULT 1,
        created_at  TEXT    DEFAULT (datetime('now'))
    )""")

    # ── Quiz questions ─────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS quiz_questions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        topic       TEXT    NOT NULL,
        question    TEXT    NOT NULL,
        option_a    TEXT,
        option_b    TEXT,
        option_c    TEXT,
        option_d    TEXT,
        answer      TEXT    NOT NULL,
        difficulty  TEXT    DEFAULT 'medium',
        qtype       TEXT    DEFAULT 'mcq',
        created_by  INTEGER,
        created_at  TEXT    DEFAULT (datetime('now'))
    )""")

    # ── Quiz attempts ──────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS quiz_attempts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL,
        topic       TEXT    NOT NULL,
        score       INTEGER NOT NULL,
        total       INTEGER NOT NULL,
        time_taken  INTEGER DEFAULT 0,
        attempted_at TEXT   DEFAULT (datetime('now'))
    )""")

    # ── Learning progress ──────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS learning_progress (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL,
        module      TEXT    NOT NULL,
        completed   INTEGER DEFAULT 0,
        progress_pct INTEGER DEFAULT 0,
        last_visited TEXT   DEFAULT (datetime('now')),
        UNIQUE(user_id, module)
    )""")

    # ── Certificates ───────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS certificates (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL,
        course      TEXT    NOT NULL,
        cert_id     TEXT    UNIQUE NOT NULL,
        issued_at   TEXT    DEFAULT (datetime('now'))
    )""")

    # ── Resume data ────────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS resume_data (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER UNIQUE NOT NULL,
        data_json   TEXT,
        ats_score   INTEGER DEFAULT 0,
        updated_at  TEXT    DEFAULT (datetime('now'))
    )""")

    # ── Announcements ──────────────────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS announcements (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        title       TEXT    NOT NULL,
        body        TEXT    NOT NULL,
        created_by  INTEGER,
        created_at  TEXT    DEFAULT (datetime('now'))
    )""")

    conn.commit()
    _seed_data(c, conn)
    conn.close()


def _seed_data(c, conn):
    """Insert default admin + sample quiz questions if DB is empty."""
    # Admin user
    c.execute("SELECT id FROM users WHERE email='manishankardey2005@gmail.com'")
    if not c.fetchone():
        pw = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute("INSERT INTO users(name,email,password,role) VALUES(?,?,?,?)",
                  ("Admin", "manishankardey2005@gmail.com", pw, "admin"))

    # Demo student
    c.execute("SELECT id FROM users WHERE email='demo@ailearn.com'")
    if not c.fetchone():
        pw = hashlib.sha256("demo123".encode()).hexdigest()
        c.execute("INSERT INTO users(name,email,password,role) VALUES(?,?,?,?)",
                  ("Demo Student", "demo@ailearn.com", pw, "student"))

    # Sample quiz questions
    c.execute("SELECT COUNT(*) FROM quiz_questions")
    if c.fetchone()[0] == 0:
        questions = [
            # ML Basics
            ("Machine Learning","What does ML stand for?","Machine Logic","Machine Learning","Model Learning","Modern Language","B","easy","mcq"),
            ("Machine Learning","Which algorithm is used for classification?","K-Means","Linear Regression","Decision Tree","PCA","C","easy","mcq"),
            ("Machine Learning","Overfitting means the model performs...","Well on train, poorly on test","Poorly on both","Well on test only","None","A","medium","mcq"),
            ("Machine Learning","Which metric is used for regression?","Accuracy","F1 Score","RMSE","Precision","C","medium","mcq"),
            ("Machine Learning","Random Forest is an ensemble of...","SVMs","Decision Trees","Neural Nets","KNN","B","easy","mcq"),
            # Deep Learning
            ("Deep Learning","A neural network layer that detects spatial features is called...","Dense","Dropout","Convolutional","Recurrent","C","medium","mcq"),
            ("Deep Learning","Backpropagation is used to...","Feed data forward","Update weights","Normalize inputs","None","B","medium","mcq"),
            ("Deep Learning","LSTM solves the problem of...","Overfitting","Vanishing gradient","Underfitting","Data leakage","B","hard","mcq"),
            ("Deep Learning","Which activation function outputs between 0 and 1?","ReLU","Tanh","Sigmoid","Softmax","C","easy","mcq"),
            ("Deep Learning","Transfer learning reuses...","New random weights","Pre-trained weights","Test data","Labels","B","medium","mcq"),
            # NLP
            ("NLP","TF-IDF stands for...","Term Frequency–Inverse Document Frequency","Text Feature Index–Doc","Term File Index Document","None","A","easy","mcq"),
            ("NLP","BERT is based on which architecture?","RNN","CNN","Transformer","LSTM","C","medium","mcq"),
            ("NLP","Tokenization means...","Training a model","Splitting text into tokens","Encoding labels","None","B","easy","mcq"),
            # Python
            ("Python","Which library is used for data manipulation?","NumPy","Pandas","Matplotlib","Seaborn","B","easy","mcq"),
            ("Python","List comprehension [x**2 for x in range(5)] produces...","[0,1,4,9,16]","[1,4,9,16,25]","[0,1,2,3,4]","Error","A","easy","mcq"),
        ]
        c.executemany(
            "INSERT INTO quiz_questions(topic,question,option_a,option_b,option_c,option_d,answer,difficulty,qtype) VALUES(?,?,?,?,?,?,?,?,?)",
            questions)
    conn.commit()


# ── Auth helpers ────────────────────────────────────────────────────────────

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def register_user(name, email, password):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users(name,email,password,role) VALUES(?,?,?,?)",
                  (name, email, hash_password(password), "student"))
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Email already registered."
    finally:
        conn.close()


def login_user(email, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=? AND password=? AND is_active=1",
              (email, hash_password(password)))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


# ── Progress helpers ─────────────────────────────────────────────────────────

def update_progress(user_id, module, pct):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO learning_progress(user_id,module,progress_pct,last_visited)
        VALUES(?,?,?,datetime('now'))
        ON CONFLICT(user_id,module) DO UPDATE SET
            progress_pct=excluded.progress_pct,
            completed=CASE WHEN excluded.progress_pct>=100 THEN 1 ELSE 0 END,
            last_visited=excluded.last_visited
    """, (user_id, module, pct))
    conn.commit()
    conn.close()


def get_progress(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM learning_progress WHERE user_id=?", (user_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ── Quiz helpers ─────────────────────────────────────────────────────────────

def get_questions(topic, limit=10):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM quiz_questions WHERE topic=? ORDER BY RANDOM() LIMIT ?",
              (topic, limit))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def save_attempt(user_id, topic, score, total, time_taken=0):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO quiz_attempts(user_id,topic,score,total,time_taken) VALUES(?,?,?,?,?)",
              (user_id, topic, score, total, time_taken))
    conn.commit()
    conn.close()


def get_attempts(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM quiz_attempts WHERE user_id=? ORDER BY attempted_at DESC", (user_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_leaderboard(topic=None):
    conn = get_connection()
    c = conn.cursor()
    if topic:
        c.execute("""
            SELECT u.name, MAX(a.score*100/a.total) as best_pct, COUNT(*) as attempts
            FROM quiz_attempts a JOIN users u ON a.user_id=u.id
            WHERE a.topic=?
            GROUP BY a.user_id ORDER BY best_pct DESC LIMIT 10
        """, (topic,))
    else:
        c.execute("""
            SELECT u.name, AVG(a.score*100.0/a.total) as avg_pct, COUNT(*) as attempts
            FROM quiz_attempts a JOIN users u ON a.user_id=u.id
            GROUP BY a.user_id ORDER BY avg_pct DESC LIMIT 10
        """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ── Certificate helpers ──────────────────────────────────────────────────────

def issue_certificate(user_id, course):
    import uuid
    cert_id = str(uuid.uuid4())[:8].upper()
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO certificates(user_id,course,cert_id) VALUES(?,?,?)",
                  (user_id, course, cert_id))
        conn.commit()
        return cert_id
    except sqlite3.IntegrityError:
        c.execute("SELECT cert_id FROM certificates WHERE user_id=? AND course=?", (user_id, course))
        row = c.fetchone()
        return row["cert_id"] if row else None
    finally:
        conn.close()


def get_certificates(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM certificates WHERE user_id=?", (user_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ── Admin helpers ─────────────────────────────────────────────────────────────

def get_all_students():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id,name,email,is_active,created_at FROM users WHERE role='student'")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def toggle_student(user_id, active):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE users SET is_active=? WHERE id=?", (1 if active else 0, user_id))
    conn.commit()
    conn.close()


def add_question(topic, question, a, b, c_, d, answer, difficulty, qtype, created_by):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO quiz_questions(topic,question,option_a,option_b,option_c,option_d,answer,difficulty,qtype,created_by)
        VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (topic, question, a, b, c_, d, answer, difficulty, qtype, created_by))
    conn.commit()
    conn.close()


def add_announcement(title, body, user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO announcements(title,body,created_by) VALUES(?,?,?)", (title, body, user_id))
    conn.commit()
    conn.close()


def get_announcements():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM announcements ORDER BY created_at DESC LIMIT 10")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def save_resume(user_id, data_json, ats_score):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO resume_data(user_id,data_json,ats_score,updated_at) VALUES(?,?,?,datetime('now'))
        ON CONFLICT(user_id) DO UPDATE SET data_json=excluded.data_json,ats_score=excluded.ats_score,updated_at=excluded.updated_at
    """, (user_id, data_json, ats_score))
    conn.commit()
    conn.close()


def get_resume(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM resume_data WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def platform_stats():
    conn = get_connection()
    c = conn.cursor()
    stats = {}
    c.execute("SELECT COUNT(*) FROM users WHERE role='student'"); stats["students"] = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM quiz_attempts"); stats["attempts"] = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM certificates"); stats["certs"] = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM quiz_questions"); stats["questions"] = c.fetchone()[0]
    conn.close()
    return stats
