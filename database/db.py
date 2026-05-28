"""
Database layer — Cloud-hosted PostgreSQL via Supabase.
Tables: users, quiz_questions, quiz_attempts, learning_progress,
        certificates, resume_data, announcements
"""

import hashlib
import os
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

# 1. Fetch connection string securely from environment variables or Streamlit secrets
DATABASE_URL = os.environ.get("DATABASE_URL") or st.secrets.get("DATABASE_URL")

@contextmanager
def get_connection():
    """Context manager for handling safe PostgreSQL database connections with DictCursor."""
    if not DATABASE_URL:
        st.error("Missing DATABASE_URL configuration. Please set it up in secrets.toml.")
        st.stop()
    
    # RealDictCursor makes rows act like dictionaries, matching your old row["column"] structure
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_db():
    """Creates the relational schema configured for PostgreSQL syntax."""
    with get_connection() as conn:
        with conn.cursor() as c:
            # ── Users ──────────────────────────────────────────────────────────────
            c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          SERIAL PRIMARY KEY,
                name        TEXT    NOT NULL,
                email       TEXT    UNIQUE NOT NULL,
                password    TEXT    NOT NULL,
                role        TEXT    NOT NULL DEFAULT 'student',
                avatar      TEXT    DEFAULT '',
                is_active   INTEGER DEFAULT 1,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );""")

            # ── Quiz questions ─────────────────────────────────────────────────────
            c.execute("""
            CREATE TABLE IF NOT EXISTS quiz_questions (
                id          SERIAL PRIMARY KEY,
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
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );""")

            # ── Quiz attempts ──────────────────────────────────────────────────────
            c.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id          SERIAL PRIMARY KEY,
                user_id     INTEGER NOT NULL,
                topic       TEXT    NOT NULL,
                score       INTEGER NOT NULL,
                total       INTEGER NOT NULL,
                time_taken  INTEGER DEFAULT 0,
                attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );""")

            # ── Learning progress ──────────────────────────────────────────────────
            c.execute("""
            CREATE TABLE IF NOT EXISTS learning_progress (
                id          SERIAL PRIMARY KEY,
                user_id     INTEGER NOT NULL,
                module      TEXT    NOT NULL,
                completed   INTEGER DEFAULT 0,
                progress_pct INTEGER DEFAULT 0,
                last_visited TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, module)
            );""")

            # ── Certificates ───────────────────────────────────────────────────────
            c.execute("""
            CREATE TABLE IF NOT EXISTS certificates (
                id          SERIAL PRIMARY KEY,
                user_id     INTEGER NOT NULL,
                course      TEXT    NOT NULL,
                cert_id     TEXT    UNIQUE NOT NULL,
                issued_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );""")

            # ── Resume data ────────────────────────────────────────────────────────
            c.execute("""
            CREATE TABLE IF NOT EXISTS resume_data (
                id          SERIAL PRIMARY KEY,
                user_id     INTEGER UNIQUE NOT NULL,
                data_json   TEXT,
                ats_score   INTEGER DEFAULT 0,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );""")

            # ── Announcements ──────────────────────────────────────────────────────
            c.execute("""
            CREATE TABLE IF NOT EXISTS announcements (
                id          SERIAL PRIMARY KEY,
                title       TEXT    NOT NULL,
                body        TEXT    NOT NULL,
                created_by  INTEGER,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );""")
            
            # Run the data seeder
            _seed_data(c)


def _seed_data(c):
    """Insert default admin if DB is empty."""
    c.execute("SELECT id FROM users WHERE email='manishankardey2005@gmail.com';")
    if not c.fetchone():
        pw = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute(
            "INSERT INTO users(name, email, password, role) VALUES (%s, %s, %s, %s);",
            ("Admin", "manishankardey2005@gmail.com", pw, "admin")
        )


# ── Auth helpers ────────────────────────────────────────────────────────────

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def register_user(name, email, password):
    try:
        with get_connection() as conn:
            with conn.cursor() as c:
                c.execute(
                    "INSERT INTO users(name, email, password, role) VALUES (%s, %s, %s, %s);",
                    (name, email.lower().strip(), hash_password(password), "student")
                )
        return True, "Registration successful!"
    except psycopg2.IntegrityError:
        return False, "Email already registered."
    except Exception as e:
        return False, f"Database error: {str(e)}"


def login_user(email, password):
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute(
                "SELECT * FROM users WHERE email = %s AND password = %s AND is_active = 1;",
                (email.lower().strip(), hash_password(password))
            )
            row = c.fetchone()
            return dict(row) if row else None


# ── Progress helpers ─────────────────────────────────────────────────────────

def update_progress(user_id, module, pct):
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO learning_progress(user_id, module, progress_pct, last_visited)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, module) DO UPDATE SET
                    progress_pct = EXCLUDED.progress_pct,
                    completed = CASE WHEN EXCLUDED.progress_pct >= 100 THEN 1 ELSE 0 END,
                    last_visited = CURRENT_TIMESTAMP;
            """, (user_id, module, pct))


def get_progress(user_id):
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute("SELECT * FROM learning_progress WHERE user_id = %s;", (user_id,))
            return [dict(r) for r in c.fetchall()]


# ── Quiz helpers ─────────────────────────────────────────────────────────────

def get_questions(topic, limit=50):
    with get_connection() as conn:
        with conn.cursor() as c:
            # PostgreSQL uses RANDOM() just like SQLite
            c.execute(
                "SELECT * FROM quiz_questions WHERE topic = %s ORDER BY RANDOM() LIMIT %s;",
                (topic, limit)
            )
            return [dict(r) for r in c.fetchall()]


def save_attempt(user_id, topic, score, total, time_taken=0):
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute(
                "INSERT INTO quiz_attempts(user_id, topic, score, total, time_taken) VALUES (%s, %s, %s, %s, %s);",
                (user_id, topic, score, total, time_taken)
            )


def get_attempts(user_id):
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute("SELECT * FROM quiz_attempts WHERE user_id = %s ORDER BY attempted_at DESC;", (user_id,))
            return [dict(r) for r in c.fetchall()]


def get_leaderboard(topic=None):
    with get_connection() as conn:
        with conn.cursor() as c:
            if topic:
                c.execute("""
                    SELECT u.name, MAX(a.score * 100 / a.total) as best_pct, COUNT(*) as attempts
                    FROM quiz_attempts a JOIN users u ON a.user_id = u.id
                    WHERE a.topic = %s
                    GROUP BY a.user_id, u.name ORDER BY best_pct DESC LIMIT 50;
                """, (topic,))
            else:
                c.execute("""
                    SELECT u.name, AVG(a.score * 100.0 / a.total) as avg_pct, COUNT(*) as attempts
                    FROM quiz_attempts a JOIN users u ON a.user_id = u.id
                    GROUP BY a.user_id, u.name ORDER BY avg_pct DESC LIMIT 50;
                """)
            return [dict(r) for r in c.fetchall()]


# ── Certificate helpers ──────────────────────────────────────────────────────

def issue_certificate(user_id, course):
    import uuid
    cert_id = str(uuid.uuid4())[:8].upper()
    try:
        with get_connection() as conn:
            with conn.cursor() as c:
                c.execute(
                    "INSERT INTO certificates(user_id, course, cert_id) VALUES (%s, %s, %s);",
                    (user_id, course, cert_id)
                )
        return cert_id
    except psycopg2.IntegrityError:
        with get_connection() as conn:
            with conn.cursor() as c:
                c.execute("SELECT cert_id FROM certificates WHERE user_id = %s AND course = %s;", (user_id, course))
                row = c.fetchone()
                return row["cert_id"] if row else None


def get_certificates(user_id):
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute("SELECT * FROM certificates WHERE user_id = %s;", (user_id,))
            return [dict(r) for r in c.fetchall()]


# ── Admin helpers ─────────────────────────────────────────────────────────────

def get_all_students():
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute("SELECT id, name, email, is_active, created_at FROM users WHERE role = 'student';")
            return [dict(r) for r in c.fetchall()]


def toggle_student(user_id, active):
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute("UPDATE users SET is_active = %s WHERE id = %s;", (1 if active else 0, user_id))


def add_question(topic, question, a, b, c_, d, answer, difficulty, qtype, created_by):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO quiz_questions(topic, question, option_a, option_b, option_c, option_d, answer, difficulty, qtype, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (topic, question, a, b, c_, d, answer, difficulty, qtype, created_by))


def add_announcement(title, body, user_id):
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute("INSERT INTO announcements(title, body, created_by) VALUES (%s, %s, %s);", (title, body, user_id))


def get_announcements():
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute("SELECT * FROM announcements ORDER BY created_at DESC LIMIT 50;")
            return [dict(r) for r in c.fetchall()]


def save_resume(user_id, data_json, ats_score):
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO resume_data(user_id, data_json, ats_score, updated_at) 
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET 
                    data_json = EXCLUDED.data_json,
                    ats_score = EXCLUDED.ats_score,
                    updated_at = CURRENT_TIMESTAMP;
            """, (user_id, data_json, ats_score))


def get_resume(user_id):
    with get_connection() as conn:
        with conn.cursor() as c:
            c.execute("SELECT * FROM resume_data WHERE user_id = %s;", (user_id,))
            row = c.fetchone()
            return dict(row) if row else None


def platform_stats():
    with get_connection() as conn:
        with conn.cursor() as c:
            stats = {}
            c.execute("SELECT COUNT(*) FROM users WHERE role = 'student';")
            stats["students"] = c.fetchone()["count"]
            
            c.execute("SELECT COUNT(*) FROM quiz_attempts;")
            stats["attempts"] = c.fetchone()["count"]
            
            c.execute("SELECT COUNT(*) FROM certificates;")
            stats["certs"] = c.fetchone()["count"]
            
            c.execute("SELECT COUNT(*) FROM quiz_questions;")
            stats["questions"] = c.fetchone()["count"]
            return stats