"""
database/db_manager.py
SQLite database manager — stores transaction IDs, image hashes,
and verification history.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "detections.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_id      TEXT NOT NULL UNIQUE,
            image_hash  TEXT,
            verified_at TEXT,
            result      TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS image_hashes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            phash       TEXT NOT NULL,
            txn_id      TEXT,
            added_at    TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS verification_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path      TEXT,
            final_result    TEXT,
            confidence      REAL,
            forensics_score REAL,
            ml_score        REAL,
            is_duplicate    INTEGER,
            checked_at      TEXT
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Transaction ID operations
# ---------------------------------------------------------------------------

def transaction_exists(txn_id: str) -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM transactions WHERE txn_id = ?", (txn_id,)
    ).fetchone()
    conn.close()
    return row is not None


def save_transaction(txn_id: str, image_hash: str = None, result: str = None):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO transactions (txn_id, image_hash, verified_at, result) VALUES (?, ?, ?, ?)",
            (txn_id, image_hash, datetime.utcnow().isoformat(), result)
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Image hash operations
# ---------------------------------------------------------------------------

def get_all_hashes() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT phash, txn_id FROM image_hashes").fetchall()
    conn.close()
    return [{"hash": r["phash"], "transaction_id": r["txn_id"]} for r in rows]


def save_hash(phash: str, txn_id: str = None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO image_hashes (phash, txn_id, added_at) VALUES (?, ?, ?)",
        (phash, txn_id, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Verification log
# ---------------------------------------------------------------------------

def log_verification(image_path: str, final_result: str, confidence: float,
                     forensics_score: float, ml_score: float, is_duplicate: bool):
    conn = get_connection()
    conn.execute(
        """INSERT INTO verification_log
           (image_path, final_result, confidence, forensics_score, ml_score, is_duplicate, checked_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (image_path, final_result, confidence, forensics_score,
         ml_score, int(is_duplicate), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_recent_verifications(limit: int = 20) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM verification_log ORDER BY checked_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
