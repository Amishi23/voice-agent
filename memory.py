import sqlite3
from datetime import datetime

DB_PATH = "conversations.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            session   TEXT NOT NULL,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (session, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def get_history(session_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session = ? ORDER BY id",
        (session_id,)
    ).fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def clear_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE session = ?", (session_id,))
    conn.commit()
    conn.close()