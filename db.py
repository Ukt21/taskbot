import sqlite3
from typing import List, Dict, Any

DB_PATH = "tasks.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            period TEXT CHECK(period IN ('day','week','month')) NOT NULL,
            done INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def add_task(user_id: int, title: str, period: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO tasks (user_id, title, period)
        VALUES (?, ?, ?)
    """, (user_id, title, period))
    conn.commit()
    conn.close()


def get_tasks(user_id: int, period: str | None = None, only_active: bool = True) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    query = "SELECT id, title, period, done FROM tasks WHERE user_id = ?"
    params = [user_id]

    if period:
        query += " AND period = ?"
        params.append(period)

    if only_active:
        query += " AND done = 0"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    return [
        {"id": r[0], "title": r[1], "period": r[2], "done": bool(r[3])}
        for r in rows
    ]


def set_task_done(task_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE tasks SET done = 1 WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()


def delete_task(task_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
