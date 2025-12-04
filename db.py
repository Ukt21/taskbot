import sqlite3
from typing import List, Dict, Any, Optional

DB_PATH = "tasks.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Базовая таблица, если её ещё нет
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            period TEXT NOT NULL,
            raw_deadline TEXT,
            category TEXT,
            priority INTEGER,
            done INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );
    """)

    # Таблица для фиксации, что ежедневный отчёт уже отправлен
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_reports (
            user_id INTEGER PRIMARY KEY,
            last_report_date TEXT
        );
    """)

    # Простейшая миграция на случай старой схемы
    # (если колонка уже есть — будет OperationalError, игнорируем)
    migrate_columns = [
        ("description", "TEXT"),
        ("raw_deadline", "TEXT"),
        ("category", "TEXT"),
        ("priority", "INTEGER"),
        ("completed_at", "TIMESTAMP"),
    ]
    for col, col_type in migrate_columns:
        try:
            cur.execute(f"ALTER TABLE tasks ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()


def add_task(
    user_id: int,
    title: str,
    period: str,
    description: Optional[str] = None,
    raw_deadline: Optional[str] = None,
    category: Optional[str] = None,
    priority: Optional[int] = None,
):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tasks (user_id, title, description, period, raw_deadline, category, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, title, description, period, raw_deadline, category, priority),
    )
    conn.commit()
    conn.close()


def get_tasks(
    user_id: int,
    period: Optional[str] = None,
    only_active: bool = True,
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    query = "SELECT * FROM tasks WHERE user_id = ?"
    params = [user_id]

    if period and period != "all":
        query += " AND period = ?"
        params.append(period)

    if only_active:
        query += " AND done = 0"

    query += " ORDER BY created_at ASC, id ASC"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_done_tasks(user_id: int, period: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    query = "SELECT * FROM tasks WHERE user_id = ? AND done = 1"
    params = [user_id]

    if period and period != "all":
        query += " AND period = ?"
        params.append(period)

    query += " ORDER BY completed_at DESC"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]


def set_task_done(task_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE tasks SET done = 1, completed_at = CURRENT_TIMESTAMP WHERE id = ?",
        (task_id,),
    )
    conn.commit()
    conn.close()


def update_task_title(task_id: int, new_title: str, new_description: Optional[str] = None):
    conn = get_connection()
    cur = conn.cursor()
    if new_description is not None:
        cur.execute(
            "UPDATE tasks SET title = ?, description = ? WHERE id = ?",
            (new_title, new_description, task_id),
        )
    else:
        cur.execute(
            "UPDATE tasks SET title = ? WHERE id = ?",
            (new_title, task_id),
        )
    conn.commit()
    conn.close()


def delete_task(task_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()


def get_all_user_ids() -> List[int]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT user_id FROM tasks")
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_daily_summary(user_id: int) -> Dict[str, Any]:
    """
    Возвращает сводку по пользователю:
    - активные задачи по периодам
    - выполнено сегодня
    """
    conn = get_connection()
    cur = conn.cursor()

    summary = {
        "active": {"day": 0, "week": 0, "month": 0},
        "done_today": 0,
    }

    # активные по периодам
    cur.execute(
        """
        SELECT period, COUNT(*) 
        FROM tasks 
        WHERE user_id = ? AND done = 0
        GROUP BY period
        """,
        (user_id,),
    )
    for period, cnt in cur.fetchall():
        if period in summary["active"]:
            summary["active"][period] = cnt

    # выполнено сегодня
    cur.execute(
        """
        SELECT COUNT(*) 
        FROM tasks 
        WHERE user_id = ? 
          AND done = 1 
          AND DATE(completed_at) = DATE('now', 'localtime')
        """,
        (user_id,),
    )
    row = cur.fetchone()
    summary["done_today"] = row[0] if row else 0

    conn.close()
    return summary


def get_stats(user_id: int) -> Dict[str, int]:
    """
    Общая статистика для истории.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM tasks WHERE user_id = ?", (user_id,))
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM tasks WHERE user_id = ? AND done = 1", (user_id,))
    done = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM tasks WHERE user_id = ? AND done = 0", (user_id,))
    active = cur.fetchone()[0]

    conn.close()
    return {"total": total, "done": done, "active": active}


def get_last_report_date(user_id: int) -> Optional[str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT last_report_date FROM daily_reports WHERE user_id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def update_last_report_date(user_id: int, date_str: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO daily_reports (user_id, last_report_date)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET last_report_date = excluded.last_report_date
        """,
        (user_id, date_str),
    )
    conn.commit()
    conn.close()

