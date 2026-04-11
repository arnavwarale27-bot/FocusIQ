"""
database.py — Shared database utilities
Handles all SQLite reads, writes, and CSV export for the project.
Tables:
  - focus_log      : per-second focus score rows (created by focus_score.py)
  - frustration_log: timestamps of detected frustration events
"""

import sqlite3
import csv
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "focus_data.db")


# ─────────────────────────────────────────────────────────────────────────────
# Schema initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH):
    """Create all tables (safe to call multiple times — uses IF NOT EXISTS)."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS focus_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            raw_score REAL NOT NULL,
            avg_score REAL NOT NULL,
            ear       REAL,
            yaw       REAL,
            pitch     REAL,
            blink_rpm REAL
        );

        CREATE TABLE IF NOT EXISTS frustration_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            eyebrow_distance REAL,
            blink_velocity   REAL,
            event_tag        TEXT DEFAULT 'FRUSTRATION'
        );
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Initialised at {db_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Frustration events (Phase 8)
# ─────────────────────────────────────────────────────────────────────────────

def log_frustration(eyebrow_dist: float, blink_velocity: float,
                    db_path: str = DB_PATH):
    """Insert a frustration event row."""
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO frustration_log (timestamp, eyebrow_distance, blink_velocity)
           VALUES (?, ?, ?)""",
        (ts, eyebrow_dist, blink_velocity),
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# CSV export — session heatmap
# ─────────────────────────────────────────────────────────────────────────────

def export_session_csv(output_path: str = None, db_path: str = DB_PATH):
    """
    Export focus_log + frustration_log joined on timestamp to a CSV.
    The CSV becomes the "educator heatmap" that teachers can inspect.
    """
    if output_path is None:
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(os.path.dirname(__file__), f"session_{ts}.csv")

    conn    = sqlite3.connect(db_path)
    cursor  = conn.cursor()

    # Fetch focus data
    cursor.execute("""
        SELECT f.timestamp, f.avg_score, f.ear, f.yaw, f.pitch, f.blink_rpm,
               CASE WHEN fr.timestamp IS NOT NULL THEN 'FRUSTRATION' ELSE '' END AS event
        FROM focus_log f
        LEFT JOIN frustration_log fr ON f.timestamp = fr.timestamp
        ORDER BY f.timestamp
    """)
    rows    = cursor.fetchall()
    headers = ["timestamp", "avg_score", "ear", "yaw", "pitch", "blink_rpm", "event"]
    conn.close()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"[DB] Session exported → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Query helpers (used by dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def get_score_history(limit: int = 300, db_path: str = DB_PATH) -> list[tuple]:
    """Return the last `limit` (timestamp, avg_score) rows."""
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, avg_score FROM focus_log ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = list(reversed(cursor.fetchall()))
    conn.close()
    return rows


def get_session_summary(db_path: str = DB_PATH) -> dict:
    """Return aggregate stats for the current session."""
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            ROUND(AVG(avg_score), 1) AS mean_score,
            ROUND(MIN(avg_score), 1) AS min_score,
            ROUND(MAX(avg_score), 1) AS max_score,
            COUNT(*) AS total_seconds
        FROM focus_log
    """)
    row  = cursor.fetchone()
    conn.close()
    if row:
        return {
            "mean_score"   : row[0],
            "min_score"    : row[1],
            "max_score"    : row[2],
            "total_seconds": row[3],
        }
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    log_frustration(eyebrow_dist=42.3, blink_velocity=0.18)
    print("Session summary:", get_session_summary())
    export_session_csv()
