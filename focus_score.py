"""
focus_score.py — Phase 4
Calculates a 0–100 focus score every second:
    Score = 100 - (gaze_away_penalty + blink_penalty + head_turn_penalty)
Smoothed using a 5-second rolling average, saved to SQLite.
"""

import time
import threading
import collections
import sqlite3
import os
from datetime import datetime

# ── Penalty configuration ────────────────────────────────────────────────────
GAZE_PENALTY_PER_SEC   = 15   # deducted each second eyes/head are off screen
BLINK_PENALTY_PER_MIN  = 2    # deducted per extra blink above normal 20 bpm
NORMAL_BLINK_RATE      = 20   # blinks/min baseline
HEAD_TURN_PENALTY      = 10   # deducted each second head is turned away

# Rolling average window (seconds)
ROLLING_WINDOW = 5

# DB path — sits in the same folder as this script
DB_PATH = os.path.join(os.path.dirname(__file__), "focus_data.db")


def _init_db(db_path: str):
    """Create the SQLite table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS focus_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            raw_score REAL    NOT NULL,
            avg_score REAL    NOT NULL,
            ear       REAL,
            yaw       REAL,
            pitch     REAL,
            blink_rpm REAL
        )
    """)
    conn.commit()
    conn.close()


class FocusScoreCalculator:
    """
    Runs every 1 second, reads sensor data from shared_state,
    computes the focus score, and writes back.

    shared_state keys read:
        is_drowsy     (bool)
        looked_away   (bool)
        blink_rate_pm (float)
        ear           (float)
        yaw           (float)
        pitch         (float)

    shared_state keys written:
        focus_score     (float)  — current smooth score 0–100
        focus_score_raw (float)  — unsmoothed score
        score_history   (list)   — list of (timestamp, score) for graph
    """

    def __init__(self, shared_state: dict):
        self.shared_state = shared_state
        self._stop_event  = threading.Event()
        # Deque of raw scores for rolling average (max = ROLLING_WINDOW entries)
        self._window: collections.deque = collections.deque(maxlen=ROLLING_WINDOW)
        # History for the dashboard graph (last 5 minutes × 1 sample/sec = 300)
        self._history: collections.deque = collections.deque(maxlen=300)

        _init_db(DB_PATH)

    def stop(self):
        self._stop_event.set()

    def _compute_raw_score(self) -> float:
        """Apply penalties to derive raw score."""
        score = 100.0

        # 1. Gaze / drowsiness penalty
        #    If EAR is low (eyes closed) or head is turned away → penalise
        if self.shared_state.get("is_drowsy", False):
            score -= GAZE_PENALTY_PER_SEC
        if self.shared_state.get("looked_away", False):
            score -= HEAD_TURN_PENALTY

        # 2. Blink penalty
        #    Each blink/min above normal costs 2 points
        blink_rpm = self.shared_state.get("blink_rate_pm", NORMAL_BLINK_RATE)
        excess_blinks = max(0, blink_rpm - NORMAL_BLINK_RATE)
        score -= excess_blinks * BLINK_PENALTY_PER_MIN

        return max(0.0, min(100.0, score))

    def run(self):
        """Main loop — call from a dedicated thread."""
        print("[FocusScore] Running.")

        while not self._stop_event.is_set():
            raw = self._compute_raw_score()
            self._window.append(raw)

            # 5-second rolling average
            smooth = sum(self._window) / len(self._window)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._history.append((ts, smooth))

            # Write to shared state
            self.shared_state.update({
                "focus_score"    : round(smooth, 1),
                "focus_score_raw": round(raw, 1),
                "score_history"  : list(self._history),
            })

            # Persist to SQLite
            self._save_to_db(ts, raw, smooth)

            time.sleep(1.0)

        print("[FocusScore] Stopped.")

    def _save_to_db(self, ts: str, raw: float, avg: float):
        """Insert one row into the focus_log table."""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                """INSERT INTO focus_log
                   (timestamp, raw_score, avg_score, ear, yaw, pitch, blink_rpm)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    ts, raw, avg,
                    self.shared_state.get("ear",           0.0),
                    self.shared_state.get("yaw",           0.0),
                    self.shared_state.get("pitch",         0.0),
                    self.shared_state.get("blink_rate_pm", 0.0),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[FocusScore] DB error: {e}")


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    state = {
        "is_drowsy"    : False,
        "looked_away"  : False,
        "blink_rate_pm": 18,
        "ear"          : 0.32,
        "yaw"          : 5.0,
        "pitch"        : 3.0,
    }
    calc = FocusScoreCalculator(shared_state=state)
    t = threading.Thread(target=calc.run, daemon=True)
    t.start()

    for _ in range(10):
        time.sleep(1)
        print(f"Focus: {state.get('focus_score')}  Raw: {state.get('focus_score_raw')}")

    calc.stop()
