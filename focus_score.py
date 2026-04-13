"""
focus_score.py — Time-based attention score with EMA smoothing

Score starts at 100 and changes per second:
  INCREASE (focused):
    +2/s when face detected, not drowsy, head stable
    +0.5/s once score > 90 (hard to reach 100)

  DECREASE (distracted):
    -5/s  face not detected
    -3/s  yaw > 20° or pitch > 20°
    -5/s  drowsy
    -2/s  blink_rate_pm < 5 or > 40
    -1/s  bad posture

  Normal blinks → NO score change
  EMA smoothing: alpha = 0.1
  Clamped 0–100
"""

import time
import threading
import collections
import sqlite3
import json
import os
from datetime import datetime

# DB / calibration paths
DB_PATH   = os.path.join(os.path.dirname(__file__), "focus_data.db")
JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")

# Dynamic thresholds loaded directly from shared_state at compute time

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
    Runs every 1 second. Computes a delta to the current score
    based on attention signals, applies EMA smoothing, and writes
    the result to shared_state.

    shared_state keys READ:
        face_detected   (bool)
        yaw             (float)  degrees
        pitch           (float)  degrees
        drowsy          (bool)
        blink_rate_pm   (float)  blinks per minute
        bad_posture     (bool)
        ear             (float)

    shared_state keys WRITTEN:
        focus_score     (float)  0–100
        score_history   (list)   list of (timestamp_str, score), max 300
    """

    def __init__(self, shared_state: dict):
        self.shared_state = shared_state
        self._stop_event = threading.Event()
        self._history: collections.deque = collections.deque(maxlen=300)

        # EMA state
        self._alpha = 0.1
        self._current_score = 100.0
        self._start_time = time.time()

        _init_db(DB_PATH)

    def stop(self):
        self._stop_event.set()

    def _compute_delta(self) -> float:
        """
        Compute the per-second score change.
        Positive = focused (score goes up).
        Negative = distracted (score goes down).
        Penalties stack additively.
        """
        state = self.shared_state
        delta = 0.0

        face_detected = state.get("face_detected", True)
        drowsy        = state.get("drowsy", False)
        yaw           = abs(state.get("yaw", 0.0))
        pitch         = abs(state.get("pitch", 0.0))
        blink_rpm     = state.get("blink_rate_pm", 15.0)
        bad_posture   = state.get("bad_posture", False)
        uptime        = time.time() - self._start_time

        yaw_thresh   = state.get("yaw_threshold", 20.0)
        pitch_thresh = state.get("pitch_threshold", 20.0)
        blink_lo     = state.get("blink_rate_min", 5.0)
        blink_hi     = state.get("blink_rate_max", 40.0)

        # ── Penalties (distraction signals) ──────────────────────────
        if not face_detected:
            delta -= 5.0

        if yaw > yaw_thresh or pitch > pitch_thresh:
            delta -= 3.0

        if drowsy:
            delta -= 5.0

        # Only check blink rate after 60s warmup (deque needs time to fill)
        if uptime > 60:
            if blink_rpm < blink_lo or blink_rpm > blink_hi:
                delta -= 2.0

        if bad_posture:
            delta -= 1.0

        phone_detected = state.get("phone_detected", False)
        if phone_detected:
            delta -= 30.0
            print("[FocusScore] 📱 Phone usage detected — score penalty applied")

        # ── Reward (focused) ─────────────────────────────────────
        # Award positive points when no penalties were applied
        if delta == 0.0 and face_detected and not drowsy:
            if self._current_score >= 90:
                delta += 0.5    # hard to reach 100
            else:
                delta += 2.0

        return delta

    def run(self):
        """Main loop — call from a dedicated thread."""
        print("[FocusScore] Running.")

        while not self._stop_event.is_set():
            delta = self._compute_delta()

            # Apply delta to get raw target score
            raw_target = self._current_score + delta
            raw_target = max(0.0, min(100.0, raw_target))

            # EMA smoothing
            smoothed = self._alpha * raw_target + (1 - self._alpha) * self._current_score
            smoothed = max(0.0, min(100.0, smoothed))
            self._current_score = smoothed

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._history.append((ts, smoothed))

            # Write to shared state
            self.shared_state["focus_score"] = round(smoothed, 1)
            self.shared_state["score_history"] = list(self._history)

            # Persist to SQLite
            self._save_to_db(ts, smoothed)


            time.sleep(1.0)

        print("[FocusScore] Stopped.")

    def _save_to_db(self, ts: str, score: float):
        """Insert one row into the focus_log table."""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                """INSERT INTO focus_log
                   (timestamp, raw_score, avg_score, ear, yaw, pitch, blink_rpm)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    ts, score, score,
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
        "face_detected" : True,
        "yaw"           : 5.0,
        "pitch"         : 3.0,
        "drowsy"        : False,
        "blink_rate_pm" : 15.0,
        "bad_posture"   : False,
        "ear"           : 0.30,
    }
    calc = FocusScoreCalculator(shared_state=state)
    t = threading.Thread(target=calc.run, daemon=True)
    t.start()

    for i in range(15):
        time.sleep(1)
        if i == 5:
            state["yaw"] = 25.0
            state["drowsy"] = True
            print("  → Simulating distraction (yaw=25, drowsy)")
        if i == 10:
            state["yaw"] = 2.0
            state["drowsy"] = False
            print("  → Back to focused")
        print(f"  Score={state.get('focus_score')}")

    calc.stop()
