"""
calibration.py — Personal threshold calibrator

Records shared_state signals during "focused" and "distracted" sessions.
After both sessions, calculates personalized thresholds and saves them
to calibration.json for use by focus_score.py.

Signals recorded: ear, yaw, pitch, blink_rate_pm, bad_posture
"""

import time
import threading
import sqlite3
import json
import os
import numpy as np
from datetime import datetime

DB_PATH   = os.path.join(os.path.dirname(__file__), "focus_data.db")
JSON_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")

SIGNALS = ["ear", "yaw", "pitch", "blink_rate_pm", "bad_posture"]


def _init_calibration_table(db_path: str = DB_PATH):
    """Create the calibration_data table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_data (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            mode          TEXT    NOT NULL,
            ear           REAL,
            yaw           REAL,
            pitch         REAL,
            blink_rate_pm REAL,
            bad_posture   REAL
        )
    """)
    conn.commit()
    conn.close()


class Calibrator:
    """
    Records shared_state values every 500ms in either "focused" or
    "distracted" mode. Data is saved to SQLite. After both sessions,
    calculate_thresholds() derives personalized values and writes
    calibration.json.
    """

    def __init__(self, shared_state: dict):
        self.shared_state = shared_state
        self._stop_event = threading.Event()
        self._thread = None
        self._mode = None       # "focused" or "distracted"
        self.status = "Idle"    # read by dashboard for display

        _init_calibration_table()

    def start(self, mode: str):
        """Start recording in the given mode ('focused' or 'distracted')."""
        if self._thread and self._thread.is_alive():
            self.stop()

        self._mode = mode
        self._stop_event.clear()
        self.status = f"Recording: {mode}"
        self._thread = threading.Thread(
            target=self._record_loop, name="Calibrator", daemon=True
        )
        self._thread.start()
        print(f"[Calibrator] Started recording ({mode}).")

    def stop(self):
        """Stop the current recording session."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._mode = None
        self.status = "Idle"
        print("[Calibrator] Stopped.")

    def _record_loop(self):
        """Record signals every 500ms until stopped."""
        while not self._stop_event.is_set():
            state = self.shared_state
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            try:
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    """INSERT INTO calibration_data
                       (timestamp, mode, ear, yaw, pitch, blink_rate_pm, bad_posture)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ts,
                        self._mode,
                        state.get("ear", 0.0),
                        abs(state.get("yaw", 0.0)),
                        abs(state.get("pitch", 0.0)),
                        state.get("blink_rate_pm", 0.0),
                        1.0 if state.get("bad_posture", False) else 0.0,
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[Calibrator] DB error: {e}")

            time.sleep(0.5)

    def calculate_thresholds(self) -> dict:
        """
        Read both focused and distracted sessions from the database,
        compute average ± std for each signal, and derive personalized
        thresholds. Saves to calibration.json and returns the dict.
        """
        conn = sqlite3.connect(DB_PATH)
        thresholds = {}

        for mode in ["focused", "distracted"]:
            rows = conn.execute(
                "SELECT ear, yaw, pitch, blink_rate_pm, bad_posture "
                "FROM calibration_data WHERE mode = ?", (mode,)
            ).fetchall()

            if not rows:
                print(f"[Calibrator] No {mode} data found — "
                      f"please record a {mode} session first.")
                conn.close()
                return {}

            arr = np.array(rows, dtype=np.float64)
            stats = {}
            for i, sig in enumerate(SIGNALS):
                stats[sig] = {
                    "mean": round(float(np.mean(arr[:, i])), 4),
                    "std":  round(float(np.std(arr[:, i])), 4),
                }
            thresholds[mode] = stats

        conn.close()

        # Derive thresholds: midpoint between focused and distracted means
        personal = {}
        for sig in SIGNALS:
            f_mean = thresholds["focused"][sig]["mean"]
            d_mean = thresholds["distracted"][sig]["mean"]

            if sig in ("yaw", "pitch"):
                # Head angles: threshold = midpoint between focused and distracted
                personal[f"{sig}_threshold"] = round((f_mean + d_mean) / 2, 2)
            elif sig == "ear":
                personal["ear_focused_mean"] = round(f_mean, 4)
                personal["ear_distracted_mean"] = round(d_mean, 4)
            elif sig == "blink_rate_pm":
                personal["blink_low"] = round(max(1, f_mean - 2 * thresholds["focused"][sig]["std"]), 1)
                personal["blink_high"] = round(f_mean + 2 * thresholds["focused"][sig]["std"], 1)
            elif sig == "bad_posture":
                personal["posture_penalty"] = round(d_mean, 2)

        personal["raw_stats"] = thresholds

        # Save to JSON
        with open(JSON_PATH, "w") as f:
            json.dump(personal, f, indent=2)

        print(f"[Calibrator] Thresholds saved → {JSON_PATH}")
        print(f"[Calibrator] {json.dumps(personal, indent=2)}")

        return personal


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    state = {"ear": 0.30, "yaw": 5.0, "pitch": 3.0,
             "blink_rate_pm": 15.0, "bad_posture": False}

    cal = Calibrator(shared_state=state)
    cal.start("focused")
    time.sleep(3)
    cal.stop()

    state.update({"yaw": 30.0, "pitch": 25.0, "bad_posture": True})
    cal.start("distracted")
    time.sleep(3)
    cal.stop()

    result = cal.calculate_thresholds()
    print("Thresholds:", result)
