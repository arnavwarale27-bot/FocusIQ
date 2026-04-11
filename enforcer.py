"""
enforcer.py — Phase 6
20-20-20 Rule Enforcer:
  Every 20 minutes → dim the screen.
  Use MediaPipe gaze to detect if user looks away for 20 seconds.
  If yes → restore brightness.
  Shows a countdown overlay on the webcam window.
"""

import threading
import time
import platform
import subprocess
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Screen brightness helpers (cross-platform)
# ─────────────────────────────────────────────────────────────────────────────

def _dim_screen():
    """Reduce screen brightness to ~30%."""
    os_name = platform.system()
    if os_name == "Darwin":                       # macOS
        # brightness 0.3 via osascript (built-in, no extra install)
        subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to '
             'set value of slider 1 of group 1 of window "Displays" '
             'of application process "System Preferences" to 0.3'],
            capture_output=True,
        )
    elif os_name == "Linux":
        subprocess.run(["xrandr", "--output", "LVDS-1", "--brightness", "0.3"],
                       capture_output=True)
    elif os_name == "Windows":
        # Use WMI / ctypes to set brightness
        import ctypes
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            ctypes.windll.gdi32.SetDeviceGammaRamp(
                hdc,
                (ctypes.c_word * 256 * 3)(
                    *([int(i * 0.3) for i in range(256)] * 3)
                )
            )
        except Exception:
            pass
    print("[Enforcer] 🌑 Screen dimmed.")


def _restore_screen():
    """Restore screen to full brightness."""
    os_name = platform.system()
    if os_name == "Darwin":
        subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to '
             'set value of slider 1 of group 1 of window "Displays" '
             'of application process "System Preferences" to 1.0'],
            capture_output=True,
        )
    elif os_name == "Linux":
        subprocess.run(["xrandr", "--output", "LVDS-1", "--brightness", "1.0"],
                       capture_output=True)
    elif os_name == "Windows":
        import ctypes
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            ctypes.windll.gdi32.SetDeviceGammaRamp(
                hdc,
                (ctypes.c_word * 256 * 3)(
                    *([int(i) for i in range(256)] * 3)
                )
            )
        except Exception:
            pass
    print("[Enforcer] ☀️  Screen restored.")


# ─────────────────────────────────────────────────────────────────────────────
# Countdown Overlay helper
# ─────────────────────────────────────────────────────────────────────────────

def _draw_countdown(frame: np.ndarray, seconds_left: int) -> np.ndarray:
    """
    Overlay a semi-transparent countdown banner on frame.
    Returns a copy with the overlay applied.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    # Dark rectangle across the bottom third
    cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    msg = f"20-20-20 Break — Look away for {seconds_left}s"
    cv2.putText(out, msg, (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Enforcer class
# ─────────────────────────────────────────────────────────────────────────────

class TwentyTwentyTwentyEnforcer:
    """
    Runs a background timer. Every 20 minutes:
      1. Dims the screen.
      2. Watches shared_state["looked_away"] for 20 consecutive seconds.
      3. When gaze is away for 20 s → restores brightness.
      4. Draws countdown on the frame in shared_state.

    shared_state keys read:
        looked_away (bool)
        frame       (ndarray)
    shared_state keys written:
        enforcer_active     (bool)
        enforcer_countdown  (int) seconds remaining
    """

    WORK_INTERVAL  = 20 * 60   # 20 minutes in seconds (set to 60 for quick test)
    GAZE_AWAY_SECS = 20        # seconds user must look away

    def __init__(self, shared_state: dict):
        self.shared_state   = shared_state
        self._stop_event    = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        print("[Enforcer] 20-20-20 enforcer running. "
              f"First break in {self.WORK_INTERVAL // 60} min.")
        while not self._stop_event.is_set():
            # Wait for the work interval
            self._stop_event.wait(timeout=self.WORK_INTERVAL)
            if self._stop_event.is_set():
                break
            self._do_break()

        print("[Enforcer] Stopped.")

    def _do_break(self):
        print("[Enforcer] ⏰ 20-minute mark — starting break.")
        _dim_screen()
        self.shared_state["enforcer_active"] = True

        gaze_away_start : float | None = None
        gaze_away_total = 0.0

        deadline = time.time() + 60  # give 60 s max for the break

        while time.time() < deadline and not self._stop_event.is_set():
            remaining = max(0, self.GAZE_AWAY_SECS - int(gaze_away_total))
            self.shared_state["enforcer_countdown"] = remaining

            # Annotate frame
            frame = self.shared_state.get("frame")
            if frame is not None:
                self.shared_state["frame"] = _draw_countdown(frame, remaining)

            if self.shared_state.get("looked_away", False):
                if gaze_away_start is None:
                    gaze_away_start = time.time()
                gaze_away_total = time.time() - gaze_away_start
            else:
                gaze_away_start = None
                gaze_away_total = 0.0

            if gaze_away_total >= self.GAZE_AWAY_SECS:
                print("[Enforcer] ✅ Gaze-away condition met — restoring screen.")
                break

            time.sleep(0.5)

        _restore_screen()
        self.shared_state["enforcer_active"]    = False
        self.shared_state["enforcer_countdown"] = 0
        print("[Enforcer] Break complete.")


# ── Standalone test (break triggers after 10 s instead of 20 min) ────────────
if __name__ == "__main__":
    TwentyTwentyTwentyEnforcer.WORK_INTERVAL = 10
    state = {"looked_away": False, "frame": None}
    enf   = TwentyTwentyTwentyEnforcer(shared_state=state)

    def fake_gaze():
        time.sleep(15)
        print("[Test] Simulating gaze away…")
        state["looked_away"] = True

    threading.Thread(target=fake_gaze, daemon=True).start()
    enf.run()
