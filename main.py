"""
main.py — Phase 9 (Entry point)
Ties together ALL modules using Python threading.
Each module runs in its own thread and shares a single state dict.

Architecture:
┌──────────────────────────────────────────┐
│            shared_state (dict)           │
│  landmarks, frame, ear, yaw, pitch,      │
│  focus_score, blink_rate_pm, bad_posture │
└──────────────────────────────────────────┘
         │           │            │
   FaceTracker  BlinkDetector HeadPose
         │           │            │
               FocusScoreCalc
                     │
             FrustrationMapper
                     │
           20-20-20 Enforcer
                     │
            PostureDetector
                     │
                Dashboard  ← main thread (PyQt5 requirement)

Usage:
    python main.py
"""

import os
import sys
import threading
import time
import webbrowser

# ── macOS camera fix — MUST be set before cv2 is imported anywhere ────────────
# Without this, OpenCV cannot open the camera from a background thread on macOS.
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

# ── Import all modules ───────────────────────────────────────────────────────
from face_tracker       import FaceTracker
from ear_blink          import BlinkDetector
from head_pose          import HeadPoseEstimator
from focus_score        import FocusScoreCalculator
from posture            import PostureDetector
from enforcer           import TwentyTwentyTwentyEnforcer
from frustration_mapper import FrustrationMapper
from database           import init_db, export_session_csv
from calibration        import Calibrator
from settings           import Settings
from phone_detector     import PhoneDetector
from xp_system          import XPSystem
from notifier           import Notifier
from session_report     import SessionReport
from web_server         import run_web_server
from thermal_manager    import ThermalManager

# ── Optional MQTT IoT module (paho-mqtt) ─────────────────────────────────────
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

MQTT_BROKER = "localhost"   # Mosquitto broker running locally
MQTT_PORT   = 1883
MQTT_TOPIC  = "home/desk_lamp"


# ─────────────────────────────────────────────────────────────────────────────
# MQTT publisher (optional)
# ─────────────────────────────────────────────────────────────────────────────

def mqtt_publisher(shared_state: dict, stop_event: threading.Event):
    """
    Publishes desk-lamp control messages based on focus score.
    Score < 40 for 30 s → turn lamp ON (room may be dim / user fatigued).
    Runs in its own daemon thread.
    """
    if not MQTT_AVAILABLE:
        print("[MQTT] paho-mqtt not installed — skipping IoT publisher.")
        return

    # Use the new callback API to suppress DeprecationWarning
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()   # older paho-mqtt fallback
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except Exception as e:
        print(f"[MQTT] Cannot connect to broker at {MQTT_BROKER}:{MQTT_PORT} — {e}")
        return

    client.loop_start()
    low_focus_start = None
    lamp_on         = False

    while not stop_event.is_set():
        score = shared_state.get("focus_score", 100)

        if score < 40:
            if low_focus_start is None:
                low_focus_start = time.time()
            elif time.time() - low_focus_start > 30 and not lamp_on:
                client.publish(MQTT_TOPIC, "ON")
                print("[MQTT] 💡 Published → lamp ON (low focus detected)")
                lamp_on = True
        else:
            low_focus_start = None
            if lamp_on:
                client.publish(MQTT_TOPIC, "OFF")
                print("[MQTT] 💡 Published → lamp OFF (focus restored)")
                lamp_on = False

        time.sleep(5)

    client.loop_stop()
    client.disconnect()
    print("[MQTT] Publisher stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def start_ai_components():
    """Initializes all AI modules and shared state without launching the web server."""
    # Initialise the database
    init_db()
    app_settings = Settings()

    shared_state = {
        "landmarks": [], "frame": None, "face_detected": False, "ear": 0.0,
        "blink_count": 0, "blink_rate_pm": 0.0, "blink_velocity": 0.0, "drowsy": False,
        "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "looked_away": False, "focus_score": 100.0,
        "focus_score_raw": 100.0, "score_history": [], "neck_angle": 0.0,
        "bad_posture": False, "posture_alert_sent": False, "frustration_detected": False,
        "frustration_count": 0, "brow_distance": 0.0, "enforcer_active": False,
        "enforcer_countdown": 0, "session_active": False, "thermal_mode": "normal",
        "xp": 0, "level": 1, "badges": [], "xp_to_next_level": 100
    }
    shared_state.update(app_settings.get_all())

    face_tracker   = FaceTracker(shared_state,        camera_index=0, headless=True)
    blink_detector = BlinkDetector(shared_state,      fps=30)
    head_pose      = HeadPoseEstimator(shared_state,  fps=30)
    focus_calc     = FocusScoreCalculator(shared_state)
    frustration    = FrustrationMapper(shared_state,  fps=30)
    enforcer       = TwentyTwentyTwentyEnforcer(shared_state)
    posture_det    = PostureDetector(shared_state,    camera_index=0)
    calibrator     = Calibrator(shared_state)
    phone_det      = PhoneDetector(shared_state)
    xp_sys         = XPSystem(shared_state)
    notifier       = Notifier(shared_state)
    session_rep    = SessionReport(shared_state)
    thermal_mgr    = ThermalManager(shared_state)

    background_threads = [
        threading.Thread(target=blink_detector.run, name="BlinkDetector", daemon=True),
        threading.Thread(target=head_pose.run,       name="HeadPose",      daemon=True),
        threading.Thread(target=focus_calc.run,      name="FocusScore",    daemon=True),
        threading.Thread(target=frustration.run,     name="Frustration",   daemon=True),
        threading.Thread(target=enforcer.run,        name="Enforcer",      daemon=True),
        threading.Thread(target=xp_sys.run,          name="XPSystem",      daemon=True),
        threading.Thread(target=notifier.run,        name="Notifier",      daemon=True),
        threading.Thread(target=session_rep.run,     name="SessionRep",    daemon=True),
        threading.Thread(target=posture_det.run,     name="Posture",       daemon=True),
        threading.Thread(target=phone_det.run,       name="PhoneDetector", daemon=True),
        threading.Thread(target=face_tracker.run,    name="FaceTracker",   daemon=True),
        threading.Thread(target=thermal_mgr.run,     name="ThermalManager",daemon=True),
    ]

    for t in background_threads: t.start()

    return shared_state, calibrator, app_settings, session_rep

def main():
    print("=" * 60)
    print("  🧠  AI Focus Monitor — Starting up")
    print("=" * 60)

    shared_state, calibrator, app_settings, session_rep = start_ai_components()


    print("\n[Main] Launching Web HUD (main thread)…")
    url = "http://127.0.0.1:8080"
    print(f"      Dashboard active at {url}\n")

    # Auto-open browser
    from threading import Timer
    Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        run_web_server(shared_state, calibrator=calibrator, settings=app_settings, session_reporter=session_rep)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        # ── Graceful shutdown ─────────────────────────────────────────────
        print("\n[Main] Shutting down…")
        stop_event.set()
        face_tracker.stop()
        blink_detector.stop()
        head_pose.stop()
        focus_calc.stop()
        frustration.stop()
        enforcer.stop()
        posture_det.stop()
        phone_det.stop()
        xp_sys.stop()
        notifier.stop()
        session_rep.stop()
        thermal_mgr.stop()

        # Wait briefly for threads
        for t in background_threads:
            t.join(timeout=2)

        # Export session CSV
        try:
            path = export_session_csv()
            print(f"[Main] Session saved → {path}")
        except Exception as e:
            print(f"[Main] Could not export CSV: {e}")

        print("[Main] Goodbye! 👋")


if __name__ == "__main__":
    main()
