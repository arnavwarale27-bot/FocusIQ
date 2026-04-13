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
from dashboard          import run_dashboard   # must run on main thread

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

def main():
    print("=" * 60)
    print("  🧠  AI Focus Monitor — Starting up")
    print("=" * 60)

    # Initialise the database
    init_db()

    # Load centralized settings
    app_settings = Settings()

    # ── Shared state dictionary ───────────────────────────────────────────
    # All threads read from and write to this single dict.
    # Python's GIL makes simple dict reads/writes effectively atomic for
    # most cases; for production use threading.Lock() around updates.
    shared_state: dict = {
        "landmarks"         : [],
        "frame"             : None,
        "face_detected"     : False,
        "ear"               : 0.0,
        "blink_count"       : 0,
        "blink_rate_pm"     : 0.0,
        "blink_velocity"    : 0.0,
        "drowsy"            : False,
        "yaw"               : 0.0,
        "pitch"             : 0.0,
        "roll"              : 0.0,
        "looked_away"       : False,
        "focus_score"       : 100.0,
        "focus_score_raw"   : 100.0,
        "score_history"     : [],
        "neck_angle"        : 0.0,
        "bad_posture"       : False,
        "posture_alert_sent": False,
        "frustration_detected": False,
        "frustration_count" : 0,
        "brow_distance"     : 0.0,
        "enforcer_active"   : False,
        "enforcer_countdown": 0,
    }
    
    # Pre-populate shared_state with initial settings
    shared_state.update(app_settings.get_all())

    # ── Instantiate modules ───────────────────────────────────────────────
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

    stop_event = threading.Event()

    # ── Build thread list ─────────────────────────────────────────────────
    background_threads = [
        threading.Thread(target=blink_detector.run, name="BlinkDetector", daemon=True),
        threading.Thread(target=head_pose.run,       name="HeadPose",      daemon=True),
        threading.Thread(target=focus_calc.run,      name="FocusScore",    daemon=True),
        threading.Thread(target=frustration.run,     name="Frustration",   daemon=True),
        threading.Thread(target=enforcer.run,        name="Enforcer",      daemon=True),
        threading.Thread(target=xp_sys.run,          name="XPSystem",      daemon=True),
        threading.Thread(target=notifier.run,        name="Notifier",      daemon=True),
        # Posture uses its own camera capture loop
        threading.Thread(target=posture_det.run,     name="Posture",       daemon=True),
        # MQTT publisher
        threading.Thread(
            target=mqtt_publisher,
            args=(shared_state, stop_event),
            name="MQTT",
            daemon=True,
        ),
        # Extractor running YOLO inference
        threading.Thread(target=phone_det.run,       name="PhoneDetector", daemon=True),
        # FaceTracker — runs its own OpenCV window AND feeds frames to dashboard
        threading.Thread(target=face_tracker.run,    name="FaceTracker",   daemon=True),
    ]

    print(f"\n[Main] Starting {len(background_threads)} background threads…")
    for t in background_threads:
        t.start()
        print(f"  ✅ {t.name} started.")

    print("\n[Main] Launching dashboard (main thread)…")
    print("      Close the dashboard window to exit.\n")

    try:
        # PyQt5 MUST run on the main thread
        run_dashboard(shared_state, calibrator=calibrator, settings=app_settings)
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
