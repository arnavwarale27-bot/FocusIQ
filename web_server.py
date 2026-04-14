import cv2
import os
import threading
import logging
import time
import base64
import numpy as np
from flask import Flask, Response, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, emit

# ── GLOBAL INSTANCES ────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='.')
app.config['SECRET_KEY'] = 'focus_monitor_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# These will be populated by setup_web_server()
_shared_state = {}
_calibrator = None
_settings = None
_session_reporter = None

# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_file('dashboard.html')

@app.route('/reports')
def reports_page():
    return send_file('reports.html')

@app.route('/reports/<path:filename>')
def serve_report(filename):
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    return send_from_directory(reports_dir, filename)

@app.route('/api/sessions')
def get_sessions():
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    index_path = os.path.join(reports_dir, "sessions.json")
    if os.path.exists(index_path):
        return send_file(index_path)
    return jsonify([])

@app.route('/api/session/latest')
def get_latest_session():
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    index_path = os.path.join(reports_dir, "sessions.json")
    if os.path.exists(index_path):
        import json
        with open(index_path, "r") as f:
            sessions = json.load(f)
            if sessions:
                return jsonify(sessions[0])
    return jsonify(None)

@app.route('/api/session/start', methods=['POST'])
def start_session():
    _shared_state["session_active"] = True
    _shared_state["session_start_time"] = time.time()
    print("[WebServer] ▶ Session started via API")
    return jsonify({"ok": True})

@app.route('/api/session/end', methods=['POST'])
def end_session_api():
    _shared_state["session_active"] = False
    if _session_reporter:
        try:
            csv_path, pdf_path = _session_reporter.generate_reports()
            print(f"[WebServer] ⏹ Session ended via API. Reports generated.")
            return jsonify({
                "ok": True, 
                "pdf": os.path.basename(pdf_path), 
                "csv": os.path.basename(csv_path)
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": False, "error": "No session reporter active"}), 400

@app.route('/api/settings', methods=['GET'])
def get_settings_call():
    if _settings:
        return jsonify(_settings.get_all())
    return jsonify({}), 400

@app.route('/api/settings/save', methods=['POST'])
def update_settings_call():
    from flask import request
    if _settings:
        data = request.get_json(force=True)
        for k, v in data.items():
            try:
                val = float(v)
                _settings.set(k, val)
                _shared_state[k] = val
            except:
                _settings.set(k, v)
                _shared_state[k] = v
        _settings.save()
        return jsonify({"ok": True, "settings": _settings.get_all()})
    return jsonify({"ok": False, "error": "No settings module"}), 400

@app.route('/api/calibration/start/<mode>', methods=['POST'])
def cal_start(mode):
    if _calibrator:
        if mode == "focused": _calibrator.start("focused")
        else: _calibrator.start("distracted")
        return jsonify({"ok": True, "status": _calibrator.status})
    return jsonify({"ok": False, "error": "No calibrator"}), 400

@app.route('/api/calibration/stop', methods=['POST'])
def cal_stop():
    if _calibrator:
        _calibrator.stop()
        return jsonify({"ok": True, "status": _calibrator.status})
    return jsonify({"ok": False, "error": "No calibrator"}), 400

@app.route('/api/calibration/calculate', methods=['POST'])
def cal_calculate():
    if _calibrator:
        results = _calibrator.calculate_thresholds()
        return jsonify({"ok": True, "results": results})
    return jsonify({"ok": False, "error": "No calibrator"}), 400

@app.route('/api/xp')
def get_xp_data():
    return jsonify({
        "xp": _shared_state.get("xp", 0),
        "level": _shared_state.get("level", 1),
        "badges": _shared_state.get("badges", []),
        "xp_to_next": _shared_state.get("xp_to_next_level", 100)
    })

# ── HELPERS ─────────────────────────────────────────────────────────────────
def _safe_convert(v):
    if isinstance(v, (np.bool_, bool)): return bool(v)
    elif isinstance(v, (np.integer, int)): return int(v)
    elif isinstance(v, (np.floating, float)): return float(v)
    elif isinstance(v, list): return [_safe_convert(x) for x in v]
    elif isinstance(v, dict): return {k2: _safe_convert(v2) for k2, v2 in v.items()}
    return v

def state_emitter():
    while True:
        state_copy = {}
        for k, v in _shared_state.items():
            if k not in ['frame', 'landmarks', 'phone_detected', 'phone_detected_this_session']:
                try: state_copy[k] = _safe_convert(v)
                except: state_copy[k] = str(v)
        
        if _shared_state.get("session_active") and _shared_state.get("session_start_time"):
            state_copy['session_elapsed'] = int(time.time() - _shared_state.get("session_start_time"))
        else:
            state_copy['session_elapsed'] = 0

        if _calibrator:
            state_copy['calibrator_status'] = _calibrator.status

        socketio.emit('state_update', state_copy)
        socketio.sleep(0.1)

def video_emitter():
    while True:
        frame = _shared_state.get("frame")
        if frame is not None:
            try:
                small = cv2.resize(frame, (640, 480))
                ret, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ret:
                    b64_frame = base64.b64encode(buffer).decode('utf-8')
                    socketio.emit('video_frame', b64_frame)
            except Exception as e:
                print(f"[WebServer] Video emission error: {e}")
        socketio.sleep(0.1)

@socketio.on('connect')
def handle_connect():
    print("[WebServer] Client connected via WebSocket")

# ── SETUP & RUN ─────────────────────────────────────────────────────────────
def setup_web_server(shared_state, calibrator=None, settings=None, session_reporter=None):
    global _shared_state, _calibrator, _settings, _session_reporter
    _shared_state = shared_state
    _calibrator = calibrator
    _settings = settings
    _session_reporter = session_reporter

    # Start background tasks
    socketio.start_background_task(state_emitter)
    socketio.start_background_task(video_emitter)

    # Disable Flask request logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

def run_web_server(shared_state, calibrator=None, settings=None, session_reporter=None):
    """Legacy entry point for local execution."""
    setup_web_server(shared_state, calibrator, settings, session_reporter)
    port = int(os.environ.get("PORT", 8080))
    print(f"\n[WebServer] 🌐 Web HUD running at http://0.0.0.0:{port}")
    socketio.run(app, host='0.0.0.0', port=port, log_output=False)
