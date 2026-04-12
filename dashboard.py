"""
dashboard.py — Phase 5
PyQt5 dashboard with:
  - Left:   live webcam with landmark overlay
  - Centre: circular QDial showing focus score 0–100
  - Right:  real-time matplotlib graph (last 5 minutes)
  - Bottom: session stats bar + calibration panel
Refreshes every 100 ms using QTimer.
"""

import sys
import time
import threading
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")   # must be set before pyplot import; safe for Qt embedding

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QDial,
    QFrame, QSizePolicy, QPushButton, QMessageBox,
)
from PyQt5.QtCore  import Qt, QTimer
from PyQt5.QtGui   import QImage, QPixmap, QFont, QColor, QPalette

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Apply dark theme to embedded figures
matplotlib.rcParams.update({
    "figure.facecolor" : "#111",
    "axes.facecolor"   : "#1a1a2e",
    "text.color"       : "#aaa",
    "axes.labelcolor"  : "#888",
    "xtick.color"      : "#888",
    "ytick.color"      : "#888",
    "axes.edgecolor"   : "#333",
})


# ─────────────────────────────────────────────────────────────────────────────
# Webcam Widget — shows the OpenCV frame
# ─────────────────────────────────────────────────────────────────────────────
class WebcamLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background: #111; border: 2px solid #333; border-radius: 8px;")
        self.setText("Waiting for camera…")

    def update_frame(self, frame: np.ndarray):
        """Convert BGR OpenCV frame → QPixmap and display."""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img   = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix   = QPixmap.fromImage(img).scaled(
            self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pix)


# ─────────────────────────────────────────────────────────────────────────────
# Focus Dial Widget — coloured QDial + numeric label
# ─────────────────────────────────────────────────────────────────────────────
class FocusDial(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.title = QLabel("FOCUS SCORE")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(QFont("Arial", 12, QFont.Bold))
        self.title.setStyleSheet("color: #aaa;")

        self.dial = QDial()
        self.dial.setRange(0, 100)
        self.dial.setNotchesVisible(True)
        self.dial.setEnabled(False)              # display only
        self.dial.setFixedSize(200, 200)
        self.dial.setStyleSheet("""
            QDial {
                background: #1a1a2e;
                border: 3px solid #00d4ff;
                border-radius: 100px;
            }
        """)

        self.score_label = QLabel("--")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFont(QFont("Arial", 48, QFont.Bold))
        self.score_label.setStyleSheet("color: #00d4ff;")

        self.status_label = QLabel("Initialising…")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 14px;")

        layout.addWidget(self.title)
        layout.addWidget(self.dial, alignment=Qt.AlignCenter)
        layout.addWidget(self.score_label)
        layout.addWidget(self.status_label)

    def update_score(self, score: float):
        self.dial.setValue(int(score))
        self.score_label.setText(str(int(score)))

        # Colour-code by score bracket
        if score >= 75:
            colour = "#00ff88"
            status = "🟢 Focused"
        elif score >= 50:
            colour = "#ffd700"
            status = "🟡 Moderate"
        else:
            colour = "#ff4444"
            status = "🔴 Distracted"

        self.score_label.setStyleSheet(f"color: {colour};")
        self.status_label.setText(status)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Widget — matplotlib line chart embedded in Qt
# ─────────────────────────────────────────────────────────────────────────────
class ScoreGraph(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 3), facecolor="#111")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _style_axes(self):
        self.ax.set_facecolor("#1a1a2e")
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("Focus Score",  color="#888", fontsize=9)
        self.ax.tick_params(colors="#888")
        for spine in self.ax.spines.values():
            spine.set_color("#333")

    def update_graph(self, history: list):
        """
        history: list of (timestamp_str, score) from shared_state.
        Wrapped in try/except so an interrupt mid-draw never crashes the app.
        """
        try:
            self.ax.cla()
            self._style_axes()

            # Colored zones behind the line
            self.ax.axhspan(75, 100, color="#00ff88", alpha=0.05, lw=0)
            self.ax.axhspan(50, 75, color="#ffd700", alpha=0.05, lw=0)
            self.ax.axhspan(0, 50, color="#ff4444", alpha=0.05, lw=0)

            # Focus threshold dashed line
            self.ax.axhline(75, color="#ffffff", alpha=0.4, linestyle="--", linewidth=1.5)
            self.ax.text(5, 77, "Focus threshold", color="#ffffff", alpha=0.6, fontsize=8)

            # Enforce exactly 5 minute window (300 seconds)
            max_points = 300
            self.ax.set_xlim(0, max_points)
            self.ax.set_xticks([0, 60, 120, 180, 240, 300])
            self.ax.set_xticklabels(["5m ago", "4m ago", "3m ago", "2m ago", "1m ago", "Now"])

            if not history:
                self.draw_idle()   # draw_idle is safer than draw() in Qt
                return

            scores = [s for _, s in history]
            # Map points so the newest is always directly at 'Now' (x=300)
            xs = [max_points - len(scores) + i for i in range(len(scores))]

            # Thicker line with glowing effect
            self.ax.plot(xs, scores, color="#00d4ff", linewidth=6.0, alpha=0.2) # glow
            self.ax.plot(xs, scores, color="#00d4ff", linewidth=2.5)            # main line
            self.ax.fill_between(xs, scores, alpha=0.15, color="#00d4ff")

            # Current score dot and annotation
            last_x = xs[-1]
            last_y = scores[-1]
            self.ax.plot([last_x], [last_y], marker="o", markersize=6, color="#00d4ff")
            self.ax.text(last_x - 3, last_y + 4, f"{int(last_y)}", color="#00d4ff", 
                         fontsize=10, fontweight="bold", ha="right", va="bottom")

            self.ax.set_title("Focus Score — Last 5 min", color="#ccc", fontsize=10)
            self.draw_idle()
        except Exception:
            pass   # silently ignore mid-draw interrupts


# ─────────────────────────────────────────────────────────────────────────────
# Stats Bar — session summary at the bottom
# ─────────────────────────────────────────────────────────────────────────────
class StatsBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background: #1a1a2e; border-top: 1px solid #333; padding: 6px;")
        layout = QHBoxLayout(self)

        self._labels = {}
        for key in ["EAR", "Blink/min", "Yaw", "Pitch", "Posture", "Frustration", "Session"]:
            lbl = QLabel(f"{key}: --")
            lbl.setStyleSheet("color: #aaa; font-size: 11px; padding: 0 12px;")
            layout.addWidget(lbl)
            self._labels[key] = lbl

        self._start = time.time()

    def update_stats(self, state: dict):
        elapsed = int(time.time() - self._start)
        mm, ss  = divmod(elapsed, 60)
        self._labels["EAR"     ].setText(f"EAR: {state.get('ear', 0):.3f}")
        self._labels["Blink/min"].setText(f"Blink/min: {state.get('blink_rate_pm', 0):.1f}")
        self._labels["Yaw"     ].setText(f"Yaw: {state.get('yaw', 0):.1f}°")
        self._labels["Pitch"   ].setText(f"Pitch: {state.get('pitch', 0):.1f}°")
        posture = "⚠️ BAD" if state.get("bad_posture") else "✅ OK"
        self._labels["Posture" ].setText(f"Posture: {posture}")
        
        frust_score = state.get('frustration_score', 0.0)
        if frust_score >= 60:
            f_color = "#ff4444"
            f_icon = "🔴"
        elif frust_score >= 30:
            f_color = "#ffd700"
            f_icon = "🟡"
        else:
            f_color = "#00ff88"
            f_icon = "🟢"
        self._labels["Frustration"].setText(f"Frustration: {f_icon} {int(frust_score)}")
        self._labels["Frustration"].setStyleSheet(f"color: {f_color}; font-size: 11px; padding: 0 12px; font-weight: bold;")
        
        self._labels["Session" ].setText(f"Session: {mm:02d}:{ss:02d}")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Panel — buttons for recording + threshold calculation
# ─────────────────────────────────────────────────────────────────────────────
class CalibrationPanel(QWidget):
    def __init__(self, calibrator):
        super().__init__()
        self.calibrator = calibrator
        self.setStyleSheet("background: #1a1a2e; border-top: 1px solid #333; padding: 4px;")
        layout = QHBoxLayout(self)

        btn_style = """
            QPushButton {
                color: #eee; font-size: 11px; padding: 6px 14px;
                border: 1px solid #444; border-radius: 4px;
                background: #2a2a3e;
            }
            QPushButton:hover { background: #3a3a5e; }
            QPushButton:pressed { background: #4a4a6e; }
        """

        self.btn_focused = QPushButton("🟢 Start Focused Calibration")
        self.btn_focused.setStyleSheet(btn_style)
        self.btn_focused.clicked.connect(lambda: self._start("focused"))

        self.btn_distracted = QPushButton("🔴 Start Distracted Calibration")
        self.btn_distracted.setStyleSheet(btn_style)
        self.btn_distracted.clicked.connect(lambda: self._start("distracted"))

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setStyleSheet(btn_style)
        self.btn_stop.clicked.connect(self._stop)

        self.btn_calc = QPushButton("⚡ Calculate My Thresholds")
        self.btn_calc.setStyleSheet(btn_style)
        self.btn_calc.clicked.connect(self._calculate)

        self.status_lbl = QLabel("Calibration: Idle")
        self.status_lbl.setStyleSheet("color: #aaa; font-size: 11px; padding: 0 12px;")

        layout.addWidget(self.btn_focused)
        layout.addWidget(self.btn_distracted)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_calc)
        layout.addWidget(self.status_lbl)

    def _start(self, mode):
        self.calibrator.start(mode)
        self.status_lbl.setText(f"Calibration: 🔴 Recording ({mode})")

    def _stop(self):
        self.calibrator.stop()
        self.status_lbl.setText("Calibration: Idle")

    def _calculate(self):
        result = self.calibrator.calculate_thresholds()
        if result:
            msg = "Personalized thresholds saved!\n\n"
            for k, v in result.items():
                if k == "raw_stats":
                    continue
                msg += f"  {k}: {v}\n"
            QMessageBox.information(self, "Calibration Complete", msg)
            self.status_lbl.setText("Calibration: ✅ Thresholds saved")
        else:
            QMessageBox.warning(
                self, "Calibration Incomplete",
                "Need both focused AND distracted sessions.\n"
                "Record both before calculating thresholds."
            )

    def update_status(self):
        """Called by timer to keep status label in sync."""
        if self.calibrator.status != "Idle":
            self.status_lbl.setText(f"Calibration: 🔴 {self.calibrator.status}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Dashboard Window
# ─────────────────────────────────────────────────────────────────────────────
class Dashboard(QMainWindow):
    """
    Instantiate this, pass in the shared_state dict, then call .show().
    The QTimer handles all refreshes internally at 10 fps (100 ms).
    """

    def __init__(self, shared_state: dict, calibrator=None):
        super().__init__()
        self.shared_state   = shared_state
        self._graph_tick    = 0   # throttle graph to every 10th timer tick (1s)
        self.setWindowTitle("AI Focus Monitor Dashboard")
        self.setStyleSheet("background: #0d0d1a;")
        self.resize(1280, 720)

        # ── Central widget & main layout ──────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 0)

        # Header
        header = QLabel("🧠  AI Focus Monitor")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setStyleSheet("color: #00d4ff; padding: 4px 0 8px 0;")
        main_layout.addWidget(header)

        # Three-column row
        row = QHBoxLayout()

        self.webcam  = WebcamLabel()
        self.dial    = FocusDial()
        self.graph   = ScoreGraph()

        row.addWidget(self.webcam, stretch=4)
        row.addWidget(self.dial,   stretch=3)
        row.addWidget(self.graph,  stretch=5)

        main_layout.addLayout(row)

        # Stats bar
        self.stats = StatsBar()
        main_layout.addWidget(self.stats)

        # Calibration panel
        self.cal_panel = None
        if calibrator is not None:
            self.cal_panel = CalibrationPanel(calibrator)
            main_layout.addWidget(self.cal_panel)

        # ── Refresh timer ─────────────────────────────────────────────────
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)   # 100 ms = 10 fps

    def _refresh(self):
        """Called every 100 ms. Pulls data from shared_state and updates UI."""
        state = self.shared_state

        # Webcam frame — update every tick (10 fps)
        frame = state.get("frame")
        if frame is not None:
            self.webcam.update_frame(frame)

        # Focus dial
        score = state.get("focus_score", 0)
        self.dial.update_score(score)

        # Graph — only redraw once per second to keep CPU usage low
        self._graph_tick += 1
        if self._graph_tick >= 10:
            self._graph_tick = 0
            history = state.get("score_history", [])
            self.graph.update_graph(history)

        # Stats
        self.stats.update_stats(state)

        # Calibration status
        if self.cal_panel:
            self.cal_panel.update_status()


# ── Standalone test ──────────────────────────────────────────────────────────
def run_dashboard(shared_state: dict, calibrator=None):
    """Call this in the MAIN thread (PyQt5 requires the GUI on the main thread)."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = Dashboard(shared_state, calibrator=calibrator)
    win.show()
    app.exec_()


if __name__ == "__main__":
    import threading
    from face_tracker  import FaceTracker
    from ear_blink     import BlinkDetector
    from head_pose     import HeadPoseEstimator
    from focus_score   import FocusScoreCalculator

    state = {}
    threads = [
        threading.Thread(target=FaceTracker(state).run,              daemon=True),
        threading.Thread(target=BlinkDetector(state).run,            daemon=True),
        threading.Thread(target=HeadPoseEstimator(state).run,        daemon=True),
        threading.Thread(target=FocusScoreCalculator(state).run,     daemon=True),
    ]
    for t in threads:
        t.start()

    run_dashboard(state)   # blocks until window closed
