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
    QFrame, QSizePolicy, QPushButton, QMessageBox, QSlider, QProgressBar
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

    def update_frame(self, frame: np.ndarray, state: dict = None):
        """Convert BGR OpenCV frame → QPixmap and display."""
        frame_copy = frame.copy()
        
        if state and state.get("phone_detected", False):
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(frame_copy, "WARNING: PHONE DETECTED", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        rgb   = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
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
        for key in ["EAR", "Blink/min", "Yaw", "Pitch", "Posture", "Frustration", "Phone", "Session"]:
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
        
        phone_detected = state.get("phone_detected", False)
        method = state.get("phone_detection_method", "none")
        if phone_detected:
            if method == "both":
                text = "📱 Both signals"
                color = "#8b0000"
            elif method == "visual":
                text = "📱 YOLO: Screen detected"
                color = "#ff4444"
            elif method == "behavioral":
                text = "📱 Behavior: Head down"
                color = "#ff8800"
            else:
                text = "📱 Phone: DETECTED"
                color = "#ff4444"
                
            self._labels["Phone"].setText(text)
            self._labels["Phone"].setStyleSheet(f"color: {color}; font-size: 11px; padding: 0 12px; font-weight: bold;")
        else:
            self._labels["Phone"].setText("📱 Phone: Clear")
            self._labels["Phone"].setStyleSheet("color: #00ff88; font-size: 11px; padding: 0 12px;")
        
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
# Settings Panel — side UI for dynamic tweaks
# ─────────────────────────────────────────────────────────────────────────────
class SettingsPanel(QFrame):
    def __init__(self, settings, shared_state):
        super().__init__()
        self.settings = settings
        self.shared_state = shared_state
        self.setStyleSheet("background: #1a1a2e; border-left: 2px solid #333; padding: 10px;")
        self.setFixedWidth(300)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        
        title = QLabel("⚙️ Settings")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #00d4ff; padding-bottom: 10px; border: none;")
        layout.addWidget(title)
        
        self.sliders = {}
        
        configs = [
            ("ear_threshold", "EAR Threshold", 0.10, 0.40, 100),
            ("yaw_threshold", "Yaw Threshold", 5, 45, 1),
            ("pitch_threshold", "Pitch Threshold", 5, 45, 1),
            ("posture_angle", "Posture Angle", 5, 45, 1),
            ("drowsy_seconds", "Drowsy Timer (s)", 1.0, 10.0, 10),
            ("blink_rate_min", "Blink Rate Min", 1, 15, 1),
            ("blink_rate_max", "Blink Rate Max", 20, 60, 1),
        ]
        
        for key, name, min_val, max_val, mult in configs:
            lbl_layout = QHBoxLayout()
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("color: #ccc; border: none; font-size: 12px;")
            
            val_lbl = QLabel("--")
            val_lbl.setStyleSheet("color: #00d4ff; border: none; font-size: 12px; font-weight: bold;")
            
            lbl_layout.addWidget(name_lbl)
            lbl_layout.addStretch()
            lbl_layout.addWidget(val_lbl)
            
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * mult))
            slider.setMaximum(int(max_val * mult))
            slider.setStyleSheet("""
                QSlider::groove:horizontal { border: 1px solid #444; height: 4px; background: #222; }
                QSlider::handle:horizontal { background: #00d4ff; width: 14px; margin: -5px 0; border-radius: 7px; }
            """)
            
            layout.addLayout(lbl_layout)
            layout.addWidget(slider)
            
            self.sliders[key] = (slider, val_lbl, mult)
            slider.valueChanged.connect(lambda v, k=key: self._on_slider_changed(k, v))
            
        layout.addSpacing(20)
        
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("💾 Save")
        self.btn_save.clicked.connect(self._save)
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.clicked.connect(self._reset)
        
        btn_style = "QPushButton { background: #333; color: white; padding: 6px; border-radius: 4px; }"
        self.btn_save.setStyleSheet(btn_style)
        self.btn_reset.setStyleSheet(btn_style)
        
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_reset)
        layout.addLayout(btn_layout)
        
        self.update_ui_from_settings()

    def update_ui_from_settings(self):
        for key, (slider, val_lbl, mult) in self.sliders.items():
            val = self.settings.get(key)
            if val is not None:
                slider.blockSignals(True)
                slider.setValue(int(val * mult))
                slider.blockSignals(False)
                if mult == 100:
                    val_lbl.setText(f"{val:.2f}")
                elif mult == 10:
                    val_lbl.setText(f"{val:.1f}")
                else:
                    val_lbl.setText(f"{int(val)}")

    def _on_slider_changed(self, key, slider_val):
        _, val_lbl, mult = self.sliders[key]
        real_val = slider_val / float(mult)
        if mult == 100:
            val_lbl.setText(f"{real_val:.2f}")
        elif mult == 10:
            val_lbl.setText(f"{real_val:.1f}")
        else:
            val_lbl.setText(f"{int(real_val)}")
            
        self.settings.set(key, real_val)
        self.shared_state[key] = real_val

    def _save(self):
        self.settings.save()
        QMessageBox.information(self, "Saved", "Settings saved to disk!")

    def _reset(self):
        self.settings.reset_to_defaults()
        for k, v in self.settings.get_all().items():
            self.shared_state[k] = v
        self.update_ui_from_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Badges Panel — side UI for visualizing accomplishments
# ─────────────────────────────────────────────────────────────────────────────
class BadgesPanel(QFrame):
    def __init__(self, shared_state):
        super().__init__()
        self.shared_state = shared_state
        self.setStyleSheet("background: #1a1a2e; border-left: 2px solid #333; padding: 10px;")
        self.setFixedWidth(260)
        
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        
        title = QLabel("🏆 Earned Badges")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #ffd700; padding-bottom: 10px; border: none;")
        self.layout.addWidget(title)
        
        self.badges_container = QVBoxLayout()
        self.layout.addLayout(self.badges_container)
        self.layout.addStretch()

    def update_ui(self):
        for i in reversed(range(self.badges_container.count())): 
            widget = self.badges_container.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
                
        badges = self.shared_state.get("badges", [])
        if not badges:
            lbl = QLabel("No badges yet.\\nKeep focusing!")
            lbl.setStyleSheet("color: #888; border: none; font-style: italic;")
            lbl.setAlignment(Qt.AlignCenter)
            self.badges_container.addWidget(lbl)
        else:
            for badge in badges:
                lbl = QLabel(badge)
                lbl.setStyleSheet("color: #eee; background: #2a2a3e; padding: 8px; border-radius: 4px; border: 1px solid #444;")
                lbl.setFont(QFont("Arial", 12))
                self.badges_container.addWidget(lbl)


# ─────────────────────────────────────────────────────────────────────────────
# Main Dashboard Window
# ─────────────────────────────────────────────────────────────────────────────
class Dashboard(QMainWindow):
    """
    Instantiate this, pass in the shared_state dict, then call .show().
    The QTimer handles all refreshes internally at 10 fps (100 ms).
    """

    def __init__(self, shared_state: dict, calibrator=None, settings=None):
        super().__init__()
        self.shared_state   = shared_state
        self._graph_tick    = 0   # throttle graph to every 10th timer tick (1s)
        self.setWindowTitle("AI Focus Monitor Dashboard")
        self.setStyleSheet("background: #0d0d1a;")
        self.resize(1280, 720)

        # ── Root layout ──────────────────────────────────
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root_layout = QHBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Main content layout (left side)
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 0)

        # Header with Settings button
        header_layout = QHBoxLayout()
        header = QLabel("🧠  AI Focus Monitor")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setStyleSheet("color: #00d4ff; padding: 4px 0 8px 0;")
        header_layout.addWidget(header)
        header_layout.addStretch()
        
        self.btn_badges = QPushButton("🏆 Badges")
        self.btn_badges.setStyleSheet("QPushButton { color: #eee; font-size: 13px; font-weight:bold; padding: 6px 14px; background: #2a2a3e; border-radius: 4px; margin-right: 8px; }")
        self.btn_badges.clicked.connect(self._toggle_badges)
        header_layout.addWidget(self.btn_badges)
        
        self.btn_settings = QPushButton("⚙️ Settings")
        self.btn_settings.setStyleSheet("QPushButton { color: #eee; font-size: 13px; font-weight:bold; padding: 6px 14px; background: #2a2a3e; border-radius: 4px; }")
        self.btn_settings.clicked.connect(self._toggle_settings)
        header_layout.addWidget(self.btn_settings)
        
        main_layout.addLayout(header_layout)
        
        # XP Progress Bar Header Layer
        xp_layout = QHBoxLayout()
        self.level_lbl = QLabel("Level 1")
        self.level_lbl.setFont(QFont("Arial", 12, QFont.Bold))
        self.level_lbl.setStyleSheet("color: #00d4ff;")
        self.xp_bar = QProgressBar()
        self.xp_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #333; border-radius: 4px; text-align: center; color: #fff; background: #1a1a2e; }
            QProgressBar::chunk { background-color: #00d4ff; border-radius: 3px; }
        """)
        self.xp_bar.setMinimum(0)
        self.xp_bar.setMaximum(100)
        self.xp_val_lbl = QLabel("0 XP")
        self.xp_val_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        xp_layout.addWidget(self.level_lbl)
        xp_layout.addWidget(self.xp_bar)
        xp_layout.addWidget(self.xp_val_lbl)
        main_layout.addLayout(xp_layout)

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

        root_layout.addWidget(central, stretch=1)
        
        # Settings panel (right side)
        if settings is not None:
            self.settings_panel = SettingsPanel(settings, shared_state)
            self.settings_panel.hide()  # hidden initially
            root_layout.addWidget(self.settings_panel)
        else:
            self.settings_panel = None
            
        self.badges_panel = BadgesPanel(shared_state)
        self.badges_panel.hide()
        root_layout.addWidget(self.badges_panel)

        # Floating Notification Block (Absolute Overlapping rendering)
        self.notif_lbl = QLabel(self)
        self.notif_lbl.setStyleSheet("""
            background-color: rgba(26, 26, 46, 240);
            color: #ffd700; border: 2px solid #ffd700;
            border-radius: 8px; padding: 12px 24px;
            font-size: 16px; font-weight: bold;
        """)
        self.notif_lbl.hide()
        self.notif_timer = QTimer(self)
        self.notif_timer.setSingleShot(True)
        self.notif_timer.timeout.connect(self.notif_lbl.hide)

        # ── Refresh timer ─────────────────────────────────────────────────
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)   # 100 ms = 10 fps

    def _toggle_badges(self):
        if self.settings_panel and self.settings_panel.isVisible():
            self.settings_panel.hide()
        if self.badges_panel.isVisible():
            self.badges_panel.hide()
        else:
            self.badges_panel.update_ui()
            self.badges_panel.show()

    def _toggle_settings(self):
        if self.badges_panel and self.badges_panel.isVisible():
            self.badges_panel.hide()
        if self.settings_panel:
            if self.settings_panel.isVisible():
                self.settings_panel.hide()
            else:
                self.settings_panel.show()

    def _refresh(self):
        """Called every 100 ms. Pulls data from shared_state and updates UI."""
        state = self.shared_state

        # Webcam frame — update every tick (10 fps)
        frame = state.get("frame")
        if frame is not None:
            self.webcam.update_frame(frame, state=state)

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

        # Update XP logic values
        level = state.get("level", 1)
        xp = state.get("xp", 0)
        next_xp = state.get("xp_to_next_level", 100)
        self.level_lbl.setText(f"Level {level}")
        self.xp_bar.setMaximum(next_xp)
        self.xp_bar.setValue(xp)
        self.xp_bar.setFormat(f"{xp} / {next_xp} XP")
        self.xp_val_lbl.setText(f"{xp} XP")

        # Modal Badges popup trigger sequence
        trigger = state.get("new_badge_trigger", None)
        if trigger is not None:
            state["new_badge_trigger"] = None
            self.notif_lbl.setText(f"🎉 Badge Unlocked: {trigger}!")
            self.notif_lbl.adjustSize()
            x = self.width() - self.notif_lbl.width() - 40
            self.notif_lbl.move(x, 60)
            self.notif_lbl.show()
            self.notif_lbl.raise_()
            self.notif_timer.start(3000)


# ── Standalone test ──────────────────────────────────────────────────────────
def run_dashboard(shared_state: dict, calibrator=None, settings=None):
    """Call this in the MAIN thread (PyQt5 requires the GUI on the main thread)."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = Dashboard(shared_state, calibrator=calibrator, settings=settings)
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
