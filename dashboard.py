import sys
import time
import math
import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QGridLayout, QFrame, QSizePolicy, QPushButton, QSlider, QDialog
)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRectF, QPropertyAnimation, pyqtProperty, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush, QRadialGradient, QLinearGradient

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

matplotlib.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "text.color": "#aaa",
    "axes.labelcolor": "#888",
    "xtick.color": "#888",
    "ytick.color": "#888",
    "axes.edgecolor": "none",
})

# ─────────────────────────────────────────────────────────────────────────────
# Custom Focus Dial using QPainter
# ─────────────────────────────────────────────────────────────────────────────
class FocusDial(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self._score = 0
        self._target_score = 0

    def setScore(self, score):
        self._target_score = score

    def update_animation(self):
        # Smooth interpolation
        diff = self._target_score - self._score
        if abs(diff) > 0.5:
            self._score += diff * 0.2
        else:
            self._score = self._target_score
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        center = QPoint(w // 2, h // 2)
        radius = min(w, h) // 2 - 20
        
        # Color based on score
        if self._score >= 75:
            color = QColor("#00e676")
        elif self._score >= 50:
            color = QColor("#ffd700")
        else:
            color = QColor("#ff4444")

        # Glow Ring
        glow_grad = QRadialGradient(center, radius + 15)
        glow_grad.setColorAt(0, QColor(0, 0, 0, 0))
        glow_grad.setColorAt(0.8, color.lighter(150))
        glow_grad.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(glow_grad))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, radius + 15, radius + 15)

        # Background Track
        pen_track = QPen(QColor(255, 255, 255, 10))
        pen_track.setWidth(12)
        pen_track.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_track)
        painter.drawArc(center.x() - radius, center.y() - radius, radius * 2, radius * 2, -45 * 16, 270 * 16)

        # Score Arc
        pen_arc = QPen(color)
        pen_arc.setWidth(12)
        pen_arc.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_arc)
        span_angle = int((self._score / 100.0) * 270 * 16)
        painter.drawArc(center.x() - radius, center.y() - radius, radius * 2, radius * 2, -45 * 16, span_angle)

        # Ticks
        for i in range(11):
            angle = -45 + i * 27
            rad = math.radians(angle)
            x1 = center.x() + (radius - 15) * math.cos(rad)
            y1 = center.y() - (radius - 15) * math.sin(rad)
            x2 = center.x() + (radius - 2) * math.cos(rad)
            y2 = center.y() - (radius - 2) * math.sin(rad)
            painter.setPen(QPen(QColor(255, 255, 255, 50), 2))
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Text
        painter.setPen(color)
        font = QFont("Arial", 42, QFont.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, str(int(self._score)))

# ─────────────────────────────────────────────────────────────────────────────
# Graph Widget
# ─────────────────────────────────────────────────────────────────────────────
class AnimatedGraph(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 3))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.setStyleSheet("background: rgba(255,255,255,0.02); border-radius: 12px;")

    def _style_axes(self):
        self.ax.set_ylim(0, 100)
        self.ax.tick_params(colors="#888")
        for spine in self.ax.spines.values():
            spine.set_color("none")

    def update_graph(self, history):
        try:
            self.ax.cla()
            self._style_axes()
            # Colored Zones
            self.ax.axhspan(75, 100, color="#00ff88", alpha=0.04)
            self.ax.axhspan(50, 75, color="#ffd700", alpha=0.04)
            self.ax.axhspan(0, 50, color="#ff4444", alpha=0.04)
            self.ax.axhline(75, color="#ffffff", alpha=0.3, linestyle="--")

            if not history:
                self.draw_idle()
                return

            max_points = 300
            self.ax.set_xlim(0, max_points)
            self.ax.set_xticks([])

            scores = [s for _, s in history]
            xs = [max_points - len(scores) + i for i in range(len(scores))]

            self.ax.plot(xs, scores, color="#00f0ff", lw=2)
            last_x, last_y = xs[-1], scores[-1]
            self.ax.plot([last_x], [last_y], marker="o", markersize=6, color="#00f0ff")

            self.draw_idle()
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# Webcam
# ─────────────────────────────────────────────────────────────────────────────
class WebcamFeed(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #0f0f20; border-radius: 12px;")
        self.blink_state = 0

    def update_frame(self, frame_bgr, face_detected):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # LIVE badge
        cv2.circle(rgb, (30, 30), 6, (255, 0, 0), -1)
        cv2.putText(rgb, "LIVE", (45, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        h, w, c = rgb.shape
        img = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(img).scaled(self.width(), self.height(), Qt.KeepAspectRatio))

        self.blink_state = (self.blink_state + 1) % 10
        if face_detected:
            # Pulse cyan
            border = "#00f0ff" if self.blink_state < 5 else "#004d55"
        else:
            border = "#ff4444" if self.blink_state < 5 else "#8b0000"
            
        self.setStyleSheet(f"background: #0f0f20; border-radius: 12px; border: 3px solid {border};")

# ─────────────────────────────────────────────────────────────────────────────
# Stat Card 
# ─────────────────────────────────────────────────────────────────────────────
class HoverStatCard(QFrame):
    def __init__(self, title, color):
        super().__init__()
        self._color = color
        self.setStyleSheet(f"background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-top: 3px solid {color}; border-radius: 8px;")
        layout = QVBoxLayout(self)
        
        self.val_lbl = QLabel("--")
        self.val_lbl.setAlignment(Qt.AlignCenter)
        self.val_lbl.setStyleSheet(f"color: {color}; font-size: 26px; font-weight: bold; background: none; border: none;")
        
        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet("color: #aaa; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; background: none; border: none;")
        
        layout.addWidget(self.val_lbl)
        layout.addWidget(title_lbl)

    def set_value(self, val):
        self.val_lbl.setText(str(val))

# ─────────────────────────────────────────────────────────────────────────────
# Start Screen Overlay
# ─────────────────────────────────────────────────────────────────────────────
class StartScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background: rgba(5, 8, 16, 0.85);")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        self.btn = QPushButton("▶ START SESSION")
        self.btn.setFixedSize(220, 60)
        self.btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00f0ff, stop:1 #0066ff);
                color: white; font-weight: bold; font-size: 16px; border-radius: 30px;
            }
            QPushButton:hover { background: #00ffff; }
        """)
        
        sub = QLabel("Click to begin monitoring")
        sub.setStyleSheet("color: #888; font-size: 14px; margin-top: 10px; background: none;")
        sub.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.btn, alignment=Qt.AlignCenter)
        layout.addWidget(sub, alignment=Qt.AlignCenter)

# ─────────────────────────────────────────────────────────────────────────────
# Main Dashboard Window
# ─────────────────────────────────────────────────────────────────────────────
class DashboardV2(QMainWindow):
    def __init__(self, shared_state, calibrator=None, settings=None, session_reporter=None):
        super().__init__()
        self.shared_state = shared_state
        self._calibrator = calibrator
        self._settings = settings
        self._session_reporter = session_reporter
        
        self.setWindowTitle("🧠 AI Focus Monitor - Pro")
        self.resize(1100, 750)
        self.setStyleSheet("background-color: #050810;")

        self.scanline_y = 0
        self.title_color_state = 0

        # Central Widget Layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Header
        header_layout = QHBoxLayout()
        self.title_lbl = QLabel("🧠 AI Focus Monitor")
        self.title_lbl.setFont(QFont("Arial", 20, QFont.Bold))
        
        self.thermal_lbl = QLabel("⚫ Thermal: Unknown")
        self.thermal_lbl.setStyleSheet("color: #888; font-size: 12px;")
        
        header_layout.addWidget(self.title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.thermal_lbl)
        main_layout.addLayout(header_layout)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("border-top: 1px solid rgba(0, 240, 255, 0.3);")
        main_layout.addWidget(line)

        # Content Grid
        content_layout = QHBoxLayout()
        
        # Left: Webcam + Dial
        left_layout = QVBoxLayout()
        self.webcam = WebcamFeed()
        self.dial = FocusDial()
        left_layout.addWidget(self.webcam, stretch=2)
        left_layout.addWidget(self.dial, stretch=1)
        
        # Right: Graph + Cards + Controls
        right_layout = QVBoxLayout()
        self.graph = AnimatedGraph()
        
        cards_layout = QHBoxLayout()
        self.card_score = HoverStatCard("Focus", "#00e676")
        self.card_frust = HoverStatCard("Frustration", "#ff4444")
        self.card_postr = HoverStatCard("Posture", "#ffd700")
        self.card_sessn = HoverStatCard("Session", "#00f0ff")
        cards_layout.addWidget(self.card_score)
        cards_layout.addWidget(self.card_frust)
        cards_layout.addWidget(self.card_postr)
        cards_layout.addWidget(self.card_sessn)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_end = QPushButton("⏹ End Session")
        self.btn_end.setStyleSheet("background: #ff4444; color: white; border-radius: 6px; padding: 10px; font-weight: bold;")
        self.btn_end.clicked.connect(self.end_session)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_end)

        right_layout.addWidget(self.graph, stretch=2)
        right_layout.addLayout(cards_layout, stretch=1)
        right_layout.addLayout(btn_layout)

        content_layout.addLayout(left_layout, stretch=1)
        content_layout.addLayout(right_layout, stretch=1)
        main_layout.addLayout(content_layout)

        # Start Screen Overlay
        self.start_overlay = StartScreen()
        self.start_overlay.setParent(central)
        self.start_overlay.resize(self.size())
        self.start_overlay.btn.clicked.connect(self.start_session)
        if self.shared_state.get("session_active"):
            self.start_overlay.hide()

        # Timers
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100) # Fast refresh for animations
        
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.animate_bg)
        self.anim_timer.start(50)

    def resizeEvent(self, event):
        self.start_overlay.resize(self.size())
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        # Background Grid
        painter.setPen(QPen(QColor(255, 255, 255, 10)))
        step = 40
        for x in range(0, self.width(), step):
            for y in range(0, self.height(), step):
                painter.drawPoint(x, y)
        
        # Scanline
        painter.setPen(QPen(QColor(0, 240, 255, 30), 2))
        painter.drawLine(0, self.scanline_y, self.width(), self.scanline_y)

    def animate_bg(self):
        # Scanline
        self.scanline_y += 3
        if self.scanline_y > self.height():
            self.scanline_y = 0
            
        # Title Color Cycle
        self.title_color_state += 0.05
        r = int(123 * (1 + math.sin(self.title_color_state)))
        g = int(94 * (1 + math.cos(self.title_color_state)))
        b = int(167 * (1 + math.sin(self.title_color_state)))
        self.title_lbl.setStyleSheet(f"color: rgb({r}, {max(100, g)}, 255);")
        
        # Tick the dial physics
        self.dial.update_animation()
        self.update()

    def start_session(self):
        self.shared_state["session_active"] = True
        self.shared_state["session_start_time"] = time.time()
        self.start_overlay.hide()

    def end_session(self):
        self.shared_state["session_active"] = False
        self.start_overlay.show()
        if self._session_reporter:
            # Force generate report
            pass

    def update_ui(self):
        if not self.shared_state.get("session_active", False):
            return

        s = self.shared_state
        score = s.get("focus_score", 0)
        self.dial.setScore(score)
        
        # Update WebCam
        frame = s.get("frame")
        if frame is not None:
            self.webcam.update_frame(frame, s.get("face_detected", False))

        # Update Thermal
        mode = s.get("thermal_mode", "normal")
        if mode == "eco":
            self.thermal_lbl.setText("🟡 Thermal: Eco (15fps)")
        elif mode == "rest":
            self.thermal_lbl.setText("🔴 Thermal: Cooling (Paused)")
        else:
            self.thermal_lbl.setText("🟢 Thermal: Normal")

        # Update Cards
        self.card_score.set_value(int(score))
        self.card_frust.set_value(s.get("frustration_count", 0))
        self.card_postr.set_value("BAD" if s.get("bad_posture") else "OK")
        
        if s.get("session_start_time"):
            el = int(time.time() - s.get("session_start_time"))
            m, sc = divmod(el, 60)
            self.card_sessn.set_value(f"{m:02d}:{sc:02d}")

        # Graph
        hist = s.get("score_history", [])
        if hist:
            self.graph.update_graph(hist)

def run_dashboard(shared_state, calibrator=None, settings=None, session_reporter=None):
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    window = DashboardV2(shared_state, calibrator, settings, session_reporter)
    window.show()
    app.exec_()
