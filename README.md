# 🧠 AI Focus Monitor

AI Focus Monitor is a state-of-the-art cognitive optimization tool that leverages computer vision and real-time biometric telemetry to help you stay in the "Flow" state. It tracks iris movement, head pose, blink velocity, and posture to calculate a live focus score and provide instant feedback.

## 🚀 Features
- **Real-time Neural HUD**: High-fidelity dashboard with live telemetry and focus scoring.
- **Biometric Tracking**: IRIS-based focus detection, blink rate analysis, and postural alignment monitoring.
- **Frustration Detection**: Identifies micro-expressions and behavioral patterns associated with cognitive fatigue.
- **Neural Gamification**: Level up your cognitive profile, earn badges, and track XP through focused work sessions.
- **Automated Reporting**: Generates detailed PDF and CSV reports with focus trends and distraction logs.
- **Smart CPU Management**: Integrated Thermal Manager for energy-efficient background monitoring.

## 🛠 Tech Stack
- **AI Core**: MediaPipe, OpenCV, YOLOv8
- **Backend**: Python 3.12, Flask, Flask-SocketIO
- **Frontend**: HTML5, Vanilla CSS (Cyberpunk HUD style), JavaScript, Chart.js
- **Persistence**: SQLite, JSON

## 💻 Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arnavwarale27/ai-focus-monitor.git
   cd ai-focus-monitor
   ```

2. **Install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the monitor**:
   ```bash
   python main.py
   ```
   The monitor will open your camera and launch the Web HUD at `http://localhost:8080`.

## ☁️ Deployment

### Frontend (Vercel)
The project is configured for Vercel. 
1. Connect your Github repository to Vercel.
2. The `vercel.json` will handle the routing to `dashboard.html`.

### Backend (Railway/Render)
The backend is production-ready via Gunicorn.
1. Set the environment variable `PORT` to dictate the listening port.
2. Use the `Procfile` to deploy a high-performance `eventlet` worker.

> [!IMPORTANT]
> **Hardware Note**: Cloud environments do not have physical cameras. To use the deployed frontend with your local camera, run the backend on your laptop and set the `FOCUS_BACKEND_URL` in your browser's local storage to your local/tunneled address.

## 📸 Screenshots
*(Add your screenshots here — e.g., HUD, Analyse Panel, Reports)*

---
**Author**: Arnav Warale
**License**: MIT
