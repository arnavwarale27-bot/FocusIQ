import os
import csv
import time
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

class SessionReport:
    def __init__(self, shared_state: dict):
        self.shared_state = shared_state
        self.history = []
        self._stop_event = threading.Event()
        self.start_time = datetime.now()
        
    def start(self):
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        
    def run(self):
        print("[SessionReport] Running (Background telemetry active).")
        while not self._stop_event.is_set():
            time.sleep(1.0)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Pull metrics seamlessly avoiding explicit SQL locks iteratively
            self.history.append({
                "timestamp": now,
                "focus_score": self.shared_state.get("focus_score", 100),
                "ear": self.shared_state.get("ear", 0.0),
                "blink_rate_pm": self.shared_state.get("blink_rate_pm", 0.0),
                "yaw": self.shared_state.get("yaw", 0.0),
                "pitch": self.shared_state.get("pitch", 0.0),
                "bad_posture": self.shared_state.get("bad_posture", False),
                "phone_detected": self.shared_state.get("phone_detected", False),
                "drowsy": self.shared_state.get("drowsy", False),
                "frustration_score": self.shared_state.get("frustration_score", 0.0)
            })
            
    def generate_reports(self) -> tuple[str, str]:
        """Generates the CSV and PDF report inside ~/ai_focus_monitor/reports/"""
        project_root = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(project_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(reports_dir, f"session_{ts_str}.csv")
        pdf_path = os.path.join(reports_dir, f"session_{ts_str}.pdf")
        graph_path = os.path.join(reports_dir, f"temp_graph_{ts_str}.png")
        
        # 1. Export CSV
        headers = [
            "timestamp", "focus_score", "ear", "blink_rate_pm", "yaw", "pitch",
            "bad_posture", "phone_detected", "drowsy", "frustration_score"
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)
                
        # 2. Calculate PDF Stats safely bounding math logic
        total_seconds = len(self.history)
        if total_seconds == 0:
            return csv_path, pdf_path

        scores = [r["focus_score"] for r in self.history]
        blinks = [r["blink_rate_pm"] for r in self.history]
        
        avg_score = sum(scores) / total_seconds
        peak_score = max(scores)
        lowest_score = min(scores)
        
        avg_blink = sum(blinks) / total_seconds
        duration_mins = total_seconds / 60.0
        total_blinks = int(avg_blink * duration_mins)
        
        focused_time = sum(1 for s in scores if s > 75)
        distracted_time = sum(1 for s in scores if s < 50)
        
        pct_focused = (focused_time / total_seconds) * 100
        pct_distracted = (distracted_time / total_seconds) * 100
        
        # Count explicit incidents (transitions False -> True mappings)
        phone_count = 0
        bad_posture_count = 0
        drowsy_count = 0
        
        prev_phone = False
        prev_posture = False
        prev_drowsy = False
        
        for r in self.history:
            if r["phone_detected"] and not prev_phone: phone_count += 1
            if r["bad_posture"] and not prev_posture: bad_posture_count += 1
            if r["drowsy"] and not prev_drowsy: drowsy_count += 1
            
            prev_phone = r["phone_detected"]
            prev_posture = r["bad_posture"]
            prev_drowsy = r["drowsy"]
            
        badges_earned = self.shared_state.get("badges", [])
        xp_earned = self.shared_state.get("xp", 0) 
        
        # 3. Render Matplotlib Graph matching explicit aesthetic
        plt.figure(figsize=(7, 3), facecolor="#ffffff")
        plt.plot(scores, color="#00d4ff", linewidth=2)
        plt.axhline(75, color="green", linestyle="--", alpha=0.5, label="Target Focus (>75)")
        plt.axhline(50, color="red", linestyle="--", alpha=0.5, label="Distracted (<50)")
        plt.title("Focus Score over Session")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Focus Score")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()
        
        # 4. Generate ReportLab PDF natively
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, height - 50, "AI Focus Monitor — Session Report")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Session Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(50, height - 100, f"Duration: {int(total_seconds//60)}m {total_seconds%60}s")
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 140, "Summary Stats:")
        
        c.setFont("Helvetica", 12)
        c.drawString(70, height - 170, f"Average Focus Score: {avg_score:.1f}/100")
        c.drawString(70, height - 190, f"Peak Focus Score: {peak_score:.0f}/100")
        c.drawString(70, height - 210, f"Lowest Focus Score: {lowest_score:.0f}/100")
        
        c.drawString(70, height - 240, f"Total Blinks: {total_blinks} blinks")
        c.drawString(70, height - 260, f"Average Blink Rate: {avg_blink:.1f} per min")
        
        c.drawString(70, height - 290, f"Time Spent Focused (>75): {pct_focused:.1f}%")
        c.drawString(70, height - 310, f"Time Spent Distracted (<50): {pct_distracted:.1f}%")
        
        phone_yn = "Yes" if phone_count > 0 else "No"
        c.drawString(70, height - 340, f"Phone Usage Detected: {phone_yn} ({phone_count} times)")
        c.drawString(70, height - 360, f"Bad Posture Incidents: {bad_posture_count}")
        c.drawString(70, height - 380, f"Drowsy Episodes: {drowsy_count}")
        
        badge_str = ", ".join(badges_earned) if badges_earned else "None"
        c.drawString(70, height - 410, f"Badges Earned: {badge_str}")
        c.drawString(70, height - 430, f"Total XP Tracked: {xp_earned}")
        
        # Embed Image smoothly natively handling exceptions implicitly
        try:
            c.drawImage(ImageReader(graph_path), 50, height - 700, width=500, height=220)
            if os.path.exists(graph_path):
                os.remove(graph_path) # Cleanup temp graph png
        except Exception as e:
            print(f"[SessionReport] Graph embed failed: {e}")
            
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, 30, "Generated by AI Focus Monitor")
        
        c.save()
        print(f"[SessionReport] Rendered gracefully matching {pdf_path}")

        # 5. Append to sessions.json index for Web Reports
        self._update_session_index(reports_dir, ts_str, {
            "date": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": f"{int(total_seconds//60)}m {total_seconds%60}s",
            "avg_focus": round(avg_score, 1),
            "peak_focus": int(peak_score),
            "xp": xp_earned,
            "badges": badges_earned,
            "pdf": os.path.basename(pdf_path),
            "csv": os.path.basename(csv_path)
        })

        return csv_path, pdf_path

    def _update_session_index(self, reports_dir, session_id, meta):
        import json
        index_path = os.path.join(reports_dir, "sessions.json")
        sessions = []
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    sessions = json.load(f)
            except:
                pass
        
        # Add new session to the top
        sessions.insert(0, meta)
        
        try:
            with open(index_path, "w") as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            print(f"[SessionReport] Failed to update sessions.json: {e}")

    def stop(self):
        self._stop_event.set()
