import os
import sys
from web_server import app, socketio, setup_web_server
from main import start_ai_components

# Initialise AI background logic and shared state
# This starts the tracking threads and prepares the shared state dictionary.
print("[Server] Initialising AI Components...")
shared_state, calibrator, settings, session_reporter = start_ai_components()

# Setup the web server with the shared state to ensure API and WebSockets have data access
setup_web_server(shared_state, calibrator, settings, session_reporter)

if __name__ == "__main__":
    # This block is used for local development via 'python server.py'
    port = int(os.environ.get("PORT", 8080))
    print(f"[Server] Starting locally on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
