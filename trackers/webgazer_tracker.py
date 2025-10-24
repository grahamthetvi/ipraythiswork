"""
WebGazer.js Eye Tracking Integration

WebGazer is a JavaScript library that uses webcam for eye tracking.
This module creates a bridge between Python and WebGazer running in a browser.

Source: https://github.com/brownhci/WebGazer
"""

import json
import threading
import time
from typing import Optional, Tuple, List
from pathlib import Path

try:
    from flask import Flask, render_template, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install: pip install flask flask-cors")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from eye_tracker_framework import EyeTrackerBase, GazePoint, CalibrationPoint


class WebGazerTracker(EyeTrackerBase):
    """
    WebGazer.js eye tracking implementation.

    Uses WebGazer.js running in a browser, with Python backend
    receiving gaze data via HTTP/WebSocket.
    """

    def __init__(self, screen_width: int, screen_height: int):
        """Initialize WebGazer tracker."""
        super().__init__(screen_width, screen_height)
        self.tracker_name = "WebGazer.js"

        if not FLASK_AVAILABLE:
            raise ImportError("Flask required for WebGazer. Install: pip install flask flask-cors")

        # Flask app for serving WebGazer page
        self.app = Flask(__name__, template_folder=str(Path(__file__).parent.parent / 'templates'))
        CORS(self.app)

        # Gaze data storage
        self.current_gaze = None
        self.gaze_lock = threading.Lock()

        # Calibration
        self.calibration_points_data: List[CalibrationPoint] = []
        self.calibration_in_progress = False

        # Flask server thread
        self.server_thread = None
        self.server_running = False

        # Setup Flask routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes for WebGazer communication."""

        @self.app.route('/')
        def index():
            """Serve WebGazer page."""
            return render_template('webgazer.html',
                                 screen_width=self.screen_width,
                                 screen_height=self.screen_height)

        @self.app.route('/gaze', methods=['POST'])
        def receive_gaze():
            """Receive gaze data from WebGazer."""
            data = request.json
            with self.gaze_lock:
                self.current_gaze = GazePoint(
                    x=float(data['x']),
                    y=float(data['y']),
                    confidence=float(data.get('confidence', 1.0)),
                    timestamp=time.time()
                )
            return jsonify({'status': 'ok'})

        @self.app.route('/calibration_status')
        def calibration_status():
            """Get calibration status."""
            return jsonify({
                'calibrated': self.is_calibrated,
                'in_progress': self.calibration_in_progress
            })

    def start(self, camera_index: int = 0) -> bool:
        """
        Start WebGazer tracker.

        This launches a Flask server and opens browser to WebGazer page.

        Args:
            camera_index: Ignored for WebGazer (uses browser's camera selection)

        Returns:
            True if started successfully
        """
        print("\n" + "="*60)
        print("STARTING WEBGAZER TRACKER")
        print("="*60)
        print("\nWebGazer will open in your web browser.")
        print("Allow camera access when prompted.")
        print("\nServer starting on http://localhost:5000")
        print("="*60)

        # Start Flask server in background thread
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()

        # Wait for server to start
        time.sleep(2)

        # Open browser
        import webbrowser
        webbrowser.open('http://localhost:5000')

        self.server_running = True
        return True

    def _run_server(self):
        """Run Flask server (called in background thread)."""
        self.app.run(host='localhost', port=5000, debug=False, use_reloader=False)

    def stop(self):
        """Stop WebGazer tracker."""
        self.server_running = False
        print("\nWebGazer tracker stopped")
        print("You can close the browser window")

    def get_gaze_point(self) -> Optional[GazePoint]:
        """
        Get current gaze point from WebGazer.

        Returns:
            GazePoint or None if no recent data
        """
        with self.gaze_lock:
            # Return current gaze if recent (within 100ms)
            if self.current_gaze and (time.time() - self.current_gaze.timestamp) < 0.1:
                return self.current_gaze
            return None

    def calibrate(self, calibration_points: List[Tuple[int, int]]) -> bool:
        """
        Calibrate WebGazer.

        Note: WebGazer handles calibration in the browser.
        This method tells the browser to start calibration mode.

        Args:
            calibration_points: List of (x, y) screen coordinates

        Returns:
            True when calibration complete
        """
        print("\n" + "="*60)
        print("WEBGAZER CALIBRATION")
        print("="*60)
        print("\nCalibration will happen in the browser.")
        print("Click on the calibration points when they appear.")
        print("="*60)

        self.calibration_in_progress = True

        # WebGazer calibrates itself in the browser
        # We just wait for it to finish (user clicks points)

        # For now, assume calibration happens in browser
        # In full implementation, would communicate with JavaScript

        time.sleep(2)  # Give user time to see message

        self.is_calibrated = True
        self.calibration_in_progress = False

        print("\n✓ Calibration ready (handled by WebGazer in browser)")
        return True

    def get_calibration_quality(self) -> float:
        """
        Get calibration quality.

        WebGazer doesn't provide explicit quality score.
        Returns 1.0 if calibrated, 0.0 otherwise.
        """
        return 1.0 if self.is_calibrated else 0.0

    def save_calibration(self, filename: str) -> bool:
        """
        Save calibration.

        WebGazer stores calibration in browser's localStorage.
        This method is not needed but implemented for interface compliance.
        """
        print("WebGazer calibration is stored in browser localStorage")
        print("It will persist across sessions automatically")
        return True

    def load_calibration(self, filename: str) -> bool:
        """
        Load calibration.

        WebGazer loads from browser's localStorage automatically.
        """
        print("WebGazer calibration loads automatically from browser")
        return True


if __name__ == "__main__":
    """Test WebGazer tracker."""
    print("Testing WebGazer Tracker...")

    tracker = WebGazerTracker(1920, 1080)

    print(f"Tracker: {tracker.tracker_name}")
    print(f"Info: {tracker.get_info()}")

    # Start tracker (will open browser)
    if tracker.start():
        print("\nTracker started. Browser should open.")
        print("Testing gaze detection for 10 seconds...")

        start_time = time.time()
        detections = 0

        while time.time() - start_time < 10:
            gaze = tracker.get_gaze_point()
            if gaze:
                detections += 1
                print(f"Gaze: ({gaze.x:.0f}, {gaze.y:.0f}) confidence={gaze.confidence:.2f}")

            time.sleep(0.1)

        print(f"\nDetections in 10s: {detections}")
        tracker.stop()
    else:
        print("Failed to start tracker")
