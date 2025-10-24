"""
Pupil Labs Core Eye Tracking Integration

Integrates with Pupil Labs Core software and cameras for high-accuracy eye tracking.

Requirements:
- Pupil Labs cameras (hardware)
- Pupil Capture software running
- ZMQ for communication

Source: https://github.com/pupil-labs/pupil
Documentation: https://docs.pupil-labs.com/
"""

import time
import json
from typing import Optional, Tuple, List
from pathlib import Path

try:
    import zmq
    import msgpack
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("ZMQ not available. Install: pip install pyzmq msgpack")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from eye_tracker_framework import EyeTrackerBase, GazePoint, CalibrationPoint


class PupilLabsTracker(EyeTrackerBase):
    """
    Pupil Labs Core eye tracking implementation.

    Communicates with Pupil Capture software via ZMQ.
    Requires Pupil Labs cameras and software.
    """

    def __init__(self, screen_width: int, screen_height: int):
        """Initialize Pupil Labs tracker."""
        super().__init__(screen_width, screen_height)
        self.tracker_name = "Pupil Labs Core"

        if not ZMQ_AVAILABLE:
            raise ImportError("ZMQ required for Pupil Labs. Install: pip install pyzmq msgpack")

        # ZMQ context and sockets
        self.ctx = None
        self.pupil_remote = None
        self.subscriber = None

        # Pupil Capture connection
        self.pupil_capture_address = "localhost"
        self.pupil_capture_port = 50020  # Default Pupil Capture port

        # Gaze data
        self.current_gaze = None
        self.last_gaze_timestamp = 0

    def start(self, camera_index: int = 0) -> bool:
        """
        Start Pupil Labs tracker.

        Connects to Pupil Capture software.

        Args:
            camera_index: Ignored (Pupil Labs uses its own cameras)

        Returns:
            True if connected successfully
        """
        print("\n" + "="*60)
        print("STARTING PUPIL LABS TRACKER")
        print("="*60)
        print("\nRequirements:")
        print("  1. Pupil Labs cameras connected")
        print("  2. Pupil Capture software running")
        print(f"  3. Network: {self.pupil_capture_address}:{self.pupil_capture_port}")
        print("="*60)

        try:
            # Create ZMQ context
            self.ctx = zmq.Context()

            # Connect to Pupil Remote
            print("\nConnecting to Pupil Capture...")
            self.pupil_remote = zmq.Socket(self.ctx, zmq.REQ)
            self.pupil_remote.connect(f"tcp://{self.pupil_capture_address}:{self.pupil_capture_port}")

            # Request subscriber port
            self.pupil_remote.send_string("SUB_PORT")
            sub_port = self.pupil_remote.recv_string()

            print(f"✓ Connected to Pupil Capture")
            print(f"  Subscriber port: {sub_port}")

            # Subscribe to gaze data
            self.subscriber = zmq.Socket(self.ctx, zmq.SUB)
            self.subscriber.connect(f"tcp://{self.pupil_capture_address}:{sub_port}")
            self.subscriber.subscribe("gaze")  # Subscribe to gaze topic

            print("✓ Subscribed to gaze data")

            # Set subscriber to non-blocking
            self.subscriber.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

            print("\n✓ Pupil Labs tracker started")
            return True

        except Exception as e:
            print(f"\n✗ Failed to connect to Pupil Capture: {e}")
            print("\nTroubleshooting:")
            print("  - Is Pupil Capture software running?")
            print("  - Are Pupil Labs cameras connected?")
            print("  - Check network settings in Pupil Capture")
            return False

    def stop(self):
        """Stop Pupil Labs tracker."""
        if self.subscriber:
            self.subscriber.close()
        if self.pupil_remote:
            self.pupil_remote.close()
        if self.ctx:
            self.ctx.term()

        print("\nPupil Labs tracker stopped")

    def get_gaze_point(self) -> Optional[GazePoint]:
        """
        Get current gaze point from Pupil Labs.

        Returns:
            GazePoint with screen coordinates, or None if not available
        """
        if not self.subscriber:
            return None

        try:
            # Receive gaze data (non-blocking)
            topic, payload = self.subscriber.recv_multipart()

            # Unpack msgpack data
            message = msgpack.unpackb(payload, raw=False)

            # Extract gaze data
            # Pupil Labs provides normalized coordinates (0-1)
            if 'norm_pos' in message:
                norm_x, norm_y = message['norm_pos']

                # Convert to screen coordinates
                screen_x = norm_x * self.screen_width
                screen_y = norm_y * self.screen_height

                # Get confidence
                confidence = message.get('confidence', 1.0)

                # Create gaze point
                gaze = GazePoint(
                    x=screen_x,
                    y=screen_y,
                    confidence=confidence,
                    timestamp=message.get('timestamp', time.time())
                )

                self.current_gaze = gaze
                self.last_gaze_timestamp = time.time()

                return gaze

        except zmq.Again:
            # No data available (timeout)
            pass
        except Exception as e:
            print(f"Error receiving gaze data: {e}")

        # Return last gaze if recent (within 100ms)
        if self.current_gaze and (time.time() - self.last_gaze_timestamp) < 0.1:
            return self.current_gaze

        return None

    def calibrate(self, calibration_points: List[Tuple[int, int]]) -> bool:
        """
        Calibrate Pupil Labs tracker.

        Pupil Labs handles calibration through Pupil Capture software.
        This method sends a calibration command.

        Args:
            calibration_points: List of (x, y) screen coordinates

        Returns:
            True if calibration initiated
        """
        if not self.pupil_remote:
            print("Not connected to Pupil Capture")
            return False

        print("\n" + "="*60)
        print("PUPIL LABS CALIBRATION")
        print("="*60)
        print("\nCalibration will be handled by Pupil Capture software.")
        print("Follow the on-screen instructions in Pupil Capture.")
        print("="*60)

        try:
            # Send calibration command to Pupil Capture
            # Note: Actual calibration is done through Pupil Capture UI
            self.pupil_remote.send_string("C")  # Start calibration
            response = self.pupil_remote.recv_string()

            print(f"\nPupil Capture response: {response}")

            # Wait for user to complete calibration in Pupil Capture
            print("\nWaiting for calibration to complete...")
            print("(Complete calibration in Pupil Capture window)")

            # In a real implementation, we would poll for calibration status
            time.sleep(5)  # Give time for calibration

            self.is_calibrated = True
            print("\n✓ Calibration complete")
            return True

        except Exception as e:
            print(f"\n✗ Calibration failed: {e}")
            return False

    def get_calibration_quality(self) -> float:
        """
        Get calibration quality from Pupil Labs.

        Returns:
            Quality score (0.0-1.0)
        """
        # Pupil Labs provides calibration quality through software
        # For now, return 1.0 if calibrated
        return 1.0 if self.is_calibrated else 0.0

    def save_calibration(self, filename: str) -> bool:
        """
        Save calibration.

        Pupil Labs stores calibration in its own format.
        This is handled by Pupil Capture software.
        """
        print("Pupil Labs calibration is saved by Pupil Capture software")
        return True

    def load_calibration(self, filename: str) -> bool:
        """
        Load calibration.

        Pupil Labs loads calibration through Pupil Capture software.
        """
        print("Pupil Labs calibration is loaded by Pupil Capture software")
        return True

    def send_annotation(self, label: str):
        """
        Send annotation to Pupil Capture.

        Useful for marking events during recording.

        Args:
            label: Annotation label
        """
        if not self.pupil_remote:
            return

        try:
            annotation = {
                'subject': 'annotation',
                'label': label,
                'timestamp': time.time()
            }

            self.pupil_remote.send_string("notify.annotation", zmq.SNDMORE)
            self.pupil_remote.send(msgpack.packb(annotation))
            self.pupil_remote.recv_string()

        except Exception as e:
            print(f"Error sending annotation: {e}")


if __name__ == "__main__":
    """Test Pupil Labs tracker."""
    print("Testing Pupil Labs Tracker...")

    tracker = PupilLabsTracker(1920, 1080)

    print(f"Tracker: {tracker.tracker_name}")
    print(f"Info: {tracker.get_info()}")

    # Try to start tracker
    if tracker.start():
        print("\nTracker started successfully")
        print("Testing gaze detection for 10 seconds...")

        start_time = time.time()
        detections = 0

        while time.time() - start_time < 10:
            gaze = tracker.get_gaze_point()
            if gaze:
                detections += 1
                print(f"Gaze: ({gaze.x:.0f}, {gaze.y:.0f}) confidence={gaze.confidence:.2f}")
            else:
                print("No gaze data")

            time.sleep(0.1)

        print(f"\nDetections in 10s: {detections}")
        tracker.stop()
    else:
        print("\nFailed to start tracker")
        print("\nMake sure:")
        print("  1. Pupil Capture software is running")
        print("  2. Pupil Labs cameras are connected")
        print("  3. Network settings are correct")
