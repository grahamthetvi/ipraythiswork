"""
Camera Selection Tool with Live Preview
Shows camera feed with pupil detection to help select best camera
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
import time

from gazetracking_wrapper import GazeTrackingWrapper, GAZETRACKING_AVAILABLE
from hough_tracker import HoughCircleTracker


class CameraSelector:
    """
    Interactive camera selection tool with live preview.
    Shows pupil detection to help choose best camera.
    """

    def __init__(self):
        """Initialize camera selector."""
        self.available_cameras = []
        self.current_camera_index = 0
        self.cap = None

        # Detection methods
        self.tracker_gaze = None
        self.tracker_hough = None
        self.active_tracker = 'hough'  # Start with simpler tracker

        # Initialize trackers
        self._initialize_trackers()

    def _initialize_trackers(self):
        """Initialize available tracking methods."""
        print("\nInitializing tracking methods...")

        # Try GazeTracking
        if GAZETRACKING_AVAILABLE:
            try:
                self.tracker_gaze = GazeTrackingWrapper()
                if self.tracker_gaze.available:
                    print("  ✓ GazeTracking available")
            except Exception as e:
                print(f"  ✗ GazeTracking failed: {e}")

        # Hough Circle (always available)
        try:
            self.tracker_hough = HoughCircleTracker()
            print("  ✓ Hough Circle available")
        except Exception as e:
            print(f"  ✗ Hough Circle failed: {e}")

    def detect_cameras(self) -> List[int]:
        """
        Detect all available cameras.

        Returns:
            List of camera indices
        """
        print("\nDetecting cameras...")
        self.available_cameras = []

        # Check first 10 camera indices
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.available_cameras.append(i)
                    height, width = frame.shape[:2]
                    print(f"  ✓ Camera {i}: {width}x{height}")
                cap.release()

        if len(self.available_cameras) == 0:
            print("  ✗ No cameras found!")
        else:
            print(f"\nFound {len(self.available_cameras)} camera(s)")

        return self.available_cameras

    def open_camera(self, camera_index: int) -> bool:
        """
        Open camera for preview.

        Args:
            camera_index: Camera device index

        Returns:
            True if opened successfully
        """
        # Close existing camera
        if self.cap is not None:
            self.cap.release()

        # Open new camera
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            print(f"✗ Failed to open camera {camera_index}")
            return False

        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Auto exposure
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        self.current_camera_index = camera_index
        print(f"✓ Opened camera {camera_index}")

        return True

    def detect_pupils(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect pupils in frame using active tracker.

        Args:
            frame: Camera frame

        Returns:
            Pupil position (x, y) or None
        """
        if self.active_tracker == 'gaze' and self.tracker_gaze:
            return self.tracker_gaze.detect_pupil(frame)
        elif self.active_tracker == 'hough' and self.tracker_hough:
            return self.tracker_hough.detect_pupil(frame)

        return None

    def draw_detection_overlay(self, frame: np.ndarray,
                               pupil: Optional[Tuple[int, int]]) -> np.ndarray:
        """
        Draw pupil detection overlay on frame.

        Args:
            frame: Camera frame
            pupil: Detected pupil position

        Returns:
            Frame with overlay
        """
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw pupil if detected
        if pupil is not None:
            # Large green circle around pupil
            cv2.circle(display, pupil, 20, (0, 255, 0), 3)
            # Center dot
            cv2.circle(display, pupil, 5, (0, 0, 255), -1)

            # Crosshair
            cv2.line(display, (pupil[0] - 30, pupil[1]),
                    (pupil[0] + 30, pupil[1]), (0, 255, 0), 2)
            cv2.line(display, (pupil[0], pupil[1] - 30),
                    (pupil[0], pupil[1] + 30), (0, 255, 0), 2)

            # Coordinates
            cv2.putText(display, f"Pupil: ({pupil[0]}, {pupil[1]})",
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)

            status_text = "TRACKING"
            status_color = (0, 255, 0)
        else:
            status_text = "NO DETECTION"
            status_color = (0, 0, 255)

        # Status bar at top
        cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 0), -1)
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # Status text
        cv2.putText(display, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        # Camera info
        cv2.putText(display, f"Camera {self.current_camera_index}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Tracker info
        tracker_name = 'GazeTracking' if self.active_tracker == 'gaze' else 'Hough Circle'
        cv2.putText(display, f"Tracker: {tracker_name}", (w - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Instructions at bottom
        instructions = [
            "ARROW KEYS: Switch camera  |  T: Switch tracker",
            "B: Brighter  D: Darker  |  SPACE: Select  ESC: Cancel"
        ]

        y_offset = h - 60
        for instruction in instructions:
            cv2.putText(display, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25

        # Draw center crosshair (for positioning)
        center_x, center_y = w // 2, h // 2
        cv2.line(display, (center_x - 20, center_y),
                (center_x + 20, center_y), (100, 100, 100), 1)
        cv2.line(display, (center_x, center_y - 20),
                (center_x, center_y + 20), (100, 100, 100), 1)

        return display

    def run_interactive_selection(self) -> Optional[int]:
        """
        Run interactive camera selection.

        Returns:
            Selected camera index or None if cancelled
        """
        # Detect cameras
        cameras = self.detect_cameras()

        if len(cameras) == 0:
            print("\n✗ No cameras found!")
            return None

        # If only one camera, auto-select
        if len(cameras) == 1:
            print(f"\n✓ Only one camera found, auto-selecting camera {cameras[0]}")
            return cameras[0]

        # Open first camera
        if not self.open_camera(cameras[0]):
            return None

        print("\n" + "="*60)
        print("CAMERA SELECTION")
        print("="*60)
        print(f"\nAvailable cameras: {cameras}")
        print("\nControls:")
        print("  LEFT/RIGHT ARROW: Switch camera")
        print("  T: Switch tracker (GazeTracking/Hough)")
        print("  B: Increase brightness")
        print("  D: Decrease brightness")
        print("  R: Reset camera settings")
        print("  SPACE: Select this camera")
        print("  ESC: Cancel")
        print("\nLook for clear pupil detection (green circles)")
        print("="*60)

        cv2.namedWindow('Camera Selection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Selection', 800, 600)

        exposure = -5
        selected_camera = None
        camera_idx_pos = 0  # Position in cameras list

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break

            # Detect pupils
            pupil = self.detect_pupils(frame)

            # Draw overlay
            display = self.draw_detection_overlay(frame, pupil)

            cv2.imshow('Camera Selection', display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("\n✗ Selection cancelled")
                selected_camera = None
                break

            elif key == ord(' '):  # SPACE
                selected_camera = self.current_camera_index
                print(f"\n✓ Selected camera {selected_camera}")
                break

            elif key == 81 or key == 2:  # LEFT ARROW
                # Previous camera
                camera_idx_pos = (camera_idx_pos - 1) % len(cameras)
                self.open_camera(cameras[camera_idx_pos])
                exposure = -5
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

            elif key == 83 or key == 3:  # RIGHT ARROW
                # Next camera
                camera_idx_pos = (camera_idx_pos + 1) % len(cameras)
                self.open_camera(cameras[camera_idx_pos])
                exposure = -5
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

            elif key == ord('t') or key == ord('T'):
                # Switch tracker
                if self.active_tracker == 'hough' and self.tracker_gaze:
                    self.active_tracker = 'gaze'
                    print("Switched to GazeTracking")
                elif self.active_tracker == 'gaze':
                    self.active_tracker = 'hough'
                    print("Switched to Hough Circle")

            elif key == ord('b') or key == ord('B'):
                # Brighter
                exposure = min(0, exposure + 1)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                print(f"Exposure: {exposure}")

            elif key == ord('d') or key == ord('D'):
                # Darker
                exposure = max(-13, exposure - 1)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                print(f"Exposure: {exposure}")

            elif key == ord('r') or key == ord('R'):
                # Reset
                exposure = -5
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                print("Reset camera settings")

        # Cleanup
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

        return selected_camera

    def test_camera(self, camera_index: int, duration: int = 10):
        """
        Test camera with pupil detection for specified duration.

        Args:
            camera_index: Camera to test
            duration: Test duration in seconds
        """
        if not self.open_camera(camera_index):
            return

        print(f"\n Testing camera {camera_index} for {duration} seconds...")
        print("Press 'q' to stop early")

        cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Test', 800, 600)

        start_time = time.time()
        detections = 0
        frames = 0

        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                break

            frames += 1
            pupil = self.detect_pupils(frame)

            if pupil is not None:
                detections += 1

            display = self.draw_detection_overlay(frame, pupil)

            # Test stats
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            detection_rate = (detections / frames * 100) if frames > 0 else 0

            cv2.putText(display, f"Test: {remaining:.1f}s remaining",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, f"Detection Rate: {detection_rate:.1f}%",
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow('Camera Test', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        # Show results
        detection_rate = (detections / frames * 100) if frames > 0 else 0
        print(f"\n Test Results:")
        print(f"  Frames: {frames}")
        print(f"  Detections: {detections}")
        print(f"  Detection Rate: {detection_rate:.1f}%")

        if detection_rate >= 80:
            print(f"  ✓ EXCELLENT - Camera works well")
        elif detection_rate >= 60:
            print(f"  ⚠ GOOD - Acceptable performance")
        else:
            print(f"  ✗ POOR - Consider different camera or lighting")


def main():
    """Main camera selection interface."""
    print("="*60)
    print("CAMERA SELECTION TOOL")
    print("="*60)
    print("\nThis tool helps you select the best camera for eye tracking.")
    print("It shows live pupil detection to verify tracking quality.")

    selector = CameraSelector()

    # Interactive selection
    selected = selector.run_interactive_selection()

    if selected is not None:
        print(f"\n{'='*60}")
        print(f"CAMERA {selected} SELECTED")
        print(f"{'='*60}")

        # Offer to test
        test = input("\nRun 10-second test? (y/n): ").strip().lower()
        if test == 'y':
            selector.test_camera(selected, duration=10)

        print(f"\n✓ Use camera index {selected} in your applications")
        print(f"  Example: tracker = Gaze9Region(camera_index={selected})")
    else:
        print("\n✗ No camera selected")


if __name__ == "__main__":
    main()
