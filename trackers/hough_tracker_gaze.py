"""
Hough Circle Gaze Tracker with Screen Coordinate Mapping

Improved version of Hough Circle tracker that maps pupil position
to actual screen coordinates through calibration.

This is the baseline/fallback tracker that works with any webcam or IR camera.
"""

import cv2
import numpy as np
import json
import time
from typing import Optional, Tuple, List
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from eye_tracker_framework import EyeTrackerBase, GazePoint, CalibrationPoint


class HoughGazeTracker(EyeTrackerBase):
    """
    Hough Circle tracker with gaze-to-screen calibration.

    Uses OpenCV's Hough Circle Transform to detect pupils,
    then maps pupil positions to screen coordinates via calibration.
    """

    def __init__(self, screen_width: int, screen_height: int):
        """Initialize Hough Gaze tracker."""
        super().__init__(screen_width, screen_height)
        self.tracker_name = "Hough Circle (Calibrated)"

        # Camera
        self.cap = None
        self.camera_width = 640
        self.camera_height = 480

        # Hough Circle parameters
        self.min_radius = 8
        self.max_radius = 35
        self.param1 = 50
        self.param2 = 30

        # Calibration data
        self.calibration_points: List[CalibrationPoint] = []
        self.calibration_matrix = None  # Transformation matrix

        # Pupil history for smoothing
        self.pupil_history = []
        self.max_history = 5

    def start(self, camera_index: int = 0) -> bool:
        """
        Start Hough tracker.

        Args:
            camera_index: Camera device index

        Returns:
            True if started successfully
        """
        print("\n" + "="*60)
        print("STARTING HOUGH CIRCLE TRACKER")
        print("="*60)

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            print(f"✗ Failed to open camera {camera_index}")
            return False

        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"✓ Camera {camera_index} opened")
        print(f"  Resolution: {self.camera_width}x{self.camera_height}")
        return True

    def stop(self):
        """Stop tracker."""
        if self.cap:
            self.cap.release()
        print("\nHough Circle tracker stopped")

    def detect_pupil(self) -> Optional[Tuple[float, float]]:
        """
        Detect pupil position in camera frame.

        Returns:
            (x, y) pupil position in camera coordinates, or None
        """
        if not self.cap:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype("int")

        # Find darkest circle (likely pupil)
        best_circle = None
        darkest_value = 255

        for (x, y, r) in circles:
            if x < r or y < r or x > self.camera_width - r or y > self.camera_height - r:
                continue

            region_size = max(3, r // 3)
            y_start = max(0, y - region_size)
            y_end = min(self.camera_height, y + region_size)
            x_start = max(0, x - region_size)
            x_end = min(self.camera_width, x + region_size)

            region = gray[y_start:y_end, x_start:x_end]
            avg_value = np.mean(region)

            if avg_value < darkest_value:
                darkest_value = avg_value
                best_circle = (float(x), float(y))

        if best_circle:
            # Smooth with history
            self.pupil_history.append(best_circle)
            if len(self.pupil_history) > self.max_history:
                self.pupil_history.pop(0)

            # Moving average
            avg_x = np.mean([p[0] for p in self.pupil_history])
            avg_y = np.mean([p[1] for p in self.pupil_history])

            return (avg_x, avg_y)

        return None

    def pupil_to_screen(self, pupil_x: float, pupil_y: float) -> Optional[Tuple[float, float]]:
        """
        Map pupil position to screen coordinates using calibration.

        Args:
            pupil_x: Pupil X in camera coordinates
            pupil_y: Pupil Y in camera coordinates

        Returns:
            (screen_x, screen_y) or None if not calibrated
        """
        if not self.is_calibrated or self.calibration_matrix is None:
            return None

        # Apply perspective transform
        pupil_point = np.array([[pupil_x, pupil_y]], dtype=np.float32)
        screen_point = cv2.perspectiveTransform(
            pupil_point.reshape(1, 1, 2),
            self.calibration_matrix
        )

        screen_x = float(screen_point[0][0][0])
        screen_y = float(screen_point[0][0][1])

        # Clamp to screen bounds
        screen_x = max(0, min(self.screen_width, screen_x))
        screen_y = max(0, min(self.screen_height, screen_y))

        return (screen_x, screen_y)

    def get_gaze_point(self) -> Optional[GazePoint]:
        """
        Get current gaze point on screen.

        Returns:
            GazePoint with screen coordinates, or None
        """
        # Detect pupil in camera frame
        pupil = self.detect_pupil()
        if not pupil:
            return None

        pupil_x, pupil_y = pupil

        if self.is_calibrated:
            # Map to screen coordinates
            screen_coords = self.pupil_to_screen(pupil_x, pupil_y)
            if not screen_coords:
                return None

            screen_x, screen_y = screen_coords

            # Confidence based on pupil detection quality
            # (Could be improved with additional metrics)
            confidence = 0.8

            return GazePoint(
                x=screen_x,
                y=screen_y,
                confidence=confidence,
                timestamp=time.time(),
                pupil_left=(pupil_x, pupil_y)
            )
        else:
            # Not calibrated - return camera coordinates
            # (for calibration process)
            return GazePoint(
                x=pupil_x,
                y=pupil_y,
                confidence=0.5,
                timestamp=time.time(),
                pupil_left=(pupil_x, pupil_y)
            )

    def calibrate(self, calibration_points: List[Tuple[int, int]]) -> bool:
        """
        Calibrate gaze tracker.

        Collects pupil positions for each calibration point,
        then computes transformation matrix.

        Args:
            calibration_points: List of (screen_x, screen_y) coordinates

        Returns:
            True if calibration successful
        """
        print("\n" + "="*60)
        print("HOUGH CIRCLE CALIBRATION")
        print("="*60)
        print(f"\nCalibration points: {len(calibration_points)}")
        print("Collecting pupil samples for each point...")

        self.calibration_points = []

        # Collect samples for each calibration point
        for i, (screen_x, screen_y) in enumerate(calibration_points):
            print(f"\nPoint {i+1}/{len(calibration_points)}: ({screen_x}, {screen_y})")
            print("  Looking for pupil...")

            cal_point = CalibrationPoint(
                screen_x=screen_x,
                screen_y=screen_y,
                gaze_samples=[]
            )

            # Collect 30 samples
            samples_needed = 30
            while len(cal_point.gaze_samples) < samples_needed:
                gaze = self.get_gaze_point()
                if gaze and gaze.pupil_left:
                    # Store raw pupil position
                    cal_point.gaze_samples.append(gaze)

                time.sleep(0.033)  # ~30 FPS

            avg_gaze = cal_point.get_average_gaze()
            print(f"  ✓ Collected {len(cal_point.gaze_samples)} samples")
            print(f"    Average pupil: ({avg_gaze[0]:.1f}, {avg_gaze[1]:.1f})")

            self.calibration_points.append(cal_point)

        # Compute calibration matrix
        print("\nComputing calibration matrix...")
        if self._compute_calibration_matrix():
            self.is_calibrated = True
            print("✓ Calibration complete")
            return True
        else:
            print("✗ Calibration failed")
            return False

    def _compute_calibration_matrix(self) -> bool:
        """
        Compute perspective transform matrix from calibration data.

        Returns:
            True if successful
        """
        if len(self.calibration_points) < 4:
            print("Need at least 4 calibration points")
            return False

        # Build source (pupil) and destination (screen) point arrays
        src_points = []
        dst_points = []

        for cal_point in self.calibration_points:
            avg_gaze = cal_point.get_average_gaze()
            if avg_gaze:
                src_points.append(avg_gaze)
                dst_points.append((cal_point.screen_x, cal_point.screen_y))

        if len(src_points) < 4:
            print("Not enough valid calibration points")
            return False

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Compute perspective transform
        # This maps pupil coordinates to screen coordinates
        self.calibration_matrix, _ = cv2.findHomography(src_points, dst_points)

        if self.calibration_matrix is None:
            return False

        print(f"Calibration matrix computed from {len(src_points)} points")
        return True

    def get_calibration_quality(self) -> float:
        """
        Get calibration quality score.

        Based on variance of calibration samples.

        Returns:
            Quality score (0.0-1.0)
        """
        if not self.is_calibrated:
            return 0.0

        total_variance = 0.0
        num_points = len(self.calibration_points)

        for cal_point in self.calibration_points:
            if len(cal_point.gaze_samples) < 2:
                continue

            # Calculate variance
            pupil_positions = [
                (s.pupil_left[0], s.pupil_left[1])
                for s in cal_point.gaze_samples
                if s.pupil_left
            ]

            if len(pupil_positions) < 2:
                continue

            xs = [p[0] for p in pupil_positions]
            ys = [p[1] for p in pupil_positions]

            variance = np.var(xs) + np.var(ys)
            total_variance += variance

        if num_points == 0:
            return 0.0

        avg_variance = total_variance / num_points

        # Lower variance = higher quality
        # Map variance (0-500) to quality (1.0-0.0)
        quality = max(0.0, min(1.0, 1.0 - (avg_variance / 500.0)))

        return quality

    def save_calibration(self, filename: str) -> bool:
        """Save calibration to file."""
        if not self.is_calibrated:
            return False

        try:
            data = {
                'screen_width': self.screen_width,
                'screen_height': self.screen_height,
                'camera_width': self.camera_width,
                'camera_height': self.camera_height,
                'calibration_matrix': self.calibration_matrix.tolist(),
                'quality': self.get_calibration_quality()
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"Calibration saved: {filename}")
            return True

        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False

    def load_calibration(self, filename: str) -> bool:
        """Load calibration from file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # Verify screen dimensions
            if (data['screen_width'] != self.screen_width or
                data['screen_height'] != self.screen_height):
                print("Warning: Screen dimensions don't match")
                return False

            self.calibration_matrix = np.array(data['calibration_matrix'], dtype=np.float32)
            self.is_calibrated = True

            print(f"Calibration loaded: {filename}")
            print(f"Quality: {data.get('quality', 'unknown')}")
            return True

        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False


if __name__ == "__main__":
    """Test Hough Gaze tracker."""
    print("Testing Hough Circle Gaze Tracker...")

    tracker = HoughGazeTracker(1920, 1080)

    if tracker.start(camera_index=0):
        print("\nTesting pupil detection for 5 seconds...")

        start_time = time.time()
        detections = 0

        while time.time() - start_time < 5:
            gaze = tracker.get_gaze_point()
            if gaze:
                detections += 1
                if gaze.pupil_left:
                    print(f"Pupil: ({gaze.pupil_left[0]:.0f}, {gaze.pupil_left[1]:.0f})")

            time.sleep(0.1)

        print(f"\nDetections in 5s: {detections}")

        # Test calibration with 9 points
        print("\nTesting calibration...")
        cal_points = [
            (200, 200), (960, 200), (1720, 200),
            (200, 540), (960, 540), (1720, 540),
            (200, 880), (960, 880), (1720, 880)
        ]

        # Note: In real use, user would look at each point
        # For this test, we'll just use current pupil position
        print("(Simulated calibration - in real use, look at each point)")

        tracker.stop()
    else:
        print("Failed to start tracker")
