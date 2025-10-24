"""
Production-Ready 9-Region Eye Gaze Tracker for AAC
Designed for students with visual impairments

Features:
- 9-region grid detection (3x3)
- Multiple detection methods (GazeTracking primary, Hough Circle fallback)
- Auto-detection of best method for current camera
- Dwell time confirmation (avoid accidental selections)
- Confidence scoring (only report high-confidence detections)
- Smoothing/filtering to reduce jitter
- Visual feedback showing current gaze region
- Lost tracking recovery
- 30+ FPS performance target
- <100ms latency target

Requirements:
- OpenCV (cv2)
- NumPy
- GazeTracking library (optional but recommended)
- calibration_system.py
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict, List
from collections import deque
from dataclasses import dataclass

# Import tracking methods
from gazetracking_wrapper import GazeTrackingWrapper, GAZETRACKING_AVAILABLE
from hough_tracker import HoughCircleTracker
from calibration_system import CalibrationSystem


@dataclass
class GazeEvent:
    """Represents a gaze event (region selection)."""
    region_id: int
    confidence: float
    timestamp: float
    pupil_position: Tuple[int, int]


class Gaze9Region:
    """
    Production-ready 9-region eye gaze tracker.

    Maps eye gaze to 9 screen regions (3x3 grid).
    Uses multiple detection methods with automatic fallback.
    """

    def __init__(self,
                 screen_width: int = 1280,
                 screen_height: int = 720,
                 camera_index: int = 0,
                 dwell_time: float = 1.0,
                 confidence_threshold: float = 0.75,
                 smoothing_window: int = 5):
        """
        Initialize 9-region gaze tracker.

        Args:
            screen_width: Display width in pixels
            screen_height: Display height in pixels
            camera_index: Camera device index
            dwell_time: Time (seconds) user must look at region to select
            confidence_threshold: Minimum confidence for region detection
            smoothing_window: Number of frames for smoothing filter
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_index = camera_index
        self.dwell_time = dwell_time
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window

        # Calibration system
        self.calibration = CalibrationSystem(screen_width, screen_height, audio_feedback=False)

        # Detection methods (in priority order)
        self.trackers = []
        self.active_tracker = None
        self._initialize_trackers()

        # Camera
        self.cap = None
        self.camera_fps = 30
        self.frame_time = 1.0 / self.camera_fps

        # Smoothing/filtering
        self.pupil_history = deque(maxlen=smoothing_window)
        self.region_history = deque(maxlen=smoothing_window)

        # Dwell time tracking
        self.current_region = None
        self.region_start_time = None
        self.last_gaze_event: Optional[GazeEvent] = None

        # Performance metrics
        self.frame_count = 0
        self.fps = 0
        self.latency_ms = 0
        self.tracking_lost_frames = 0
        self.last_fps_update = time.time()

        # Visual feedback
        self.show_visualization = True
        self.show_regions = True
        self.show_metrics = True

        # Region colors for visualization
        self.region_colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 100), (150, 200, 100), (100, 150, 200)
        ]

    def _initialize_trackers(self):
        """Initialize available tracking methods."""
        print("\nInitializing tracking methods...")

        # Try GazeTracking first (proven to work well)
        if GAZETRACKING_AVAILABLE:
            try:
                tracker = GazeTrackingWrapper()
                if tracker.available:
                    self.trackers.append(('GazeTracking', tracker))
                    print("  ✓ GazeTracking available (PRIMARY)")
            except Exception as e:
                print(f"  ✗ GazeTracking failed: {e}")

        # Add Hough Circle as fallback
        try:
            tracker = HoughCircleTracker()
            self.trackers.append(('HoughCircle', tracker))
            print("  ✓ Hough Circle available (FALLBACK)")
        except Exception as e:
            print(f"  ✗ Hough Circle failed: {e}")

        if len(self.trackers) == 0:
            raise RuntimeError("No tracking methods available!")

        # Use first available tracker
        self.active_tracker = self.trackers[0]
        print(f"\nActive tracker: {self.active_tracker[0]}")

    def start_camera(self) -> bool:
        """
        Start camera capture.

        Returns:
            True if camera started successfully
        """
        print(f"\nStarting camera {self.camera_index}...")

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"✗ Failed to open camera {self.camera_index}")
            return False

        # Configure camera for best performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # Get actual FPS
        self.camera_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.camera_fps == 0:
            self.camera_fps = 30

        print(f"✓ Camera started: {self.camera_fps} FPS")
        return True

    def stop_camera(self):
        """Stop camera capture."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def detect_pupil(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect pupil position using active tracker.

        Args:
            frame: Camera frame (BGR)

        Returns:
            Pupil position (x, y) or None if not detected
        """
        if not self.active_tracker:
            return None

        tracker_name, tracker = self.active_tracker

        try:
            pupil = tracker.detect_pupil(frame)
            return pupil
        except Exception as e:
            print(f"Tracker error: {e}")
            return None

    def smooth_pupil_position(self, pupil: Tuple[int, int]) -> Tuple[int, int]:
        """
        Smooth pupil position using moving average.

        Args:
            pupil: Current pupil position

        Returns:
            Smoothed pupil position
        """
        if pupil is None:
            return None

        # Add to history
        self.pupil_history.append(pupil)

        # Calculate weighted average (recent positions weighted more)
        if len(self.pupil_history) == 0:
            return pupil

        weights = np.linspace(0.5, 1.0, len(self.pupil_history))
        weights /= weights.sum()

        avg_x = sum(p[0] * w for p, w in zip(self.pupil_history, weights))
        avg_y = sum(p[1] * w for p, w in zip(self.pupil_history, weights))

        return (int(avg_x), int(avg_y))

    def detect_region(self, frame: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Detect which region user is looking at.

        Args:
            frame: Camera frame

        Returns:
            (region_id, confidence) or None if not detected
        """
        # Measure latency
        start_time = time.time()

        # Detect pupil
        pupil = self.detect_pupil(frame)

        if pupil is None:
            self.tracking_lost_frames += 1
            self.pupil_history.clear()
            self.region_history.clear()
            return None

        # Reset lost tracking counter
        self.tracking_lost_frames = 0

        # Smooth pupil position
        smoothed_pupil = self.smooth_pupil_position(pupil)

        # Map to region using calibration
        region_id = self.calibration.map_pupil_to_region(smoothed_pupil)

        if region_id is None:
            return None

        # Get confidence score
        confidence = self.calibration.get_region_confidence(smoothed_pupil, region_id)

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            return None

        # Add to region history for stability
        self.region_history.append(region_id)

        # Require majority vote from recent history
        if len(self.region_history) >= 3:
            # Find most common region
            region_counts = {}
            for r in self.region_history:
                region_counts[r] = region_counts.get(r, 0) + 1

            most_common = max(region_counts.items(), key=lambda x: x[1])
            region_id = most_common[0]

        # Calculate latency
        self.latency_ms = (time.time() - start_time) * 1000

        return (region_id, confidence)

    def process_dwell_time(self, region_id: Optional[int],
                          confidence: float) -> Optional[GazeEvent]:
        """
        Process dwell time and generate gaze event.

        Args:
            region_id: Current region being looked at
            confidence: Confidence score

        Returns:
            GazeEvent if dwell time completed, None otherwise
        """
        current_time = time.time()

        # Check if region changed
        if region_id != self.current_region:
            # Region changed, restart dwell timer
            self.current_region = region_id
            self.region_start_time = current_time
            return None

        # Still looking at same region
        if self.region_start_time is None:
            self.region_start_time = current_time
            return None

        # Check dwell time
        dwell_duration = current_time - self.region_start_time

        if dwell_duration >= self.dwell_time:
            # Dwell time completed - generate event
            event = GazeEvent(
                region_id=region_id,
                confidence=confidence,
                timestamp=current_time,
                pupil_position=self.pupil_history[-1] if self.pupil_history else None
            )

            # Reset dwell timer
            self.region_start_time = current_time

            return event

        return None

    def update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1

        current_time = time.time()
        elapsed = current_time - self.last_fps_update

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time

    def draw_visualization(self, frame: np.ndarray,
                          region_id: Optional[int],
                          confidence: float) -> np.ndarray:
        """
        Draw visualization overlay on frame.

        Args:
            frame: Camera frame
            region_id: Current region (or None)
            confidence: Confidence score

        Returns:
            Annotated frame
        """
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw 9 regions if enabled
        if self.show_regions:
            self._draw_regions(display, region_id, confidence)

        # Draw pupil position
        if len(self.pupil_history) > 0:
            pupil = self.pupil_history[-1]
            cv2.circle(display, pupil, 8, (255, 0, 255), -1)
            cv2.circle(display, pupil, 15, (255, 0, 255), 2)

        # Draw metrics if enabled
        if self.show_metrics:
            self._draw_metrics(display, region_id, confidence)

        # Draw dwell time progress
        if region_id is not None and self.region_start_time is not None:
            self._draw_dwell_progress(display, region_id)

        return display

    def _draw_regions(self, display: np.ndarray, current_region: Optional[int],
                     confidence: float):
        """Draw 9 region grid."""
        h, w = display.shape[:2]

        # Calculate region boundaries
        region_w = w // 3
        region_h = h // 3

        for region_id in range(9):
            row = region_id // 3
            col = region_id % 3

            x1 = col * region_w
            y1 = row * region_h
            x2 = x1 + region_w
            y2 = y1 + region_h

            # Draw region border
            color = (100, 100, 100)
            thickness = 1

            if region_id == current_region:
                # Highlight current region
                color = self.region_colors[region_id]
                thickness = 3

                # Fill with semi-transparent color
                overlay = display.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

            # Draw region number
            text_size = cv2.getTextSize(str(region_id), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = x1 + (region_w - text_size[0]) // 2
            text_y = y1 + (region_h + text_size[1]) // 2

            cv2.putText(display, str(region_id), (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    def _draw_metrics(self, display: np.ndarray, region_id: Optional[int],
                     confidence: float):
        """Draw performance metrics."""
        h, w = display.shape[:2]

        # Background for metrics
        cv2.rectangle(display, (0, 0), (w, 120), (0, 0, 0), -1)
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        # Metrics text
        metrics = [
            f"FPS: {self.fps:.1f}",
            f"Latency: {self.latency_ms:.1f}ms",
            f"Region: {region_id if region_id is not None else 'None'}",
            f"Confidence: {confidence:.2f}",
        ]

        y_offset = 25
        for metric in metrics:
            cv2.putText(display, metric, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        # Tracker name
        tracker_name = self.active_tracker[0] if self.active_tracker else "None"
        cv2.putText(display, f"Tracker: {tracker_name}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Calibration quality
        if self.calibration.is_calibrated:
            quality_text = f"Cal: {self.calibration.get_quality_description()}"
            quality_color = (0, 255, 0) if self.calibration.calibration_quality > 0.7 else (0, 255, 255)
            cv2.putText(display, quality_text, (w - 200, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)

    def _draw_dwell_progress(self, display: np.ndarray, region_id: int):
        """Draw dwell time progress indicator."""
        if self.region_start_time is None:
            return

        current_time = time.time()
        dwell_duration = current_time - self.region_start_time
        progress = min(1.0, dwell_duration / self.dwell_time)

        # Calculate region center
        h, w = display.shape[:2]
        region_w = w // 3
        region_h = h // 3

        row = region_id // 3
        col = region_id % 3

        center_x = col * region_w + region_w // 2
        center_y = row * region_h + region_h // 2

        # Draw circular progress indicator
        radius = 50
        thickness = 8

        # Background circle
        cv2.circle(display, (center_x, center_y), radius, (100, 100, 100), thickness)

        # Progress arc
        angle = int(360 * progress)
        if angle > 0:
            # Draw arc (OpenCV uses angles in degrees, starting from 3 o'clock)
            cv2.ellipse(display, (center_x, center_y), (radius, radius),
                       -90, 0, angle, (0, 255, 0), thickness)

        # Progress percentage
        progress_text = f"{int(progress * 100)}%"
        text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2

        cv2.putText(display, progress_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def run_interactive(self, on_gaze_event=None):
        """
        Run interactive gaze tracking loop.

        Args:
            on_gaze_event: Callback function(GazeEvent) called on selection
        """
        if not self.start_camera():
            print("Failed to start camera")
            return

        print("\n" + "="*60)
        print("9-REGION GAZE TRACKER - INTERACTIVE MODE")
        print("="*60)
        print("\nControls:")
        print("  ESC - Quit")
        print("  'c' - Start calibration")
        print("  's' - Save calibration")
        print("  'l' - Load calibration")
        print("  'v' - Toggle visualization")
        print("  'm' - Toggle metrics")
        print("  '+' - Increase dwell time")
        print("  '-' - Decrease dwell time")
        print("\nPress any key to start...")

        cv2.namedWindow('9-Region Gaze Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('9-Region Gaze Tracker', self.screen_width, self.screen_height)

        running = True

        while running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Detect region
            result = self.detect_region(frame)

            region_id = None
            confidence = 0.0

            if result is not None:
                region_id, confidence = result

            # Process dwell time
            gaze_event = self.process_dwell_time(region_id, confidence)

            if gaze_event is not None:
                print(f"\nGaze Event: Region {gaze_event.region_id} "
                      f"(Confidence: {gaze_event.confidence:.2f})")

                if on_gaze_event:
                    on_gaze_event(gaze_event)

            # Update performance metrics
            self.update_fps()

            # Draw visualization
            if self.show_visualization:
                display = self.draw_visualization(frame, region_id, confidence)
            else:
                display = frame

            cv2.imshow('9-Region Gaze Tracker', display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                running = False
            elif key == ord('c'):
                print("\nStarting calibration...")
                self._run_calibration_interactive()
            elif key == ord('s'):
                student_name = input("\nEnter student name: ").strip()
                if student_name:
                    self.calibration.save_calibration(student_name)
            elif key == ord('l'):
                students = self.calibration.list_calibrations()
                if students:
                    print("\nAvailable calibrations:")
                    for i, name in enumerate(students):
                        print(f"  {i+1}. {name}")
                    choice = input("Enter number or name: ").strip()
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(students):
                            self.calibration.load_calibration(students[idx])
                    except:
                        self.calibration.load_calibration(choice)
            elif key == ord('v'):
                self.show_visualization = not self.show_visualization
            elif key == ord('m'):
                self.show_metrics = not self.show_metrics
            elif key == ord('+'):
                self.dwell_time = min(5.0, self.dwell_time + 0.1)
                print(f"Dwell time: {self.dwell_time:.1f}s")
            elif key == ord('-'):
                self.dwell_time = max(0.5, self.dwell_time - 0.1)
                print(f"Dwell time: {self.dwell_time:.1f}s")

        self.stop_camera()
        cv2.destroyAllWindows()

    def _run_calibration_interactive(self):
        """Run interactive calibration process."""
        self.calibration.start_calibration()

        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Calibration', self.screen_width, self.screen_height)

        while self.calibration.current_point_index < 9:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect pupil
            pupil = self.detect_pupil(frame)

            # Add calibration sample
            if pupil is not None:
                completed = self.calibration.add_sample(pupil)
                if completed:
                    is_done = self.calibration.next_point()
                    if is_done:
                        break

            # Draw calibration screen
            display = self.calibration.draw_calibration_screen(frame, pupil)
            cv2.imshow('Calibration', display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - skip point
                self.calibration.next_point()

        cv2.destroyWindow('Calibration')

        print(f"\nCalibration complete!")
        print(f"Quality: {self.calibration.calibration_quality:.2%} "
              f"({self.calibration.get_quality_description()})")


if __name__ == "__main__":
    """Test 9-region gaze tracker."""
    print("="*60)
    print("9-REGION GAZE TRACKER - PRODUCTION TEST")
    print("="*60)

    # Create tracker
    tracker = Gaze9Region(
        screen_width=1280,
        screen_height=720,
        camera_index=0,
        dwell_time=1.5,
        confidence_threshold=0.75,
        smoothing_window=5
    )

    # Callback for gaze events
    def handle_gaze_event(event: GazeEvent):
        """Handle gaze selection events."""
        print(f"Selected Region {event.region_id}!")

    # Run interactive mode
    tracker.run_interactive(on_gaze_event=handle_gaze_event)

    print("\n" + "="*60)
    print("SHUTDOWN COMPLETE")
    print("="*60)
