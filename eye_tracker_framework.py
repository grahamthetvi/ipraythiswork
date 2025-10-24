"""
Unified Eye Tracking Testing Framework

This framework allows testing multiple eye tracking systems to find
which one provides the most accurate gaze-to-screen-coordinate mapping.

Supported Systems:
1. Hough Circle (current, baseline)
2. GazeTracking (if available)
3. WebGazer.js (web-based)
4. Pupil Labs Core (if cameras available)

Purpose: Proof-of-concept testing to determine best system for AAC
"""

import abc
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum


class TrackerType(Enum):
    """Available eye tracking systems."""
    HOUGH_CIRCLE = "hough_circle"
    GAZETRACKING = "gazetracking"
    WEBGAZER = "webgazer"
    PUPIL_LABS = "pupil_labs"


@dataclass
class GazePoint:
    """
    Represents a gaze point on screen.

    Attributes:
        x: Screen X coordinate (0-screen_width)
        y: Screen Y coordinate (0-screen_height)
        confidence: Confidence score (0.0-1.0)
        timestamp: Unix timestamp
        pupil_left: Left pupil position in camera frame (optional)
        pupil_right: Right pupil position in camera frame (optional)
    """
    x: float
    y: float
    confidence: float
    timestamp: float
    pupil_left: Optional[Tuple[float, float]] = None
    pupil_right: Optional[Tuple[float, float]] = None


@dataclass
class CalibrationPoint:
    """A calibration point for gaze mapping."""
    screen_x: int
    screen_y: int
    gaze_samples: List[GazePoint]

    def get_average_gaze(self) -> Optional[Tuple[float, float]]:
        """Get average gaze position for this calibration point."""
        if not self.gaze_samples:
            return None
        avg_x = np.mean([s.x for s in self.gaze_samples])
        avg_y = np.mean([s.y for s in self.gaze_samples])
        return (avg_x, avg_y)


class EyeTrackerBase(abc.ABC):
    """
    Abstract base class for all eye tracking systems.

    All eye trackers must implement these methods to work with
    the testing framework.
    """

    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize eye tracker.

        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.is_calibrated = False
        self.tracker_name = "Unknown"

    @abc.abstractmethod
    def start(self, camera_index: int = 0) -> bool:
        """
        Start the eye tracker.

        Args:
            camera_index: Camera device index

        Returns:
            True if started successfully
        """
        pass

    @abc.abstractmethod
    def stop(self):
        """Stop the eye tracker and release resources."""
        pass

    @abc.abstractmethod
    def get_gaze_point(self) -> Optional[GazePoint]:
        """
        Get current gaze point on screen.

        Returns:
            GazePoint with screen coordinates, or None if not detected
        """
        pass

    @abc.abstractmethod
    def calibrate(self, calibration_points: List[Tuple[int, int]]) -> bool:
        """
        Calibrate the eye tracker.

        Args:
            calibration_points: List of (x, y) screen coordinates to calibrate

        Returns:
            True if calibration successful
        """
        pass

    @abc.abstractmethod
    def get_calibration_quality(self) -> float:
        """
        Get calibration quality score.

        Returns:
            Quality score (0.0-1.0)
        """
        pass

    @abc.abstractmethod
    def save_calibration(self, filename: str) -> bool:
        """Save calibration to file."""
        pass

    @abc.abstractmethod
    def load_calibration(self, filename: str) -> bool:
        """Load calibration from file."""
        pass

    def get_info(self) -> Dict[str, str]:
        """
        Get tracker information.

        Returns:
            Dictionary with tracker details
        """
        return {
            'name': self.tracker_name,
            'type': self.__class__.__name__,
            'calibrated': str(self.is_calibrated),
            'screen_resolution': f"{self.screen_width}x{self.screen_height}"
        }


class TrackerFactory:
    """Factory for creating eye tracker instances."""

    @staticmethod
    def create_tracker(tracker_type: TrackerType,
                      screen_width: int,
                      screen_height: int) -> Optional[EyeTrackerBase]:
        """
        Create an eye tracker instance.

        Args:
            tracker_type: Type of tracker to create
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels

        Returns:
            Eye tracker instance, or None if not available
        """
        if tracker_type == TrackerType.HOUGH_CIRCLE:
            from trackers.hough_tracker_gaze import HoughGazeTracker
            return HoughGazeTracker(screen_width, screen_height)

        elif tracker_type == TrackerType.GAZETRACKING:
            try:
                from trackers.gazetracking_gaze import GazeTrackingGaze
                return GazeTrackingGaze(screen_width, screen_height)
            except ImportError:
                print("GazeTracking not available")
                return None

        elif tracker_type == TrackerType.WEBGAZER:
            from trackers.webgazer_tracker import WebGazerTracker
            return WebGazerTracker(screen_width, screen_height)

        elif tracker_type == TrackerType.PUPIL_LABS:
            try:
                from trackers.pupil_labs_tracker import PupilLabsTracker
                return PupilLabsTracker(screen_width, screen_height)
            except ImportError:
                print("Pupil Labs not available")
                return None

        return None

    @staticmethod
    def get_available_trackers() -> List[TrackerType]:
        """
        Get list of available trackers on this system.

        Returns:
            List of available TrackerType enums
        """
        available = []

        # Hough Circle is always available (uses OpenCV)
        available.append(TrackerType.HOUGH_CIRCLE)

        # Check GazeTracking
        try:
            from gaze_tracking import GazeTracking
            available.append(TrackerType.GAZETRACKING)
        except ImportError:
            pass

        # WebGazer is always available (JavaScript-based)
        available.append(TrackerType.WEBGAZER)

        # Check Pupil Labs
        try:
            import zmq  # Pupil Labs uses ZMQ
            available.append(TrackerType.PUPIL_LABS)
        except ImportError:
            pass

        return available


if __name__ == "__main__":
    """Test the framework."""
    print("="*60)
    print("EYE TRACKING TESTING FRAMEWORK")
    print("="*60)

    # Get available trackers
    available = TrackerFactory.get_available_trackers()

    print(f"\nAvailable trackers: {len(available)}")
    for tracker_type in available:
        print(f"  - {tracker_type.value}")

    # Try to create each tracker
    print("\nTesting tracker creation...")
    for tracker_type in available:
        tracker = TrackerFactory.create_tracker(tracker_type, 1920, 1080)
        if tracker:
            info = tracker.get_info()
            print(f"\n✓ {tracker_type.value}:")
            for key, value in info.items():
                print(f"    {key}: {value}")
        else:
            print(f"\n✗ {tracker_type.value}: Failed to create")

    print("\n" + "="*60)
    print("Framework ready for testing")
    print("="*60)
