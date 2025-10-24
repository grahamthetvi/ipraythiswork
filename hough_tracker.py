"""
Hough Circle Transform Eye Tracking Method

Uses OpenCV's Hough Circle Transform to detect circular pupils.
Works especially well with IR camera images where pupils appear as
clear dark circles.

Advantages:
- Fast (25-30 FPS)
- Mathematically robust
- No ML dependencies
- Excellent with IR illumination

Algorithm:
1. Convert to grayscale
2. Apply Gaussian blur to reduce noise
3. Use Hough Circle Transform to find circles
4. Filter by size (pupils are typically 10-30 pixels)
5. Return center of strongest circle candidate
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class HoughCircleTracker:
    """Eye tracking using Hough Circle Transform."""
    
    def __init__(self):
        """Initialize Hough Circle tracker."""
        self.name = "Hough Circle Transform"
        
        # Tunable parameters
        self.min_radius = 8      # Minimum pupil radius (pixels)
        self.max_radius = 35     # Maximum pupil radius (pixels)
        self.dp = 1              # Inverse ratio of accumulator resolution
        self.min_dist = 50       # Minimum distance between circle centers
        self.param1 = 50         # Canny edge detector threshold
        self.param2 = 30         # Accumulator threshold (lower = more circles)
        
        # For smoothing/filtering
        self.last_pupil = None
        self.pupil_history = []
        self.max_history = 5
    
    def detect_pupil(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect pupil center using Hough Circle Transform.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            tuple: (x, y) pupil center, or None if not detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        if circles is None:
            # No circles detected, return last known position
            return self.last_pupil
        
        # Convert to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # Find best circle (darkest center = likely pupil)
        best_circle = self._find_best_circle(gray, circles)
        
        if best_circle is not None:
            x, y, r = best_circle
            pupil = (int(x), int(y))
            
            # Smooth using history
            pupil = self._smooth_position(pupil)
            
            self.last_pupil = pupil
            return pupil
        
        return self.last_pupil
    
    def _find_best_circle(self, gray_image: np.ndarray,
                         circles: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the circle most likely to be a pupil.
        
        Criteria:
        - Darkest center (pupils are dark)
        - Reasonable size
        - Not at edge of frame
        
        Args:
            gray_image: Grayscale image
            circles: Array of detected circles (x, y, radius)
            
        Returns:
            Best circle [x, y, radius] or None
        """
        best_circle = None
        darkest_value = 255  # Lighter is higher value
        
        height, width = gray_image.shape
        
        for (x, y, r) in circles:
            # Skip circles at edge of frame
            if x < r or y < r or x > width - r or y > height - r:
                continue
            
            # Get center brightness (pupil should be dark)
            center_value = gray_image[y, x]
            
            # Get average brightness in small region around center
            region_size = max(3, r // 3)
            y_start = max(0, y - region_size)
            y_end = min(height, y + region_size)
            x_start = max(0, x - region_size)
            x_end = min(width, x + region_size)
            
            region = gray_image[y_start:y_end, x_start:x_end]
            avg_value = np.mean(region)
            
            # Prefer darker circles (pupils absorb light)
            if avg_value < darkest_value:
                darkest_value = avg_value
                best_circle = np.array([x, y, r])
        
        return best_circle
    
    def _smooth_position(self, pupil: Tuple[int, int]) -> Tuple[int, int]:
        """
        Smooth pupil position using moving average.
        
        Args:
            pupil: Current pupil position (x, y)
            
        Returns:
            Smoothed position (x, y)
        """
        # Add to history
        self.pupil_history.append(pupil)
        
        # Keep only recent history
        if len(self.pupil_history) > self.max_history:
            self.pupil_history.pop(0)
        
        # Calculate moving average
        if len(self.pupil_history) > 0:
            avg_x = int(np.mean([p[0] for p in self.pupil_history]))
            avg_y = int(np.mean([p[1] for p in self.pupil_history]))
            return (avg_x, avg_y)
        
        return pupil
    
    def set_parameters(self, min_radius: int = None, max_radius: int = None,
                      param2: int = None):
        """
        Adjust detection parameters for different camera setups.
        
        Args:
            min_radius: Minimum pupil radius (pixels)
            max_radius: Maximum pupil radius (pixels)
            param2: Accumulator threshold (lower = more sensitive)
        """
        if min_radius is not None:
            self.min_radius = min_radius
        if max_radius is not None:
            self.max_radius = max_radius
        if param2 is not None:
            self.param2 = param2
    
    def optimize_for_ir(self):
        """Optimize parameters for IR camera."""
        # IR cameras show very clear circular pupils
        self.param2 = 20  # More sensitive (clear circles in IR)
        self.min_radius = 10
        self.max_radius = 40
        print("Optimized for IR camera")
    
    def optimize_for_webcam(self):
        """Optimize parameters for standard webcam."""
        # Standard webcams need less sensitivity
        self.param2 = 30  # Less sensitive (noisier images)
        self.min_radius = 8
        self.max_radius = 30
        print("Optimized for standard webcam")


if __name__ == "__main__":
    # Test Hough Circle tracker
    print("Testing Hough Circle Transform Eye Tracker...")
    
    tracker = HoughCircleTracker()
    cap = cv2.VideoCapture(0)
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  'i' - Optimize for IR camera")
    print("  'w' - Optimize for webcam")
    print("  '+' - Increase sensitivity")
    print("  '-' - Decrease sensitivity")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect pupil
        pupil = tracker.detect_pupil(frame)
        
        # Draw result
        if pupil:
            cv2.circle(frame, pupil, 5, (0, 255, 0), 2)
            cv2.circle(frame, pupil, 2, (0, 0, 255), -1)
            cv2.putText(
                frame, f"Pupil: {pupil}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Show parameters
        cv2.putText(
            frame, f"Sensitivity: {tracker.param2}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        cv2.putText(
            frame, f"Radius: {tracker.min_radius}-{tracker.max_radius}",
            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        
        cv2.imshow('Hough Circle Eye Tracking', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            tracker.optimize_for_ir()
        elif key == ord('w'):
            tracker.optimize_for_webcam()
        elif key == ord('+'):
            tracker.param2 = max(10, tracker.param2 - 5)
            print(f"Sensitivity increased (param2={tracker.param2})")
        elif key == ord('-'):
            tracker.param2 = min(100, tracker.param2 + 5)
            print(f"Sensitivity decreased (param2={tracker.param2})")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete")

