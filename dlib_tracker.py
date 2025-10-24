"""
Dlib Eye Tracking Method for Mode 6 Beta Tester

Uses Dlib's facial landmark detection (68 landmarks) to track eyes.
Eye landmarks: 36-41 (left eye), 42-47 (right eye)

Installation:
    pip install dlib
    
Note: dlib requires CMake and C++ compiler. If installation fails:
    Windows: Install Visual Studio Build Tools
    Linux: sudo apt-get install build-essential cmake
    Mac: brew install cmake
"""

import cv2
import numpy as np
import logging

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logging.warning("Dlib not installed. Install with: pip install dlib")


class DlibEyeTracker:
    """Eye tracking using Dlib facial landmarks."""
    
    def __init__(self):
        """Initialize Dlib eye tracker."""
        self.name = "Dlib Landmarks"
        
        if not DLIB_AVAILABLE:
            raise ImportError("Dlib not installed")
        
        # Initialize Dlib's face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Download shape predictor if not present
        # User needs: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        try:
            self.predictor = dlib.shape_predictor(
                "shape_predictor_68_face_landmarks.dat"
            )
        except RuntimeError:
            raise FileNotFoundError(
                "Download shape_predictor_68_face_landmarks.dat from:\n"
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                "Extract and place in project root"
            )
        
        # Eye landmark indices (Dlib 68-point model)
        self.LEFT_EYE_INDICES = list(range(36, 42))   # 36-41
        self.RIGHT_EYE_INDICES = list(range(42, 48))  # 42-47
        
        logging.info("Dlib eye tracker initialized")
    
    def detect_pupil(self, frame):
        """
        Detect pupil center using Dlib facial landmarks.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            tuple: (x, y) pupil center, or None if not detected
        """
        # Convert to grayscale for Dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None
        
        # Use first detected face
        face = faces[0]
        
        # Get facial landmarks
        landmarks = self.predictor(gray, face)
        
        # Extract eye region landmarks
        left_eye_points = [
            (landmarks.part(i).x, landmarks.part(i).y)
            for i in self.LEFT_EYE_INDICES
        ]
        right_eye_points = [
            (landmarks.part(i).x, landmarks.part(i).y)
            for i in self.RIGHT_EYE_INDICES
        ]
        
        # Calculate eye centers (average of landmarks)
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
        
        # Find pupil within each eye region
        left_pupil = self._find_pupil_in_region(gray, left_eye_points)
        right_pupil = self._find_pupil_in_region(gray, right_eye_points)
        
        # Average both pupils if detected
        if left_pupil is not None and right_pupil is not None:
            pupil_center = (
                (left_pupil[0] + right_pupil[0]) // 2,
                (left_pupil[1] + right_pupil[1]) // 2
            )
        elif left_pupil is not None:
            pupil_center = left_pupil
        elif right_pupil is not None:
            pupil_center = right_pupil
        else:
            # Fallback: use eye center
            pupil_center = tuple(
                ((left_eye_center + right_eye_center) // 2).astype(int)
            )
        
        return pupil_center
    
    def _find_pupil_in_region(self, gray_frame, eye_points):
        """
        Find pupil within eye region defined by landmarks.
        
        Args:
            gray_frame: Grayscale image
            eye_points: List of (x, y) tuples defining eye region
            
        Returns:
            tuple: (x, y) pupil position or None
        """
        # Create mask for eye region
        eye_points_array = np.array(eye_points, dtype=np.int32)
        
        # Get bounding box
        x_coords = eye_points_array[:, 0]
        y_coords = eye_points_array[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Extract eye region
        eye_region = gray_frame[y_min:y_max, x_min:x_max]
        
        if eye_region.size == 0:
            return None
        
        # Apply threshold to find dark pupil
        # Pupil is darkest part of eye
        _, threshold = cv2.threshold(
            eye_region, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Find contours (pupil should be largest dark region)
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour (likely pupil)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Find center of contour
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"]) + x_min
        cy = int(M["m01"] / M["m00"]) + y_min
        
        return (cx, cy)
    
    def get_eye_landmarks(self, frame):
        """
        Get all eye landmarks for visualization.
        
        Args:
            frame: BGR image
            
        Returns:
            dict: {'left_eye': [(x,y), ...], 'right_eye': [(x,y), ...]}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None
        
        landmarks = self.predictor(gray, faces[0])
        
        left_eye = [
            (landmarks.part(i).x, landmarks.part(i).y)
            for i in self.LEFT_EYE_INDICES
        ]
        right_eye = [
            (landmarks.part(i).x, landmarks.part(i).y)
            for i in self.RIGHT_EYE_INDICES
        ]
        
        return {'left_eye': left_eye, 'right_eye': right_eye}
    
    def draw_landmarks(self, frame, landmarks):
        """
        Draw eye landmarks on frame for visualization.
        
        Args:
            frame: BGR image
            landmarks: Dict from get_eye_landmarks()
        """
        if landmarks is None:
            return frame
        
        # Draw left eye
        for i, point in enumerate(landmarks['left_eye']):
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
            if i < len(landmarks['left_eye']) - 1:
                cv2.line(
                    frame, point, landmarks['left_eye'][i + 1],
                    (0, 255, 0), 1
                )
        # Close the loop
        cv2.line(
            frame, landmarks['left_eye'][-1], landmarks['left_eye'][0],
            (0, 255, 0), 1
        )
        
        # Draw right eye
        for i, point in enumerate(landmarks['right_eye']):
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
            if i < len(landmarks['right_eye']) - 1:
                cv2.line(
                    frame, point, landmarks['right_eye'][i + 1],
                    (0, 255, 0), 1
                )
        cv2.line(
            frame, landmarks['right_eye'][-1], landmarks['right_eye'][0],
            (0, 255, 0), 1
        )
        
        return frame


def check_dlib_installation():
    """Check if Dlib is installed and shape predictor is available."""
    if not DLIB_AVAILABLE:
        return False, "Dlib not installed. Run: pip install dlib"
    
    import os
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        return False, (
            "shape_predictor_68_face_landmarks.dat not found.\n"
            "Download from: http://dlib.net/files/"
            "shape_predictor_68_face_landmarks.dat.bz2\n"
            "Extract and place in project root."
        )
    
    return True, "Dlib ready"


if __name__ == "__main__":
    # Test Dlib tracker
    print("Testing Dlib Eye Tracker...")
    
    # Check installation
    ok, msg = check_dlib_installation()
    print(f"Installation check: {msg}")
    
    if not ok:
        print("\nSetup instructions:")
        print("1. pip install dlib")
        print("2. Download shape predictor:")
        print("   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("3. Extract and place in project root")
        exit(1)
    
    # Test with webcam
    tracker = DlibEyeTracker()
    cap = cv2.VideoCapture(0)
    
    print("\nPress 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect pupil
        pupil = tracker.detect_pupil(frame)
        
        # Get landmarks for visualization
        landmarks = tracker.get_eye_landmarks(frame)
        
        # Draw landmarks
        frame = tracker.draw_landmarks(frame, landmarks)
        
        # Draw pupil
        if pupil:
            cv2.circle(frame, pupil, 5, (255, 0, 0), 2)
            cv2.putText(
                frame, f"Pupil: {pupil}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
            )
        
        cv2.imshow('Dlib Eye Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete")

