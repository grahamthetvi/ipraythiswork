"""
GazeTracking Library Integration

Wrapper for Antoine Lamé's GazeTracking library:
https://github.com/antoinelame/GazeTracking

This provides:
- Directional gaze detection (left/center/right)
- Pupil position tracking
- Blink detection
- Clean, simple API

Installation:
    git clone https://github.com/antoinelame/GazeTracking.git
    cd GazeTracking
    pip install -r requirements.txt
    
Or copy the gaze_tracking folder to this project directory.
"""

import numpy as np
from typing import Optional, Tuple

try:
    from gaze_tracking import GazeTracking
    GAZETRACKING_AVAILABLE = True
except ImportError:
    GAZETRACKING_AVAILABLE = False
    print("GazeTracking not available. Install from:")
    print("https://github.com/antoinelame/GazeTracking")


class GazeTrackingWrapper:
    """
    Wrapper for Antoine Lamé's GazeTracking library.
    
    Provides standardized interface for Mode 6.
    """
    
    def __init__(self):
        """Initialize GazeTracking wrapper."""
        self.name = "GazeTracking (Antoine Lamé)"
        
        if not GAZETRACKING_AVAILABLE:
            self.available = False
            return
        
        try:
            self.gaze = GazeTracking()
            self.available = True
            print("✓ GazeTracking library loaded")
            print("  Source: https://github.com/antoinelame/GazeTracking")
        except Exception as e:
            print(f"Failed to initialize GazeTracking: {e}")
            self.available = False
    
    def detect_pupil(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect pupil position using GazeTracking.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            tuple: (x, y) pupil center (averaged from both eyes)
        """
        if not self.available:
            return None
        
        try:
            # Refresh the gaze tracking with current frame
            self.gaze.refresh(frame)
            
            # Get pupil positions
            left_pupil = self.gaze.pupil_left_coords()
            right_pupil = self.gaze.pupil_right_coords()
            
            # Average both pupils if available
            if left_pupil is not None and right_pupil is not None:
                avg_x = (left_pupil[0] + right_pupil[0]) // 2
                avg_y = (left_pupil[1] + right_pupil[1]) // 2
                return (avg_x, avg_y)
            elif left_pupil is not None:
                return left_pupil
            elif right_pupil is not None:
                return right_pupil
            
            return None
            
        except Exception as e:
            print(f"GazeTracking error: {e}")
            return None
    
    def get_direction(self, frame: np.ndarray) -> Optional[str]:
        """
        Get gaze direction (left/center/right).
        
        Args:
            frame: BGR image from camera
            
        Returns:
            str: "left", "center", "right", or None
        """
        if not self.available:
            return None
        
        try:
            self.gaze.refresh(frame)
            
            if self.gaze.is_left():
                return "left"
            elif self.gaze.is_right():
                return "right"
            elif self.gaze.is_center():
                return "center"
            
            return None
        except:
            return None
    
    def is_blinking(self, frame: np.ndarray) -> bool:
        """
        Detect if user is blinking.
        
        Args:
            frame: BGR image
            
        Returns:
            bool: True if blinking
        """
        if not self.available:
            return False
        
        try:
            self.gaze.refresh(frame)
            return self.gaze.is_blinking()
        except:
            return False
    
    def get_horizontal_ratio(self, frame: np.ndarray) -> Optional[float]:
        """
        Get horizontal gaze ratio (0.0=right, 0.5=center, 1.0=left).
        
        Args:
            frame: BGR image
            
        Returns:
            float: Ratio between 0.0 and 1.0, or None
        """
        if not self.available:
            return None
        
        try:
            self.gaze.refresh(frame)
            return self.gaze.horizontal_ratio()
        except:
            return None
    
    def get_annotated_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Get frame with pupils highlighted.
        
        Args:
            frame: BGR image
            
        Returns:
            Annotated frame with pupil overlay
        """
        if not self.available:
            return frame
        
        try:
            self.gaze.refresh(frame)
            return self.gaze.annotated_frame()
        except:
            return frame


def check_gazetracking_installation():
    """Check if GazeTracking is properly installed."""
    if not GAZETRACKING_AVAILABLE:
        return False, (
            "GazeTracking library not found.\n\n"
            "Installation:\n"
            "1. git clone https://github.com/antoinelame/GazeTracking.git\n"
            "2. cd GazeTracking\n"
            "3. pip install -r requirements.txt\n"
            "4. Copy 'gaze_tracking' folder to this project\n\n"
            "Or install dependencies:\n"
            "  pip install opencv-python dlib numpy\n"
            "  Download shape_predictor_68_face_landmarks.dat"
        )
    
    return True, "GazeTracking ready"


def list_cameras():
    """Find all available cameras."""
    print("\n" + "="*60)
    print("DETECTING CAMERAS")
    print("="*60)
    available_cameras = []
    
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"  Camera {i}: Found ✓")
            cap.release()
    
    return available_cameras


def select_camera():
    """Let user select a camera."""
    cameras = list_cameras()
    
    if len(cameras) == 0:
        print("\n❌ No cameras found!")
        return None
    
    if len(cameras) == 1:
        print(f"\n✓ Using camera {cameras[0]}")
        return cameras[0]
    
    print("\nAvailable cameras:")
    for i, cam_idx in enumerate(cameras):
        print(f"  {i+1}. Camera {cam_idx}")
    
    while True:
        try:
            choice = input(f"\nSelect camera (1-{len(cameras)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(cameras):
                print(f"\n✓ Selected camera {cameras[idx]}")
                return cameras[idx]
            print("❌ Invalid choice!")
        except:
            print("❌ Invalid input!")


if __name__ == "__main__":
    # Test GazeTracking wrapper
    import cv2
    
    print("="*60)
    print("GAZETRACKING WRAPPER TEST")
    print("="*60)
    print("\nSource: https://github.com/antoinelame/GazeTracking")
    print("By: Antoine Lamé (2.4k GitHub stars)")
    
    # Check installation
    ok, msg = check_gazetracking_installation()
    print(f"\n{msg}")
    
    if not ok:
        print("\n❌ Please install GazeTracking library first!")
        print("\nRun: install_gazetracking.bat")
        input("\nPress Enter to exit...")
        exit(1)
    
    # Select camera
    camera_index = select_camera()
    if camera_index is None:
        input("\nPress Enter to exit...")
        exit(1)
    
    # Initialize wrapper
    print("\n" + "="*60)
    print("INITIALIZING")
    print("="*60)
    wrapper = GazeTrackingWrapper()
    
    if not wrapper.available:
        print("❌ Failed to initialize GazeTracking")
        input("\nPress Enter to exit...")
        exit(1)
    
    # Setup camera with good brightness
    print("\n✓ Setting up camera...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_index}")
        input("\nPress Enter to exit...")
        exit(1)
    
    # Camera settings for normal viewing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)      # Moderate exposure
    
    exposure = -5
    
    print("✓ Camera ready!")
    print("\n" + "="*60)
    print("CONTROLS")
    print("="*60)
    print("  'q' - Quit")
    print("  'b' - Brighter")
    print("  'd' - Darker")
    print("  'r' - Reset brightness")
    print("  's' - Take screenshot")
    print("\n" + "="*60)
    print("FEATURES")
    print("="*60)
    print("  ✓ Pupil detection (large green circle)")
    print("  ✓ Eye landmarks (GazeTracking's visualization)")
    print("  ✓ Gaze direction (LEFT/CENTER/RIGHT)")
    print("  ✓ Blink detection")
    print("  ✓ Horizontal ratio (0.0-1.0)")
    print("="*60)
    print("\nStarting in 2 seconds...")
    
    import time
    time.sleep(2)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Get annotated frame from GazeTracking
        annotated = wrapper.get_annotated_frame(frame)
        
        # Get pupil position and draw LARGE circle
        pupil = wrapper.detect_pupil(frame)
        if pupil:
            # Large green circle around pupil
            cv2.circle(annotated, pupil, 15, (0, 255, 0), 3)
            # Center dot
            cv2.circle(annotated, pupil, 3, (0, 0, 255), -1)
            
            # Show coordinates
            cv2.putText(
                annotated, f"Pupil: ({pupil[0]}, {pupil[1]})",
                (10, annotated.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        # Get direction - BIG TEXT
        direction = wrapper.get_direction(frame)
        if direction:
            text = f"LOOKING {direction.upper()}"
            color = (0, 255, 0)
            status = "TRACKING"
        else:
            text = "NO DETECTION"
            color = (0, 0, 255)
            status = "LOST"
        
        # Draw background rectangle for text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.rectangle(annotated, (5, 5), (text_size[0] + 15, 50), (0, 0, 0), -1)
        
        cv2.putText(
            annotated, text, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
        )
        
        # Check blink - LARGE WARNING
        if wrapper.is_blinking(frame):
            cv2.putText(
                annotated, "⚠ BLINKING", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3
            )
        
        # Get horizontal ratio with visual bar
        ratio = wrapper.get_horizontal_ratio(frame)
        if ratio is not None:
            # Draw ratio bar
            bar_width = 300
            bar_height = 30
            bar_x = annotated.shape[1] - bar_width - 20
            bar_y = 20
            
            # Background
            cv2.rectangle(annotated, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Ratio indicator
            indicator_x = int(bar_x + ratio * bar_width)
            cv2.rectangle(annotated, (bar_x, bar_y),
                         (indicator_x, bar_y + bar_height),
                         (0, 255, 0), -1)
            
            # Center marker
            center_x = bar_x + bar_width // 2
            cv2.line(annotated, (center_x, bar_y), 
                    (center_x, bar_y + bar_height), (255, 255, 255), 2)
            
            # Labels
            cv2.putText(annotated, "LEFT", (bar_x, bar_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(annotated, "CENTER", (center_x - 20, bar_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(annotated, "RIGHT", (bar_x + bar_width - 35, bar_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.putText(
                annotated, f"Ratio: {ratio:.3f}", (bar_x, bar_y + bar_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
        
        # Status bar at bottom
        status_bg = annotated.copy()
        cv2.rectangle(status_bg, (0, annotated.shape[0] - 60),
                     (annotated.shape[1], annotated.shape[0]), (0, 0, 0), -1)
        annotated = cv2.addWeighted(annotated, 0.7, status_bg, 0.3, 0)
        
        cv2.putText(
            annotated, f"Status: {status} | Exposure: {exposure} | Frame: {frame_count}",
            (10, annotated.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        cv2.putText(
            annotated, "Press 'b' for brighter | 'd' for darker | 'q' to quit",
            (10, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        cv2.imshow('GazeTracking Test - WINNER! 🏆', annotated)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n✓ Quitting...")
            break
        elif key == ord('b'):
            # Brighter
            exposure = min(0, exposure + 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            print(f"Exposure: {exposure} (brighter)")
        elif key == ord('d'):
            # Darker
            exposure = max(-13, exposure - 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            print(f"Exposure: {exposure} (darker)")
        elif key == ord('r'):
            # Reset
            exposure = -5
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            print("Reset to default settings")
        elif key == ord('s'):
            # Screenshot
            filename = f"gazetracking_screenshot_{frame_count}.png"
            cv2.imwrite(filename, annotated)
            print(f"✓ Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print("\nIf this worked well for you:")
    print("  ✓ Use it in Mode 6 as Method #7")
    print("  ✓ GazeTracking is production-ready!")
    print("  ✓ Consider this the winner! 🏆")
    print("="*60)

