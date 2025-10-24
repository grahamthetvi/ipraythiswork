"""
Live Pupil Detection Preview

Simple tool to test pupil detection before calibration.
Shows live camera feed with detected pupils highlighted.
"""

import cv2
import sys
from pathlib import Path

# Add trackers directory to path
sys.path.append(str(Path(__file__).parent))
from trackers.hough_tracker_gaze import HoughGazeTracker


def main():
    """Live preview of pupil detection."""
    print("\n" + "="*60)
    print("PUPIL DETECTION LIVE PREVIEW")
    print("="*60)
    print("\nThis tool shows live camera feed with detected pupils.")
    print("Use this to test your camera and lighting before calibration.")
    print("="*60)

    # Get camera index
    camera_index = int(input("\nCamera index (0=default, 1=second camera): ") or "0")

    # Start tracker
    print(f"\nStarting camera {camera_index}...")
    tracker = HoughGazeTracker(screen_width=1920, screen_height=1080)

    if not tracker.start(camera_index):
        print("✗ Failed to start camera")
        return

    print("✓ Camera started")
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  ESC - Exit")
    print("  + - Increase pupil detection sensitivity")
    print("  - - Decrease pupil detection sensitivity")
    print("  R - Reset parameters to default")
    print("="*60)

    cv2.namedWindow('Pupil Detection Preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pupil Detection Preview', 960, 720)

    print("\nLive preview running. Adjust lighting for best results.")
    print("Look for GREEN circle and crosshair on your pupil.\n")

    frame_count = 0
    detection_count = 0

    while True:
        # Get frame with pupil visualization
        frame = tracker.get_frame_with_pupil()

        if frame is None:
            print("✗ Failed to get frame")
            break

        # Get detection status
        gaze = tracker.get_gaze_point()
        if gaze and gaze.pupil_left:
            detection_count += 1

        frame_count += 1

        # Show detection rate
        if frame_count > 0:
            detection_rate = (detection_count / frame_count) * 100
            status_text = f"Detection rate: {detection_rate:.1f}%"
            status_color = (0, 255, 0) if detection_rate > 70 else (0, 165, 255) if detection_rate > 40 else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Show current parameters
        cv2.putText(frame, f"param2: {tracker.param2} (sensitivity)",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press +/- to adjust",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show tips
        if detection_rate < 50:
            cv2.putText(frame, "TIP: Adjust lighting or try different camera",
                       (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(frame, "Try pressing + to increase sensitivity",
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.imshow('Pupil Detection Preview', frame)

        # Handle keys
        key = cv2.waitKey(30)

        if key == 27:  # ESC
            break
        elif key == ord('+') or key == ord('='):
            # Increase sensitivity (lower param2)
            tracker.param2 = max(10, tracker.param2 - 2)
            print(f"Sensitivity increased (param2={tracker.param2})")
            # Reset stats
            frame_count = 0
            detection_count = 0
        elif key == ord('-') or key == ord('_'):
            # Decrease sensitivity (higher param2)
            tracker.param2 = min(50, tracker.param2 + 2)
            print(f"Sensitivity decreased (param2={tracker.param2})")
            # Reset stats
            frame_count = 0
            detection_count = 0
        elif key == ord('r') or key == ord('R'):
            # Reset to default
            tracker.param2 = 30
            print("Parameters reset to default")
            frame_count = 0
            detection_count = 0

    # Cleanup
    cv2.destroyAllWindows()
    tracker.stop()

    # Final stats
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    if frame_count > 0:
        final_rate = (detection_count / frame_count) * 100
        print(f"Overall detection rate: {final_rate:.1f}%")
        print(f"Frames processed: {frame_count}")
        print(f"Pupil detected: {detection_count}")

        if final_rate > 80:
            print("\n✓ Excellent! Your camera and lighting are good.")
            print("  You're ready to run calibration.")
        elif final_rate > 60:
            print("\n⚠ Good, but could be better.")
            print("  Consider adjusting lighting or camera angle.")
        else:
            print("\n✗ Poor detection rate.")
            print("  Try:")
            print("    - Different camera (run camera_selector.py)")
            print("    - Better lighting (diffuse, not too bright)")
            print("    - Adjust camera angle")
            print("    - Remove glasses if causing glare")

    print("\nNext step: Run calibrate_with_gui.py to calibrate")
    print("="*60)


if __name__ == "__main__":
    main()
