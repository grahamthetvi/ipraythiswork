"""
Visual Calibration Tool with GUI

Provides visual feedback during calibration:
- Shows calibration points on screen
- Displays camera feed with detected pupils
- Real-time progress indicators
- Audio feedback
"""

import cv2
import numpy as np
import time
import pyttsx3
from typing import List, Tuple, Optional
from pathlib import Path

from eye_tracker_framework import EyeTrackerBase


class VisualCalibrationTool:
    """Interactive visual calibration with GUI feedback."""

    def __init__(self, tracker: EyeTrackerBase, screen_width: int, screen_height: int):
        """
        Initialize calibration tool.

        Args:
            tracker: Eye tracker to calibrate (must be started)
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.tracker = tracker
        self.screen_width = screen_width
        self.screen_height = screen_height

        # TTS for audio feedback
        try:
            self.tts = pyttsx3.init()
            self.tts.setProperty('rate', 150)
        except:
            self.tts = None
            print("Warning: TTS not available")

        # Colors (BGR)
        self.COLOR_BG = (0, 0, 0)  # Black
        self.COLOR_TARGET = (0, 0, 255)  # Red
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.COLOR_PROGRESS_BAR = (0, 255, 0)  # Green
        self.COLOR_PUPIL = (0, 255, 0)  # Green

    def speak(self, text: str):
        """Speak text via TTS."""
        if self.tts:
            try:
                self.tts.say(text)
                self.tts.runAndWait()
            except:
                pass

    def generate_calibration_points(self, num_points: int = 9) -> List[Tuple[int, int]]:
        """
        Generate calibration point positions.

        Args:
            num_points: 5 (CVI-friendly) or 9 (standard)

        Returns:
            List of (x, y) screen coordinates
        """
        if num_points == 5:
            # 5-point pattern
            positions = []
            for row, col in [(0.15, 0.15), (0.15, 0.85), (0.5, 0.5), (0.85, 0.15), (0.85, 0.85)]:
                x = int(col * self.screen_width)
                y = int(row * self.screen_height)
                positions.append((x, y))
            return positions

        elif num_points == 9:
            # 3x3 grid
            positions = []
            for row in [0.1, 0.5, 0.9]:
                for col in [0.1, 0.5, 0.9]:
                    x = int(col * self.screen_width)
                    y = int(row * self.screen_height)
                    positions.append((x, y))
            return positions

        else:
            raise ValueError(f"Unsupported number of points: {num_points}")

    def calibrate_with_visual_feedback(self, num_points: int = 9, samples_per_point: int = 40) -> bool:
        """
        Run calibration with full visual GUI.

        Args:
            num_points: Number of calibration points
            samples_per_point: Samples to collect per point

        Returns:
            True if calibration successful
        """
        print("\n" + "="*60)
        print("VISUAL CALIBRATION")
        print("="*60)
        print(f"Points: {num_points}")
        print(f"Samples per point: {samples_per_point}")
        print("\nInstructions:")
        print("  1. Look at the RED target when it appears")
        print("  2. Keep your head still")
        print("  3. The camera preview shows your pupil detection")
        print("  4. Press ESC to cancel")
        print("="*60)

        self.speak("Starting calibration. Look at each red target.")

        # Generate calibration points
        cal_points = self.generate_calibration_points(num_points)

        # Create windows
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.namedWindow('Pupil Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pupil Detection', 640, 480)

        # Show instructions screen
        self._show_instructions(5)  # 5 second countdown

        # Calibrate each point
        calibration_data = []

        for i, (target_x, target_y) in enumerate(cal_points):
            print(f"\nCalibration point {i+1}/{len(cal_points)}: ({target_x}, {target_y})")
            self.speak(f"Look at target {i+1}")

            # Collect samples for this point
            samples = []
            start_time = time.time()

            while len(samples) < samples_per_point:
                # Get camera frame with pupil drawn
                if hasattr(self.tracker, 'get_frame_with_pupil'):
                    camera_frame = self.tracker.get_frame_with_pupil()
                    if camera_frame is None:
                        camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(camera_frame, "Camera preview not available",
                                  (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(camera_frame, "Camera preview not available",
                              (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Get gaze data
                gaze = self.tracker.get_gaze_point()

                # Collect sample if pupil detected
                if gaze and gaze.pupil_left:
                    samples.append(gaze)

                # Status on camera frame
                cv2.putText(camera_frame, f"Point {i+1}/{len(cal_points)} | Samples: {len(samples)}/{samples_per_point}",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)

                cv2.imshow('Pupil Detection', camera_frame)

                # Draw calibration target screen
                target_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

                # Draw completed points
                for j in range(i):
                    cv2.circle(target_frame, cal_points[j], 15, self.COLOR_TEXT, 2)

                # Draw current target (pulsing)
                pulse = int(20 + 15 * abs(np.sin(time.time() * 4)))
                cv2.circle(target_frame, (target_x, target_y), 40 + pulse, self.COLOR_TARGET, 4)
                cv2.circle(target_frame, (target_x, target_y), 15, self.COLOR_TARGET, -1)

                # Progress bar
                progress = len(samples) / samples_per_point
                bar_width = 500
                bar_height = 40
                bar_x = (self.screen_width - bar_width) // 2
                bar_y = self.screen_height - 120

                # Background
                cv2.rectangle(target_frame, (bar_x, bar_y),
                            (bar_x + bar_width, bar_y + bar_height),
                            self.COLOR_TEXT, 2)

                # Filled portion
                fill_width = int(bar_width * progress)
                cv2.rectangle(target_frame, (bar_x, bar_y),
                            (bar_x + fill_width, bar_y + bar_height),
                            self.COLOR_PROGRESS_BAR, -1)

                # Text
                text = f"Point {i+1}/{len(cal_points)} | Samples: {len(samples)}/{samples_per_point}"
                cv2.putText(target_frame, text, (bar_x, bar_y - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_TEXT, 2)

                # Instructions
                cv2.putText(target_frame, "Look at the red target",
                          (bar_x, bar_y + bar_height + 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)

                cv2.imshow('Calibration', target_frame)

                # Check for ESC key
                key = cv2.waitKey(33)
                if key == 27:  # ESC
                    print("\nCalibration cancelled by user")
                    cv2.destroyAllWindows()
                    return False

                # Timeout after 15 seconds
                if time.time() - start_time > 15:
                    print(f"  ⚠ Timeout (only collected {len(samples)} samples)")
                    break

            # Store calibration data
            calibration_data.append({
                'screen_x': target_x,
                'screen_y': target_y,
                'samples': samples
            })

            print(f"  ✓ Collected {len(samples)} samples")

            # Brief pause
            time.sleep(0.3)

        cv2.destroyAllWindows()

        # Now pass data to tracker's calibration method
        print("\nComputing calibration matrix...")

        # Convert to format expected by tracker
        cal_points_list = [(d['screen_x'], d['screen_y']) for d in calibration_data]

        # Manually set calibration data if tracker supports it
        if hasattr(self.tracker, 'calibration_points'):
            from eye_tracker_framework import CalibrationPoint

            self.tracker.calibration_points = []
            for data in calibration_data:
                cal_point = CalibrationPoint(
                    screen_x=data['screen_x'],
                    screen_y=data['screen_y'],
                    gaze_samples=data['samples']
                )
                self.tracker.calibration_points.append(cal_point)

            # Compute matrix
            if hasattr(self.tracker, '_compute_calibration_matrix'):
                if self.tracker._compute_calibration_matrix():
                    self.tracker.is_calibrated = True
                    print("✓ Calibration successful!")
                    self.speak("Calibration complete")
                    return True
                else:
                    print("✗ Failed to compute calibration matrix")
                    return False
        else:
            # Use tracker's own calibrate method
            return self.tracker.calibrate(cal_points_list)

    def _show_instructions(self, countdown_seconds: int = 5):
        """Show instruction screen with countdown."""
        for remaining in range(countdown_seconds, 0, -1):
            frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

            # Title
            cv2.putText(frame, "CALIBRATION STARTING", (self.screen_width // 2 - 250, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.COLOR_TEXT, 3)

            # Instructions
            instructions = [
                "Instructions:",
                "",
                "1. Look directly at each RED target",
                "2. Keep your head still during calibration",
                "3. Focus on the center of each target",
                "4. The camera preview shows pupil detection",
                "",
                f"Starting in {remaining} seconds..."
            ]

            y = 300
            for line in instructions:
                cv2.putText(frame, line, (self.screen_width // 2 - 300, y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_TEXT, 2)
                y += 50

            cv2.imshow('Calibration', frame)
            cv2.waitKey(1000)


def main():
    """Interactive calibration tool."""
    print("\n" + "="*60)
    print("VISUAL CALIBRATION TOOL")
    print("="*60)

    # Get configuration
    screen_width = int(input("\nScreen width (pixels, default 1920): ") or "1920")
    screen_height = int(input("Screen height (pixels, default 1080): ") or "1080")

    print("\nCalibration patterns:")
    print("  5 points - CVI-friendly (faster)")
    print("  9 points - Standard (more accurate)")
    num_points = int(input("Number of points (5/9, default 9): ") or "9")

    print("\nWhich tracker to calibrate?")
    print("  1. Hough Circle (webcam/IR camera)")
    print("  2. WebGazer (browser-based)")
    tracker_choice = input("Select (1-2): ")

    if tracker_choice == "1":
        from trackers.hough_tracker_gaze import HoughGazeTracker

        camera_index = int(input("\nCamera index (0=default, 1=second): ") or "0")

        tracker = HoughGazeTracker(screen_width, screen_height)

        if not tracker.start(camera_index):
            print("Failed to start tracker")
            return

        print("\n✓ Tracker started")
        print("\nStarting visual calibration in 3 seconds...")
        time.sleep(3)

        # Run calibration with GUI
        calibrator = VisualCalibrationTool(tracker, screen_width, screen_height)

        if calibrator.calibrate_with_visual_feedback(num_points):
            # Save calibration
            save_choice = input("\nSave calibration? (y/n): ")
            if save_choice.lower() == 'y':
                filename = input("Filename (default: calibration.json): ") or "calibration.json"
                if hasattr(tracker, 'save_calibration'):
                    tracker.save_calibration(filename)
                    print(f"✓ Saved to {filename}")

        tracker.stop()

    elif tracker_choice == "2":
        print("\nWebGazer calibration happens in the browser.")
        print("Open http://localhost:5000 to calibrate WebGazer.")

    else:
        print("Invalid selection")


if __name__ == "__main__":
    main()
