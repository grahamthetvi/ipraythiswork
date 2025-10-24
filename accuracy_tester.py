"""
Eye Tracking Accuracy Testing Tool

Displays visual targets at precise screen coordinates and measures
how accurately each eye tracking system can pinpoint gaze location.

This is the core validation tool for comparing different eye trackers.
"""

import cv2
import numpy as np
import time
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import pyttsx3

from eye_tracker_framework import EyeTrackerBase, GazePoint


@dataclass
class AccuracyTest:
    """Results from a single accuracy test."""
    tracker_name: str
    target_x: int
    target_y: int
    gaze_samples: List[GazePoint] = field(default_factory=list)

    def get_average_gaze(self) -> Optional[Tuple[float, float]]:
        """Calculate average gaze position."""
        if not self.gaze_samples:
            return None

        avg_x = np.mean([s.x for s in self.gaze_samples])
        avg_y = np.mean([s.y for s in self.gaze_samples])
        return (avg_x, avg_y)

    def get_error_distance(self) -> Optional[float]:
        """Calculate Euclidean distance from target to average gaze."""
        avg_gaze = self.get_average_gaze()
        if not avg_gaze:
            return None

        dx = avg_gaze[0] - self.target_x
        dy = avg_gaze[1] - self.target_y
        return np.sqrt(dx**2 + dy**2)

    def get_accuracy_percentage(self, tolerance_px: int = 100) -> float:
        """
        Get accuracy as percentage within tolerance.

        Args:
            tolerance_px: Distance in pixels considered "accurate"

        Returns:
            Percentage of samples within tolerance
        """
        if not self.gaze_samples:
            return 0.0

        accurate_samples = 0
        for sample in self.gaze_samples:
            dx = sample.x - self.target_x
            dy = sample.y - self.target_y
            distance = np.sqrt(dx**2 + dy**2)
            if distance <= tolerance_px:
                accurate_samples += 1

        return (accurate_samples / len(self.gaze_samples)) * 100


class AccuracyTester:
    """
    Visual accuracy testing tool for eye trackers.

    Displays targets at known coordinates and measures tracker accuracy.
    """

    def __init__(self, screen_width: int, screen_height: int):
        """Initialize accuracy tester."""
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Test configuration
        self.samples_per_target = 30  # Number of gaze samples to collect
        self.target_radius = 30  # Target circle radius
        self.test_results: List[AccuracyTest] = []

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
        self.COLOR_TARGET_DONE = (255, 255, 255)  # White
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.COLOR_GAZE = (0, 255, 0)  # Green

    def speak(self, text: str):
        """Speak text via TTS."""
        if self.tts:
            try:
                self.tts.say(text)
                self.tts.runAndWait()
            except:
                pass

    def generate_test_points(self, num_points: int = 9) -> List[Tuple[int, int]]:
        """
        Generate test target positions.

        Args:
            num_points: Number of test points (9 or 13 recommended)

        Returns:
            List of (x, y) screen coordinates
        """
        if num_points == 9:
            # 3x3 grid
            positions = []
            for row in [0.1, 0.5, 0.9]:
                for col in [0.1, 0.5, 0.9]:
                    x = int(col * self.screen_width)
                    y = int(row * self.screen_height)
                    positions.append((x, y))
            return positions

        elif num_points == 13:
            # 9-point grid + 4 intermediate points
            positions = []
            # Main 9 points
            for row in [0.1, 0.5, 0.9]:
                for col in [0.1, 0.5, 0.9]:
                    x = int(col * self.screen_width)
                    y = int(row * self.screen_height)
                    positions.append((x, y))
            # Additional 4 points
            for row, col in [(0.3, 0.3), (0.3, 0.7), (0.7, 0.3), (0.7, 0.7)]:
                x = int(col * self.screen_width)
                y = int(row * self.screen_height)
                positions.append((x, y))
            return positions

        elif num_points == 5:
            # 5-point (CVI-friendly)
            positions = []
            for row, col in [(0.15, 0.15), (0.15, 0.85), (0.5, 0.5), (0.85, 0.15), (0.85, 0.85)]:
                x = int(col * self.screen_width)
                y = int(row * self.screen_height)
                positions.append((x, y))
            return positions

        else:
            raise ValueError(f"Unsupported number of points: {num_points}")

    def run_accuracy_test(self, tracker: EyeTrackerBase, num_points: int = 9) -> List[AccuracyTest]:
        """
        Run accuracy test on a tracker.

        Args:
            tracker: Eye tracker instance (must be started and calibrated)
            num_points: Number of test points

        Returns:
            List of AccuracyTest results
        """
        print("\n" + "="*60)
        print("ACCURACY TEST")
        print("="*60)
        print(f"Tracker: {tracker.tracker_name}")
        print(f"Test points: {num_points}")
        print(f"Samples per point: {self.samples_per_target}")
        print("="*60)

        if not tracker.is_calibrated:
            print("\n⚠ WARNING: Tracker not calibrated!")
            print("Results may be inaccurate without calibration.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return []

        self.speak(f"Starting accuracy test for {tracker.tracker_name}")

        # Generate test points
        test_points = self.generate_test_points(num_points)

        # Create display window
        cv2.namedWindow('Accuracy Test', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Accuracy Test', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        results = []

        # Test each point
        for i, (target_x, target_y) in enumerate(test_points):
            print(f"\nTest point {i+1}/{len(test_points)}: ({target_x}, {target_y})")
            self.speak(f"Look at target {i+1}")

            test = AccuracyTest(
                tracker_name=tracker.tracker_name,
                target_x=target_x,
                target_y=target_y
            )

            # Show target and collect samples
            samples_collected = 0
            start_time = time.time()

            while samples_collected < self.samples_per_target:
                # Create frame
                frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

                # Draw completed targets
                for prev_result in results:
                    cv2.circle(frame, (prev_result.target_x, prev_result.target_y),
                             15, self.COLOR_TARGET_DONE, 2)

                # Draw current target (pulsing)
                pulse = int(20 + 10 * abs(np.sin(time.time() * 4)))
                cv2.circle(frame, (target_x, target_y),
                         self.target_radius + pulse, self.COLOR_TARGET, 3)
                cv2.circle(frame, (target_x, target_y), 10, self.COLOR_TARGET, -1)

                # Get gaze data
                gaze = tracker.get_gaze_point()
                if gaze:
                    # Draw gaze point
                    cv2.circle(frame, (int(gaze.x), int(gaze.y)), 10, self.COLOR_GAZE, -1)

                    # Collect sample
                    test.gaze_samples.append(gaze)
                    samples_collected += 1

                # Progress bar
                progress = samples_collected / self.samples_per_target
                bar_width = 400
                bar_height = 30
                bar_x = (self.screen_width - bar_width) // 2
                bar_y = self.screen_height - 100

                cv2.rectangle(frame, (bar_x, bar_y),
                            (bar_x + bar_width, bar_y + bar_height),
                            self.COLOR_TEXT, 2)
                cv2.rectangle(frame, (bar_x, bar_y),
                            (bar_x + int(bar_width * progress), bar_y + bar_height),
                            self.COLOR_TARGET, -1)

                # Instructions
                text = f"Look at the red target | Point {i+1}/{len(test_points)} | Samples: {samples_collected}/{self.samples_per_target}"
                cv2.putText(frame, text, (bar_x, bar_y - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)

                cv2.imshow('Accuracy Test', frame)

                key = cv2.waitKey(33)
                if key == 27:  # ESC
                    print("\nTest cancelled")
                    cv2.destroyAllWindows()
                    return results

                # Timeout after 10 seconds
                if time.time() - start_time > 10:
                    print(f"  ⚠ Timeout (only collected {samples_collected} samples)")
                    break

            # Calculate results for this point
            avg_gaze = test.get_average_gaze()
            error = test.get_error_distance()
            accuracy = test.get_accuracy_percentage(tolerance_px=100)

            if avg_gaze and error is not None:
                print(f"  Average gaze: ({avg_gaze[0]:.1f}, {avg_gaze[1]:.1f})")
                print(f"  Error distance: {error:.1f} pixels")
                print(f"  Accuracy (±100px): {accuracy:.1f}%")
            else:
                print(f"  ✗ No valid gaze data collected")

            results.append(test)

            # Brief pause between points
            time.sleep(0.5)

        cv2.destroyAllWindows()

        # Overall statistics
        self._print_summary(results)
        self.test_results.extend(results)

        return results

    def _print_summary(self, results: List[AccuracyTest]):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        errors = [r.get_error_distance() for r in results if r.get_error_distance() is not None]
        accuracies = [r.get_accuracy_percentage() for r in results]

        if errors:
            print(f"\nError Distance (pixels):")
            print(f"  Mean:   {np.mean(errors):.1f}")
            print(f"  Median: {np.median(errors):.1f}")
            print(f"  Std:    {np.std(errors):.1f}")
            print(f"  Min:    {np.min(errors):.1f}")
            print(f"  Max:    {np.max(errors):.1f}")

        if accuracies:
            print(f"\nAccuracy (±100px):")
            print(f"  Mean:   {np.mean(accuracies):.1f}%")
            print(f"  Median: {np.median(accuracies):.1f}%")

        # Success rate
        valid_results = len([r for r in results if r.get_error_distance() is not None])
        print(f"\nValid measurements: {valid_results}/{len(results)} ({100*valid_results/len(results):.1f}%)")

        print("="*60)

    def save_results(self, filename: str):
        """
        Save test results to CSV.

        Args:
            filename: Output CSV filename
        """
        if not self.test_results:
            print("No results to save")
            return

        filepath = Path(filename)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Tracker', 'Target_X', 'Target_Y', 'Avg_Gaze_X', 'Avg_Gaze_Y',
                'Error_Distance', 'Accuracy_100px', 'Samples_Collected'
            ])

            # Data
            for result in self.test_results:
                avg_gaze = result.get_average_gaze()
                error = result.get_error_distance()
                accuracy = result.get_accuracy_percentage(tolerance_px=100)

                if avg_gaze:
                    writer.writerow([
                        result.tracker_name,
                        result.target_x,
                        result.target_y,
                        f"{avg_gaze[0]:.2f}",
                        f"{avg_gaze[1]:.2f}",
                        f"{error:.2f}" if error else "N/A",
                        f"{accuracy:.2f}",
                        len(result.gaze_samples)
                    ])
                else:
                    writer.writerow([
                        result.tracker_name,
                        result.target_x,
                        result.target_y,
                        "N/A", "N/A", "N/A", "0.00", len(result.gaze_samples)
                    ])

        print(f"\n✓ Results saved to: {filepath}")

    def visualize_results(self, results: List[AccuracyTest]):
        """
        Visualize accuracy test results.

        Shows targets and measured gaze positions with error vectors.
        """
        # Create visualization frame
        vis = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        for result in results:
            target_x, target_y = result.target_x, result.target_y
            avg_gaze = result.get_average_gaze()

            # Draw target
            cv2.circle(vis, (target_x, target_y), 20, self.COLOR_TARGET, 2)
            cv2.circle(vis, (target_x, target_y), 5, self.COLOR_TARGET, -1)

            if avg_gaze:
                gaze_x, gaze_y = int(avg_gaze[0]), int(avg_gaze[1])

                # Draw measured gaze position
                cv2.circle(vis, (gaze_x, gaze_y), 15, self.COLOR_GAZE, 2)

                # Draw error vector
                cv2.arrowedLine(vis, (target_x, target_y), (gaze_x, gaze_y),
                              (0, 255, 255), 2, tipLength=0.3)

                # Show error distance
                error = result.get_error_distance()
                if error:
                    mid_x = (target_x + gaze_x) // 2
                    mid_y = (target_y + gaze_y) // 2
                    cv2.putText(vis, f"{error:.0f}px", (mid_x + 10, mid_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Legend
        cv2.putText(vis, "Red = Target | Green = Measured Gaze | Yellow = Error",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)

        cv2.imshow('Accuracy Visualization', vis)
        print("\nVisualization shown. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Interactive accuracy testing."""
    print("\n" + "="*60)
    print("EYE TRACKING ACCURACY TESTER")
    print("="*60)
    print("\nThis tool tests eye tracker accuracy by showing visual targets")
    print("at precise screen coordinates and measuring gaze position.")
    print("="*60)

    # Get screen dimensions
    screen_width = int(input("\nScreen width (pixels, default 1920): ") or "1920")
    screen_height = int(input("Screen height (pixels, default 1080): ") or "1080")

    # Get test configuration
    print("\nTest configurations:")
    print("  5 points - CVI-friendly (faster)")
    print("  9 points - Standard (3x3 grid)")
    print("  13 points - Detailed (9 + 4 intermediate)")
    num_points = int(input("Number of test points (5/9/13, default 9): ") or "9")

    tester = AccuracyTester(screen_width, screen_height)

    # Example: Test with Hough Circle tracker
    print("\n" + "="*60)
    print("TRACKER SELECTION")
    print("="*60)
    print("\nAvailable trackers:")
    print("  1. Hough Circle (baseline, any webcam)")
    print("  2. WebGazer (web-based)")
    print("  3. Pupil Labs (requires hardware)")

    tracker_choice = input("\nSelect tracker (1-3): ")

    if tracker_choice == "1":
        from trackers.hough_tracker_gaze import HoughGazeTracker

        tracker = HoughGazeTracker(screen_width, screen_height)
        camera_index = int(input("Camera index (default 0): ") or "0")

        if not tracker.start(camera_index):
            print("Failed to start tracker")
            return

        # Calibration
        print("\nCalibration required for accuracy testing")
        cal_choice = input("Calibrate now? (y/n): ")

        if cal_choice.lower() == 'y':
            # Generate calibration points
            cal_points = tester.generate_test_points(num_points)
            if not tracker.calibrate(cal_points):
                print("Calibration failed")
                tracker.stop()
                return
        else:
            # Try to load calibration
            cal_file = input("Calibration file path (or blank to skip): ")
            if cal_file and Path(cal_file).exists():
                tracker.load_calibration(cal_file)

        # Run accuracy test
        results = tester.run_accuracy_test(tracker, num_points)

        # Save and visualize
        if results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"accuracy_test_{tracker.tracker_name.replace(' ', '_')}_{timestamp}.csv"
            tester.save_results(output_file)

            vis_choice = input("\nVisualize results? (y/n): ")
            if vis_choice.lower() == 'y':
                tester.visualize_results(results)

        tracker.stop()

    elif tracker_choice == "2":
        print("\nWebGazer integration:")
        print("  1. Start WebGazer tracker")
        print("  2. Calibrate in browser")
        print("  3. Return here and press Enter to begin test")
        print("\n(Full WebGazer test script coming soon)")

    elif tracker_choice == "3":
        print("\nPupil Labs integration:")
        print("  1. Start Pupil Capture software")
        print("  2. Calibrate in Pupil Capture")
        print("  3. Run this test")
        print("\n(Full Pupil Labs test script coming soon)")

    else:
        print("Invalid selection")


if __name__ == "__main__":
    main()
