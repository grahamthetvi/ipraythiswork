"""
Eye Gaze Testing and Validation Tool
For testing 9-region gaze tracker accuracy and performance

Features:
- Accuracy testing mode with hit rate per region
- Real-time visualization of gaze position
- Performance metrics (FPS, accuracy, latency)
- Export test results to CSV
- Confusion matrix for region detection
- Statistical analysis of tracking performance
"""

import cv2
import numpy as np
import csv
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from gaze_9region import Gaze9Region, GazeEvent


@dataclass
class TestResult:
    """Single test result."""
    timestamp: float
    target_region: int
    detected_region: Optional[int]
    confidence: float
    latency_ms: float
    correct: bool


class GazeTester:
    """
    Testing and validation tool for 9-region gaze tracker.
    """

    def __init__(self, tracker: Gaze9Region):
        """
        Initialize gaze tester.

        Args:
            tracker: Gaze9Region tracker instance
        """
        self.tracker = tracker
        self.test_results: List[TestResult] = []
        self.current_test_active = False
        self.target_region = None
        self.test_start_time = None
        self.tests_per_region = 5

        # Statistics
        self.region_stats = {i: {'correct': 0, 'total': 0, 'latencies': []}
                           for i in range(9)}

        # Confusion matrix (actual vs detected)
        self.confusion_matrix = np.zeros((9, 9), dtype=int)

        # Visual settings
        self.screen_width = tracker.screen_width
        self.screen_height = tracker.screen_height

    def start_accuracy_test(self):
        """Start comprehensive accuracy test."""
        print("\n" + "="*60)
        print("ACCURACY TEST MODE")
        print("="*60)
        print(f"\nTests per region: {self.tests_per_region}")
        print(f"Total tests: {self.tests_per_region * 9}")
        print("\nInstructions:")
        print("1. Look at the highlighted region")
        print("2. Hold your gaze steady")
        print("3. System will automatically record results")
        print("4. Press ESC to stop test")
        print("\nPress any key to start...")
        input()

        self.test_results.clear()
        self.region_stats = {i: {'correct': 0, 'total': 0, 'latencies': []}
                           for i in range(9)}
        self.confusion_matrix = np.zeros((9, 9), dtype=int)

        # Generate test sequence (randomized for fairness)
        test_sequence = []
        for region in range(9):
            test_sequence.extend([region] * self.tests_per_region)

        np.random.shuffle(test_sequence)

        # Run tests
        cv2.namedWindow('Accuracy Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Accuracy Test', self.screen_width, self.screen_height)

        test_index = 0
        current_target = None
        target_start_time = None
        target_duration = 3.0  # Seconds to display each target

        while test_index < len(test_sequence):
            ret, frame = self.tracker.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Get current target
            if current_target is None:
                current_target = test_sequence[test_index]
                target_start_time = time.time()
                print(f"\nTest {test_index + 1}/{len(test_sequence)}: "
                      f"Region {current_target}")

            # Detect gaze
            result = self.tracker.detect_region(frame)

            detected_region = None
            confidence = 0.0
            latency = self.tracker.latency_ms

            if result is not None:
                detected_region, confidence = result

            # Draw test display
            display = self._draw_test_display(frame, current_target,
                                             detected_region, confidence,
                                             test_index, len(test_sequence))

            cv2.imshow('Accuracy Test', display)

            # Check if target duration elapsed
            if time.time() - target_start_time >= target_duration:
                # Record result
                is_correct = (detected_region == current_target)

                test_result = TestResult(
                    timestamp=time.time(),
                    target_region=current_target,
                    detected_region=detected_region,
                    confidence=confidence,
                    latency_ms=latency,
                    correct=is_correct
                )

                self.test_results.append(test_result)

                # Update statistics
                self.region_stats[current_target]['total'] += 1
                if is_correct:
                    self.region_stats[current_target]['correct'] += 1
                self.region_stats[current_target]['latencies'].append(latency)

                # Update confusion matrix
                if detected_region is not None:
                    self.confusion_matrix[current_target][detected_region] += 1

                # Move to next test
                test_index += 1
                current_target = None

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cv2.destroyWindow('Accuracy Test')

        # Show results
        self._print_test_results()

    def _draw_test_display(self, frame: np.ndarray, target_region: int,
                          detected_region: Optional[int], confidence: float,
                          test_index: int, total_tests: int) -> np.ndarray:
        """Draw test display with target region highlighted."""
        # Create display
        display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # Draw 9 regions
        region_w = self.screen_width // 3
        region_h = self.screen_height // 3

        for region_id in range(9):
            row = region_id // 3
            col = region_id % 3

            x1 = col * region_w
            y1 = row * region_h
            x2 = x1 + region_w
            y2 = y1 + region_h

            # Color based on target/detected
            if region_id == target_region:
                # Target region (yellow)
                color = (0, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, -1)
                overlay = display.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

            # Border
            border_color = (100, 100, 100)
            border_thickness = 2

            if region_id == detected_region:
                # Show detected region with green border
                border_color = (0, 255, 0)
                border_thickness = 5

            cv2.rectangle(display, (x1, y1), (x2, y2), border_color, border_thickness)

            # Region number
            text = str(region_id)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
            text_x = x1 + (region_w - text_size[0]) // 2
            text_y = y1 + (region_h + text_size[1]) // 2

            cv2.putText(display, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

        # Instructions
        instructions = [
            f"Test {test_index + 1}/{total_tests}",
            f"Look at YELLOW region ({target_region})",
            "Hold your gaze steady",
        ]

        y_offset = 40
        for instruction in instructions:
            cv2.putText(display, instruction, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            y_offset += 40

        # Detected region and confidence
        if detected_region is not None:
            status_text = f"Detected: Region {detected_region} ({confidence:.2f})"
            status_color = (0, 255, 0) if detected_region == target_region else (0, 0, 255)
        else:
            status_text = "Detected: None"
            status_color = (100, 100, 100)

        cv2.putText(display, status_text, (20, self.screen_height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        return display

    def _print_test_results(self):
        """Print comprehensive test results."""
        if len(self.test_results) == 0:
            print("\nNo test results to display")
            return

        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)

        # Overall accuracy
        total_tests = len(self.test_results)
        correct_tests = sum(1 for r in self.test_results if r.correct)
        accuracy = (correct_tests / total_tests) * 100

        print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct_tests}/{total_tests})")

        # Per-region statistics
        print("\n" + "-"*60)
        print("PER-REGION ACCURACY")
        print("-"*60)
        print(f"{'Region':<10} {'Accuracy':<15} {'Tests':<10} {'Avg Latency':<15}")
        print("-"*60)

        for region_id in range(9):
            stats = self.region_stats[region_id]
            if stats['total'] > 0:
                region_accuracy = (stats['correct'] / stats['total']) * 100
                avg_latency = np.mean(stats['latencies']) if stats['latencies'] else 0
            else:
                region_accuracy = 0
                avg_latency = 0

            print(f"{region_id:<10} {region_accuracy:>6.1f}%{'':<8} "
                  f"{stats['total']:<10} {avg_latency:>8.1f}ms")

        # Performance metrics
        print("\n" + "-"*60)
        print("PERFORMANCE METRICS")
        print("-"*60)

        all_latencies = [r.latency_ms for r in self.test_results if r.latency_ms > 0]
        if all_latencies:
            print(f"Average Latency: {np.mean(all_latencies):.1f}ms")
            print(f"Min Latency: {np.min(all_latencies):.1f}ms")
            print(f"Max Latency: {np.max(all_latencies):.1f}ms")

        # Confidence statistics
        all_confidences = [r.confidence for r in self.test_results if r.detected_region is not None]
        if all_confidences:
            print(f"\nAverage Confidence: {np.mean(all_confidences):.2f}")
            print(f"Min Confidence: {np.min(all_confidences):.2f}")
            print(f"Max Confidence: {np.max(all_confidences):.2f}")

        # Detection rate
        detected_tests = sum(1 for r in self.test_results if r.detected_region is not None)
        detection_rate = (detected_tests / total_tests) * 100
        print(f"\nDetection Rate: {detection_rate:.1f}% ({detected_tests}/{total_tests})")

        # Confusion matrix
        print("\n" + "-"*60)
        print("CONFUSION MATRIX (Target vs Detected)")
        print("-"*60)
        print("Target\\Detected", end="")
        for i in range(9):
            print(f"{i:>5}", end="")
        print()
        print("-"*60)

        for target in range(9):
            print(f"{target:>15}", end="")
            for detected in range(9):
                count = self.confusion_matrix[target][detected]
                if count > 0:
                    print(f"{count:>5}", end="")
                else:
                    print(f"{'':>5}", end="")
            print()

        # Quality assessment
        print("\n" + "="*60)
        print("QUALITY ASSESSMENT")
        print("="*60)

        if accuracy >= 85:
            print("✓ EXCELLENT - Meets production requirements (≥85%)")
        elif accuracy >= 75:
            print("⚠ GOOD - Close to target, minor calibration may help")
        elif accuracy >= 60:
            print("⚠ FAIR - Recalibration recommended")
        else:
            print("✗ POOR - Recalibration required or check camera setup")

        avg_latency = np.mean(all_latencies) if all_latencies else 999

        if avg_latency < 100:
            print("✓ EXCELLENT - Latency meets requirement (<100ms)")
        elif avg_latency < 150:
            print("⚠ GOOD - Acceptable latency")
        else:
            print("✗ POOR - Latency too high, check system performance")

    def export_results_csv(self, filename: Optional[str] = None):
        """
        Export test results to CSV file.

        Args:
            filename: Output filename (auto-generated if None)
        """
        if len(self.test_results) == 0:
            print("No results to export")
            return

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gaze_test_results_{timestamp}.csv"

        # Write CSV
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Header
                writer.writerow([
                    'Timestamp', 'Target Region', 'Detected Region',
                    'Confidence', 'Latency (ms)', 'Correct'
                ])

                # Data
                for result in self.test_results:
                    writer.writerow([
                        datetime.fromtimestamp(result.timestamp).isoformat(),
                        result.target_region,
                        result.detected_region if result.detected_region is not None else 'None',
                        f"{result.confidence:.3f}",
                        f"{result.latency_ms:.1f}",
                        result.correct
                    ])

            print(f"\n✓ Results exported to: {filename}")

        except Exception as e:
            print(f"✗ Error exporting results: {e}")

    def export_summary_csv(self, filename: Optional[str] = None):
        """
        Export summary statistics to CSV.

        Args:
            filename: Output filename (auto-generated if None)
        """
        if len(self.test_results) == 0:
            print("No results to export")
            return

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gaze_test_summary_{timestamp}.csv"

        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Overall statistics
                writer.writerow(['OVERALL STATISTICS'])
                writer.writerow(['Metric', 'Value'])

                total_tests = len(self.test_results)
                correct_tests = sum(1 for r in self.test_results if r.correct)
                accuracy = (correct_tests / total_tests) * 100

                writer.writerow(['Total Tests', total_tests])
                writer.writerow(['Correct Tests', correct_tests])
                writer.writerow(['Overall Accuracy (%)', f"{accuracy:.1f}"])

                all_latencies = [r.latency_ms for r in self.test_results if r.latency_ms > 0]
                if all_latencies:
                    writer.writerow(['Average Latency (ms)', f"{np.mean(all_latencies):.1f}"])
                    writer.writerow(['Min Latency (ms)', f"{np.min(all_latencies):.1f}"])
                    writer.writerow(['Max Latency (ms)', f"{np.max(all_latencies):.1f}"])

                # Per-region statistics
                writer.writerow([])
                writer.writerow(['PER-REGION STATISTICS'])
                writer.writerow(['Region', 'Accuracy (%)', 'Correct', 'Total', 'Avg Latency (ms)'])

                for region_id in range(9):
                    stats = self.region_stats[region_id]
                    if stats['total'] > 0:
                        region_accuracy = (stats['correct'] / stats['total']) * 100
                        avg_latency = np.mean(stats['latencies']) if stats['latencies'] else 0
                    else:
                        region_accuracy = 0
                        avg_latency = 0

                    writer.writerow([
                        region_id,
                        f"{region_accuracy:.1f}",
                        stats['correct'],
                        stats['total'],
                        f"{avg_latency:.1f}"
                    ])

            print(f"✓ Summary exported to: {filename}")

        except Exception as e:
            print(f"✗ Error exporting summary: {e}")

    def run_performance_benchmark(self, duration: int = 60):
        """
        Run performance benchmark test.

        Args:
            duration: Test duration in seconds
        """
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)
        print(f"\nDuration: {duration} seconds")
        print("Testing FPS, latency, and tracking stability")
        print("\nPress any key to start...")
        input()

        start_time = time.time()
        frame_times = []
        latencies = []
        detection_count = 0
        total_frames = 0

        cv2.namedWindow('Performance Benchmark', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Performance Benchmark', self.screen_width, self.screen_height)

        while time.time() - start_time < duration:
            frame_start = time.time()

            ret, frame = self.tracker.cap.read()
            if not ret:
                break

            total_frames += 1

            # Detect region
            result = self.tracker.detect_region(frame)

            if result is not None:
                detection_count += 1
                latencies.append(self.tracker.latency_ms)

            # Draw visualization
            region_id = result[0] if result else None
            confidence = result[1] if result else 0.0
            display = self.tracker.draw_visualization(frame, region_id, confidence)

            # Add benchmark info
            elapsed = time.time() - start_time
            remaining = duration - elapsed

            cv2.putText(display, f"Benchmark: {remaining:.1f}s remaining",
                       (20, self.screen_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            cv2.imshow('Performance Benchmark', display)

            # Record frame time
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyWindow('Performance Benchmark')

        # Calculate statistics
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)

        avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
        min_fps = 1.0 / np.max(frame_times) if frame_times else 0
        max_fps = 1.0 / np.min(frame_times) if frame_times else 0

        print(f"\nFrame Rate:")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Min FPS: {min_fps:.1f}")
        print(f"  Max FPS: {max_fps:.1f}")

        if latencies:
            print(f"\nLatency:")
            print(f"  Average: {np.mean(latencies):.1f}ms")
            print(f"  Min: {np.min(latencies):.1f}ms")
            print(f"  Max: {np.max(latencies):.1f}ms")

        detection_rate = (detection_count / total_frames) * 100 if total_frames > 0 else 0
        print(f"\nDetection Rate: {detection_rate:.1f}%")

        # Assessment
        print("\n" + "="*60)
        print("PERFORMANCE ASSESSMENT")
        print("="*60)

        if avg_fps >= 30:
            print("✓ FPS: EXCELLENT (≥30 FPS)")
        elif avg_fps >= 20:
            print("⚠ FPS: ACCEPTABLE (≥20 FPS)")
        else:
            print("✗ FPS: POOR (<20 FPS)")

        avg_latency = np.mean(latencies) if latencies else 999
        if avg_latency < 100:
            print("✓ Latency: EXCELLENT (<100ms)")
        elif avg_latency < 150:
            print("⚠ Latency: ACCEPTABLE (<150ms)")
        else:
            print("✗ Latency: POOR (≥150ms)")


def main():
    """Main test interface."""
    print("="*60)
    print("GAZE TRACKER TESTING TOOL")
    print("="*60)

    # Create tracker
    print("\nInitializing tracker...")
    tracker = Gaze9Region(
        screen_width=1280,
        screen_height=720,
        camera_index=0,
        dwell_time=1.0,
        confidence_threshold=0.70,
        smoothing_window=5
    )

    if not tracker.start_camera():
        print("✗ Failed to start camera")
        return

    # Create tester
    tester = GazeTester(tracker)

    # Main menu
    while True:
        print("\n" + "="*60)
        print("TEST MENU")
        print("="*60)
        print("1. Run Accuracy Test")
        print("2. Run Performance Benchmark")
        print("3. Export Results to CSV")
        print("4. Load Calibration")
        print("5. New Calibration")
        print("6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == '1':
            tester.start_accuracy_test()
            # Ask to export
            export = input("\nExport results to CSV? (y/n): ").strip().lower()
            if export == 'y':
                tester.export_results_csv()
                tester.export_summary_csv()

        elif choice == '2':
            duration = input("Duration in seconds (default 60): ").strip()
            duration = int(duration) if duration.isdigit() else 60
            tester.run_performance_benchmark(duration)

        elif choice == '3':
            if len(tester.test_results) > 0:
                tester.export_results_csv()
                tester.export_summary_csv()
            else:
                print("No results to export. Run a test first.")

        elif choice == '4':
            students = tracker.calibration.list_calibrations()
            if students:
                print("\nAvailable calibrations:")
                for i, name in enumerate(students):
                    print(f"  {i+1}. {name}")
                student = input("Enter student name: ").strip()
                tracker.calibration.load_calibration(student)
            else:
                print("No calibrations found")

        elif choice == '5':
            tracker._run_calibration_interactive()
            save = input("\nSave calibration? (y/n): ").strip().lower()
            if save == 'y':
                name = input("Student name: ").strip()
                tracker.calibration.save_calibration(name)

        elif choice == '6':
            break

        else:
            print("Invalid option")

    tracker.stop_camera()
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
