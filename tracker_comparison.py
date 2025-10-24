"""
Eye Tracker Comparison Tool

Tests multiple eye tracking systems with identical test points
and generates comparative accuracy reports.

This is the main tool for determining which tracker works best
for your specific hardware setup (webcam vs IR camera).
"""

import cv2
import numpy as np
import time
import csv
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from eye_tracker_framework import EyeTrackerBase
from accuracy_tester import AccuracyTester, AccuracyTest


@dataclass
class TrackerComparison:
    """Comparison results between multiple trackers."""
    tracker_name: str
    mean_error: float
    median_error: float
    std_error: float
    min_error: float
    max_error: float
    mean_accuracy_100px: float
    success_rate: float  # Percentage of valid measurements
    total_tests: int


class TrackerComparisonTool:
    """
    Compare multiple eye tracking systems.

    Tests each tracker with identical test points and generates
    comparative accuracy metrics.
    """

    def __init__(self, screen_width: int, screen_height: int):
        """Initialize comparison tool."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.accuracy_tester = AccuracyTester(screen_width, screen_height)
        self.comparison_results: List[TrackerComparison] = []

    def compare_trackers(self, trackers: List[EyeTrackerBase], num_points: int = 9) -> List[TrackerComparison]:
        """
        Compare multiple trackers with identical test points.

        Args:
            trackers: List of initialized and calibrated trackers
            num_points: Number of test points

        Returns:
            List of comparison results
        """
        print("\n" + "="*60)
        print("MULTI-TRACKER COMPARISON")
        print("="*60)
        print(f"Trackers to test: {len(trackers)}")
        print(f"Test points: {num_points}")
        print("="*60)

        # Generate consistent test points
        test_points = self.accuracy_tester.generate_test_points(num_points)

        all_results = {}

        # Test each tracker
        for i, tracker in enumerate(trackers):
            print(f"\n{'='*60}")
            print(f"TESTING TRACKER {i+1}/{len(trackers)}: {tracker.tracker_name}")
            print(f"{'='*60}")

            if not tracker.is_calibrated:
                print(f"\n⚠ {tracker.tracker_name} is not calibrated!")
                print("Skipping this tracker...")
                continue

            # Run accuracy test
            results = self.accuracy_tester.run_accuracy_test(tracker, num_points)
            all_results[tracker.tracker_name] = results

            # Pause between trackers
            if i < len(trackers) - 1:
                print("\nPreparing next tracker...")
                time.sleep(2)

        # Generate comparison
        self.comparison_results = self._compute_comparison(all_results)

        # Print comparison table
        self._print_comparison_table()

        return self.comparison_results

    def _compute_comparison(self, all_results: Dict[str, List[AccuracyTest]]) -> List[TrackerComparison]:
        """Compute comparison statistics."""
        comparisons = []

        for tracker_name, results in all_results.items():
            errors = [r.get_error_distance() for r in results if r.get_error_distance() is not None]
            accuracies = [r.get_accuracy_percentage() for r in results]

            if not errors:
                print(f"\nWarning: No valid data for {tracker_name}")
                continue

            comparison = TrackerComparison(
                tracker_name=tracker_name,
                mean_error=float(np.mean(errors)),
                median_error=float(np.median(errors)),
                std_error=float(np.std(errors)),
                min_error=float(np.min(errors)),
                max_error=float(np.max(errors)),
                mean_accuracy_100px=float(np.mean(accuracies)),
                success_rate=100.0 * len(errors) / len(results),
                total_tests=len(results)
            )

            comparisons.append(comparison)

        # Sort by mean error (lower is better)
        comparisons.sort(key=lambda c: c.mean_error)

        return comparisons

    def _print_comparison_table(self):
        """Print formatted comparison table."""
        if not self.comparison_results:
            print("\nNo comparison results available")
            return

        print("\n" + "="*100)
        print("TRACKER COMPARISON RESULTS")
        print("="*100)

        # Header
        print(f"\n{'Tracker':<25} {'Mean Error':<12} {'Median':<10} {'Std Dev':<10} {'Accuracy':<12} {'Success Rate':<12}")
        print(f"{'Name':<25} {'(pixels)':<12} {'(px)':<10} {'(px)':<10} {'(±100px)':<12} {'(%)':<12}")
        print("-" * 100)

        # Data rows
        for i, comp in enumerate(self.comparison_results):
            rank = f"#{i+1}"
            print(f"{comp.tracker_name:<25} {comp.mean_error:>10.1f}  {comp.median_error:>8.1f}  {comp.std_error:>8.1f}  "
                  f"{comp.mean_accuracy_100px:>9.1f}%   {comp.success_rate:>9.1f}%   {rank}")

        print("="*100)

        # Winner
        if len(self.comparison_results) > 0:
            winner = self.comparison_results[0]
            print(f"\n🏆 BEST PERFORMER: {winner.tracker_name}")
            print(f"   Mean error: {winner.mean_error:.1f} pixels")
            print(f"   Accuracy (±100px): {winner.mean_accuracy_100px:.1f}%")

        print()

    def save_comparison(self, filename: str):
        """
        Save comparison results to CSV.

        Args:
            filename: Output CSV filename
        """
        if not self.comparison_results:
            print("No comparison results to save")
            return

        filepath = Path(filename)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Rank', 'Tracker_Name', 'Mean_Error_px', 'Median_Error_px',
                'Std_Dev_px', 'Min_Error_px', 'Max_Error_px',
                'Accuracy_100px', 'Success_Rate', 'Total_Tests'
            ])

            # Data
            for rank, comp in enumerate(self.comparison_results, 1):
                writer.writerow([
                    rank,
                    comp.tracker_name,
                    f"{comp.mean_error:.2f}",
                    f"{comp.median_error:.2f}",
                    f"{comp.std_error:.2f}",
                    f"{comp.min_error:.2f}",
                    f"{comp.max_error:.2f}",
                    f"{comp.mean_accuracy_100px:.2f}",
                    f"{comp.success_rate:.2f}",
                    comp.total_tests
                ])

        print(f"\n✓ Comparison saved to: {filepath}")

    def visualize_comparison(self):
        """
        Visualize comparison as bar chart.

        Shows mean error for each tracker.
        """
        if not self.comparison_results:
            print("No comparison results to visualize")
            return

        # Create visualization
        height = 600
        width = 800
        vis = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Chart parameters
        margin_left = 200
        margin_right = 50
        margin_top = 80
        margin_bottom = 80
        chart_width = width - margin_left - margin_right
        chart_height = height - margin_top - margin_bottom

        # Find max error for scaling
        max_error = max(c.mean_error for c in self.comparison_results)

        # Draw bars
        num_trackers = len(self.comparison_results)
        bar_height = chart_height // (num_trackers * 2)
        spacing = bar_height

        for i, comp in enumerate(self.comparison_results):
            y = margin_top + i * (bar_height + spacing)
            bar_length = int((comp.mean_error / max_error) * chart_width)

            # Bar color (green = best, red = worst)
            color_ratio = i / max(1, num_trackers - 1)
            color = (0, int(255 * (1 - color_ratio)), int(255 * color_ratio))

            # Draw bar
            cv2.rectangle(vis, (margin_left, y),
                        (margin_left + bar_length, y + bar_height),
                        color, -1)

            # Tracker name
            cv2.putText(vis, comp.tracker_name, (10, y + bar_height // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Error value
            cv2.putText(vis, f"{comp.mean_error:.1f}px",
                       (margin_left + bar_length + 10, y + bar_height // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Title
        cv2.putText(vis, "Tracker Comparison - Mean Error (Lower is Better)",
                   (width // 2 - 250, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Show
        cv2.imshow('Tracker Comparison', vis)
        print("\nComparison visualization shown. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def setup_and_test_trackers(screen_width: int, screen_height: int,
                            num_points: int = 9,
                            test_hough: bool = True,
                            test_webgazer: bool = False,
                            test_pupil_labs: bool = False,
                            camera_index: int = 0):
    """
    Complete workflow: setup, calibrate, and test multiple trackers.

    Args:
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        num_points: Number of test points
        test_hough: Test Hough Circle tracker
        test_webgazer: Test WebGazer tracker
        test_pupil_labs: Test Pupil Labs tracker
        camera_index: Camera device index
    """
    print("\n" + "="*60)
    print("EYE TRACKER COMPARISON - COMPLETE WORKFLOW")
    print("="*60)

    comparison_tool = TrackerComparisonTool(screen_width, screen_height)
    trackers = []

    # Setup Hough Circle tracker
    if test_hough:
        print("\n" + "="*60)
        print("SETTING UP: Hough Circle Tracker")
        print("="*60)

        from trackers.hough_tracker_gaze import HoughGazeTracker

        hough = HoughGazeTracker(screen_width, screen_height)

        if hough.start(camera_index):
            print("\n✓ Hough Circle tracker started")

            # Calibration
            print("\nCalibration required...")
            cal_points = comparison_tool.accuracy_tester.generate_test_points(num_points)

            print("Look at each calibration point as it appears.")
            input("Press Enter to start calibration...")

            if hough.calibrate(cal_points):
                print("✓ Calibration complete")
                trackers.append(hough)

                # Save calibration
                hough.save_calibration("hough_calibration.json")
            else:
                print("✗ Calibration failed")
                hough.stop()
        else:
            print("✗ Failed to start Hough Circle tracker")

    # Setup WebGazer
    if test_webgazer:
        print("\n" + "="*60)
        print("SETTING UP: WebGazer Tracker")
        print("="*60)
        print("\nWebGazer requires:")
        print("  1. Browser window will open")
        print("  2. Complete calibration in browser")
        print("  3. Keep browser window open during test")

        from trackers.webgazer_tracker import WebGazerTracker

        webgazer = WebGazerTracker(screen_width, screen_height)

        if webgazer.start():
            print("\n✓ WebGazer started (check browser window)")
            print("\nComplete calibration in browser, then return here.")
            input("Press Enter when calibration is complete...")

            webgazer.is_calibrated = True  # User confirms calibration
            trackers.append(webgazer)
        else:
            print("✗ Failed to start WebGazer")

    # Setup Pupil Labs
    if test_pupil_labs:
        print("\n" + "="*60)
        print("SETTING UP: Pupil Labs Tracker")
        print("="*60)
        print("\nPupil Labs requires:")
        print("  1. Pupil Capture software running")
        print("  2. Cameras connected and working")
        print("  3. Calibration completed in Pupil Capture")

        from trackers.pupil_labs_tracker import PupilLabsTracker

        pupil = PupilLabsTracker(screen_width, screen_height)

        if pupil.start():
            print("\n✓ Pupil Labs connected")
            print("\nComplete calibration in Pupil Capture if not already done.")
            input("Press Enter when ready to test...")

            pupil.is_calibrated = True
            trackers.append(pupil)
        else:
            print("✗ Failed to connect to Pupil Labs")

    # Run comparison
    if len(trackers) == 0:
        print("\n✗ No trackers available for testing")
        return

    print(f"\n\n{'='*60}")
    print(f"READY TO TEST {len(trackers)} TRACKER(S)")
    print(f"{'='*60}")
    input("Press Enter to begin accuracy testing...")

    # Compare all trackers
    results = comparison_tool.compare_trackers(trackers, num_points)

    # Save results
    if results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"tracker_comparison_{timestamp}.csv"
        comparison_tool.save_comparison(output_file)

        # Visualize
        visualize = input("\nVisualize comparison chart? (y/n): ")
        if visualize.lower() == 'y':
            comparison_tool.visualize_comparison()

    # Cleanup
    for tracker in trackers:
        try:
            tracker.stop()
        except:
            pass

    print("\n✓ Comparison complete!")


def main():
    """Interactive comparison tool."""
    print("\n" + "="*60)
    print("EYE TRACKER COMPARISON TOOL")
    print("="*60)
    print("\nThis tool tests multiple eye tracking systems with identical")
    print("test points and compares their accuracy.")
    print("="*60)

    # Configuration
    screen_width = int(input("\nScreen width (pixels, default 1920): ") or "1920")
    screen_height = int(input("Screen height (pixels, default 1080): ") or "1080")

    print("\nTest configuration:")
    print("  5 points - Quick test (CVI-friendly)")
    print("  9 points - Standard test")
    print("  13 points - Detailed test")
    num_points = int(input("Number of test points (5/9/13, default 9): ") or "9")

    # Select trackers
    print("\n" + "="*60)
    print("SELECT TRACKERS TO TEST")
    print("="*60)
    print("\n1. Hough Circle (OpenCV-based, any webcam/IR camera)")
    print("   - Pros: Works with any camera, no special software")
    print("   - Cons: Moderate accuracy, needs good pupil detection")
    print("\n2. WebGazer.js (Browser-based)")
    print("   - Pros: No installation, works in browser")
    print("   - Cons: Requires webcam, JavaScript enabled")
    print("\n3. Pupil Labs (Professional)")
    print("   - Pros: High accuracy, research-grade")
    print("   - Cons: Requires Pupil Labs hardware (~$2000+)")

    test_hough = input("\nTest Hough Circle? (y/n, default y): ").lower() != 'n'
    test_webgazer = input("Test WebGazer? (y/n, default n): ").lower() == 'y'
    test_pupil_labs = input("Test Pupil Labs? (y/n, default n): ").lower() == 'y'

    if not (test_hough or test_webgazer or test_pupil_labs):
        print("\nNo trackers selected. Exiting.")
        return

    # Camera selection
    camera_index = 0
    if test_hough:
        print("\n" + "="*60)
        print("CAMERA SELECTION")
        print("="*60)
        print("\nYou mentioned having both webcam and IR camera.")
        print("Hough Circle tracker can use either.")
        print("\nTip: Run camera_selector.py first to find best camera")
        camera_index = int(input("\nCamera index (0=default, 1=second camera): ") or "0")

    # Run complete workflow
    setup_and_test_trackers(
        screen_width, screen_height, num_points,
        test_hough, test_webgazer, test_pupil_labs,
        camera_index
    )


if __name__ == "__main__":
    main()
