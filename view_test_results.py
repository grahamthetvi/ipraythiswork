"""
Test Results Viewer

Loads and visualizes accuracy test results from CSV files.
Generates heat maps, error distributions, and comparative charts.
"""

import csv
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TestResult:
    """Single test result from CSV."""
    tracker: str
    target_x: int
    target_y: int
    gaze_x: float
    gaze_y: float
    error: float
    accuracy: float
    samples: int


class ResultsViewer:
    """Visualize eye tracking accuracy test results."""

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        """Initialize results viewer."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.results: List[TestResult] = []

    def load_csv(self, filename: str) -> bool:
        """
        Load test results from CSV file.

        Args:
            filename: Path to CSV file

        Returns:
            True if loaded successfully
        """
        filepath = Path(filename)

        if not filepath.exists():
            print(f"File not found: {filepath}")
            return False

        self.results = []

        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Skip rows with invalid data
                    if row.get('Avg_Gaze_X') == 'N/A' or row.get('Error_Distance') == 'N/A':
                        continue

                    result = TestResult(
                        tracker=row['Tracker'],
                        target_x=int(row['Target_X']),
                        target_y=int(row['Target_Y']),
                        gaze_x=float(row['Avg_Gaze_X']),
                        gaze_y=float(row['Avg_Gaze_Y']),
                        error=float(row['Error_Distance']),
                        accuracy=float(row['Accuracy_100px']),
                        samples=int(row['Samples_Collected'])
                    )

                    self.results.append(result)

            print(f"✓ Loaded {len(self.results)} test results from {filepath}")
            return True

        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False

    def print_summary(self):
        """Print summary statistics."""
        if not self.results:
            print("No results loaded")
            return

        # Group by tracker
        trackers = {}
        for result in self.results:
            if result.tracker not in trackers:
                trackers[result.tracker] = []
            trackers[result.tracker].append(result)

        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)

        for tracker_name, tracker_results in trackers.items():
            print(f"\n{tracker_name}:")
            print("-" * 40)

            errors = [r.error for r in tracker_results]
            accuracies = [r.accuracy for r in tracker_results]

            print(f"  Tests: {len(tracker_results)}")
            print(f"  Mean Error: {np.mean(errors):.1f} pixels")
            print(f"  Median Error: {np.median(errors):.1f} pixels")
            print(f"  Std Dev: {np.std(errors):.1f} pixels")
            print(f"  Min Error: {np.min(errors):.1f} pixels")
            print(f"  Max Error: {np.max(errors):.1f} pixels")
            print(f"  Mean Accuracy (±100px): {np.mean(accuracies):.1f}%")

        print("="*60)

    def visualize_error_vectors(self):
        """
        Visualize error vectors showing target vs measured gaze.

        Shows targets in red, measured gaze in green, with yellow arrows.
        """
        if not self.results:
            print("No results to visualize")
            return

        # Create visualization
        vis = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # Group by tracker (use different colors if multiple trackers)
        trackers = list(set(r.tracker for r in self.results))
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
        ]

        for result in self.results:
            # Color based on tracker
            tracker_idx = trackers.index(result.tracker)
            gaze_color = colors[tracker_idx % len(colors)]

            # Draw target
            cv2.circle(vis, (result.target_x, result.target_y), 20, (0, 0, 255), 2)
            cv2.circle(vis, (result.target_x, result.target_y), 5, (0, 0, 255), -1)

            # Draw measured gaze
            gaze_x, gaze_y = int(result.gaze_x), int(result.gaze_y)
            cv2.circle(vis, (gaze_x, gaze_y), 15, gaze_color, 2)

            # Draw error vector
            cv2.arrowedLine(vis, (result.target_x, result.target_y),
                          (gaze_x, gaze_y), (0, 255, 255), 2, tipLength=0.3)

            # Label error distance
            mid_x = (result.target_x + gaze_x) // 2
            mid_y = (result.target_y + gaze_y) // 2
            cv2.putText(vis, f"{result.error:.0f}px", (mid_x + 10, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Legend
        y_offset = 30
        cv2.putText(vis, "Red = Target | Arrows = Error Vector",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

        for i, tracker_name in enumerate(trackers):
            color = colors[i % len(colors)]
            cv2.putText(vis, f"{tracker_name}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25

        cv2.imshow('Error Vectors', vis)
        print("\nError vector visualization. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize_heat_map(self):
        """
        Visualize error heat map.

        Shows where on screen the tracker is more/less accurate.
        """
        if not self.results:
            print("No results to visualize")
            return

        # Create heat map (downscaled for visualization)
        scale = 4
        heat_width = self.screen_width // scale
        heat_height = self.screen_height // scale

        heat_map = np.zeros((heat_height, heat_width), dtype=np.float32)
        count_map = np.zeros((heat_height, heat_width), dtype=np.int32)

        # Accumulate errors
        for result in self.results:
            x = int(result.target_x / scale)
            y = int(result.target_y / scale)

            if 0 <= x < heat_width and 0 <= y < heat_height:
                heat_map[y, x] += result.error
                count_map[y, x] += 1

        # Average
        mask = count_map > 0
        heat_map[mask] /= count_map[mask]

        # Normalize to 0-255
        if np.max(heat_map) > 0:
            heat_map = (heat_map / np.max(heat_map) * 255).astype(np.uint8)

        # Apply colormap (red = high error, green = low error)
        heat_colored = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

        # Resize back to screen dimensions
        heat_colored = cv2.resize(heat_colored, (self.screen_width, self.screen_height))

        # Add test points
        for result in self.results:
            cv2.circle(heat_colored, (result.target_x, result.target_y),
                     10, (255, 255, 255), 2)

        # Add scale
        cv2.putText(heat_colored, "Heat Map: Error Distribution",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(heat_colored, "Red = High Error | Blue = Low Error",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Error Heat Map', heat_colored)
        print("\nError heat map visualization. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize_error_distribution(self):
        """
        Visualize error distribution as histogram.

        Shows how errors are distributed across test points.
        """
        if not self.results:
            print("No results to visualize")
            return

        # Group by tracker
        trackers = {}
        for result in self.results:
            if result.tracker not in trackers:
                trackers[result.tracker] = []
            trackers[result.tracker].append(result.error)

        # Create histogram visualization
        height = 600
        width = 800
        vis = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Chart parameters
        margin_left = 80
        margin_right = 50
        margin_top = 80
        margin_bottom = 80
        chart_width = width - margin_left - margin_right
        chart_height = height - margin_top - margin_bottom

        # Determine bins
        all_errors = [r.error for r in self.results]
        max_error = max(all_errors)
        num_bins = 20
        bin_width = max_error / num_bins

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

        # Draw histogram for each tracker
        for tracker_idx, (tracker_name, errors) in enumerate(trackers.items()):
            color = colors[tracker_idx % len(colors)]

            # Compute histogram
            hist, _ = np.histogram(errors, bins=num_bins, range=(0, max_error))
            max_count = max(hist)

            # Draw bars
            bar_width = chart_width // num_bins

            for i, count in enumerate(hist):
                if count == 0:
                    continue

                bar_height = int((count / max_count) * chart_height)
                x = margin_left + i * bar_width
                y = margin_top + chart_height - bar_height

                # Offset bars for multiple trackers
                offset = tracker_idx * 5
                cv2.rectangle(vis, (x + offset, y),
                            (x + bar_width - 5 + offset, margin_top + chart_height),
                            color, -1)

        # Axes
        cv2.line(vis, (margin_left, margin_top + chart_height),
                (margin_left + chart_width, margin_top + chart_height),
                (0, 0, 0), 2)
        cv2.line(vis, (margin_left, margin_top),
                (margin_left, margin_top + chart_height),
                (0, 0, 0), 2)

        # Labels
        cv2.putText(vis, "Error Distribution", (width // 2 - 100, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(vis, "Error (pixels)", (width // 2 - 60, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(vis, "Frequency", (10, height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # X-axis ticks
        for i in range(0, num_bins + 1, 5):
            x = margin_left + int(i / num_bins * chart_width)
            error_val = int(i * bin_width)
            cv2.putText(vis, str(error_val), (x - 15, height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Legend
        legend_y = margin_top + 20
        for tracker_idx, tracker_name in enumerate(trackers.keys()):
            color = colors[tracker_idx % len(colors)]
            cv2.rectangle(vis, (width - 200, legend_y),
                        (width - 180, legend_y + 15), color, -1)
            cv2.putText(vis, tracker_name, (width - 170, legend_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            legend_y += 25

        cv2.imshow('Error Distribution', vis)
        print("\nError distribution histogram. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize_accuracy_by_region(self):
        """
        Show accuracy broken down by screen region.

        Divides screen into 9 regions and shows mean error for each.
        """
        if not self.results:
            print("No results to visualize")
            return

        # Create visualization
        vis = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # Define 9 regions (3x3 grid)
        region_width = self.screen_width // 3
        region_height = self.screen_height // 3

        # Calculate mean error for each region
        region_errors = {}

        for row in range(3):
            for col in range(3):
                region_id = row * 3 + col
                region_x = col * region_width
                region_y = row * region_height

                # Find results in this region
                region_results = [
                    r for r in self.results
                    if region_x <= r.target_x < region_x + region_width and
                       region_y <= r.target_y < region_y + region_height
                ]

                if region_results:
                    mean_error = np.mean([r.error for r in region_results])
                    region_errors[region_id] = mean_error
                else:
                    region_errors[region_id] = None

        # Find max error for color scaling
        valid_errors = [e for e in region_errors.values() if e is not None]
        max_error = max(valid_errors) if valid_errors else 100

        # Draw regions with color coding
        for row in range(3):
            for col in range(3):
                region_id = row * 3 + col
                region_x = col * region_width
                region_y = row * region_height

                error = region_errors[region_id]

                if error is not None:
                    # Color: green (low error) to red (high error)
                    error_ratio = min(1.0, error / max_error)
                    color = (0, int(255 * (1 - error_ratio)), int(255 * error_ratio))

                    # Fill region
                    cv2.rectangle(vis,
                                (region_x + 5, region_y + 5),
                                (region_x + region_width - 5, region_y + region_height - 5),
                                color, -1)

                    # Label
                    text = f"{error:.1f}px"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                    text_x = region_x + (region_width - text_size[0]) // 2
                    text_y = region_y + (region_height + text_size[1]) // 2

                    cv2.putText(vis, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                else:
                    # No data
                    cv2.rectangle(vis,
                                (region_x + 5, region_y + 5),
                                (region_x + region_width - 5, region_y + region_height - 5),
                                (50, 50, 50), 2)

        # Grid lines
        for i in range(1, 3):
            cv2.line(vis, (i * region_width, 0), (i * region_width, self.screen_height),
                    (255, 255, 255), 2)
            cv2.line(vis, (0, i * region_height), (self.screen_width, i * region_height),
                    (255, 255, 255), 2)

        # Title
        cv2.putText(vis, "Mean Error by Screen Region",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(vis, "Green = Low Error | Red = High Error",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Accuracy by Region', vis)
        print("\nAccuracy by region visualization. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Interactive results viewer."""
    print("\n" + "="*60)
    print("TEST RESULTS VIEWER")
    print("="*60)

    # Get CSV file
    print("\nEnter path to test results CSV file:")
    print("(Files generated by accuracy_tester.py or tracker_comparison.py)")
    filename = input("File path: ")

    if not filename:
        print("No file specified")
        return

    # Load results
    viewer = ResultsViewer()

    if not viewer.load_csv(filename):
        return

    # Print summary
    viewer.print_summary()

    # Interactive menu
    while True:
        print("\n" + "="*60)
        print("VISUALIZATION OPTIONS")
        print("="*60)
        print("1. Error vectors (target vs measured gaze)")
        print("2. Error heat map (spatial accuracy distribution)")
        print("3. Error distribution histogram")
        print("4. Accuracy by screen region (3x3 grid)")
        print("5. Print summary again")
        print("6. Load different file")
        print("0. Exit")

        choice = input("\nSelect option (0-6): ")

        if choice == '1':
            viewer.visualize_error_vectors()
        elif choice == '2':
            viewer.visualize_heat_map()
        elif choice == '3':
            viewer.visualize_error_distribution()
        elif choice == '4':
            viewer.visualize_accuracy_by_region()
        elif choice == '5':
            viewer.print_summary()
        elif choice == '6':
            filename = input("New file path: ")
            viewer.load_csv(filename)
            viewer.print_summary()
        elif choice == '0':
            break
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
