"""
9-Region Eye Gaze Calibration System for AAC
Designed for students with visual impairments

Features:
- 9-point calibration process (3x3 grid)
- Large, high-contrast calibration targets
- Audio feedback for accessibility
- Save/load calibration profiles for individual students
- Simple keyboard controls (no mouse needed)
- Calibration quality indicator
"""

import cv2
import numpy as np
import json
import os
import time
from typing import Optional, Tuple, Dict, List
from datetime import datetime

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Install for audio feedback: pip install pyttsx3")


class CalibrationSystem:
    """
    9-point calibration system for eye gaze tracking.
    Maps pupil positions to 9 screen regions.
    """

    def __init__(self, screen_width: int = 1280, screen_height: int = 720,
                 audio_feedback: bool = True):
        """
        Initialize calibration system.

        Args:
            screen_width: Display width in pixels
            screen_height: Display height in pixels
            audio_feedback: Enable text-to-speech feedback
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.audio_feedback = audio_feedback and TTS_AVAILABLE

        # Text-to-speech engine
        self.tts_engine = None
        if self.audio_feedback:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)  # Speed
                self.tts_engine.setProperty('volume', 1.0)  # Volume
            except:
                self.audio_feedback = False
                print("Warning: Could not initialize TTS engine")

        # 9 calibration points (3x3 grid)
        self.calibration_points = self._generate_calibration_points()

        # Calibration data: {region_id: [(pupil_x, pupil_y), ...]}
        self.calibration_data: Dict[int, List[Tuple[int, int]]] = {
            i: [] for i in range(9)
        }

        # Calibration result: {region_id: (avg_pupil_x, avg_pupil_y)}
        self.calibration_map: Optional[Dict[int, Tuple[float, float]]] = None

        # Current calibration state
        self.current_point_index = 0
        self.samples_collected = 0
        self.samples_needed = 30  # Collect 30 samples per point
        self.is_calibrated = False
        self.calibration_quality = 0.0

        # Visual settings (high contrast for accessibility)
        self.target_size = 80  # Large target for visibility
        self.target_color = (0, 255, 255)  # Bright yellow
        self.bg_color = (0, 0, 0)  # Black background
        self.text_color = (255, 255, 255)  # White text

    def _generate_calibration_points(self) -> List[Tuple[int, int]]:
        """
        Generate 9 calibration points in 3x3 grid.

        Returns:
            List of (x, y) screen positions
        """
        points = []
        margin_x = self.screen_width // 6  # Margin from edges
        margin_y = self.screen_height // 6

        for row in range(3):
            for col in range(3):
                x = margin_x + col * (self.screen_width - 2 * margin_x) // 2
                y = margin_y + row * (self.screen_height - 2 * margin_y) // 2
                points.append((x, y))

        return points

    def speak(self, text: str):
        """
        Speak text using TTS (if enabled).

        Args:
            text: Text to speak
        """
        if self.audio_feedback and self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass  # Silently fail if TTS fails

    def start_calibration(self):
        """Start new calibration session."""
        self.current_point_index = 0
        self.samples_collected = 0
        self.calibration_data = {i: [] for i in range(9)}
        self.is_calibrated = False
        self.calibration_quality = 0.0

        self.speak("Starting calibration. Look at the yellow circle.")

    def add_sample(self, pupil_position: Tuple[int, int]) -> bool:
        """
        Add calibration sample for current point.

        Args:
            pupil_position: (x, y) pupil coordinates

        Returns:
            True if point completed, False otherwise
        """
        if pupil_position is None:
            return False

        region_id = self.current_point_index
        self.calibration_data[region_id].append(pupil_position)
        self.samples_collected += 1

        # Check if we have enough samples for this point
        if self.samples_collected >= self.samples_needed:
            return True

        return False

    def next_point(self) -> bool:
        """
        Move to next calibration point.

        Returns:
            True if calibration complete, False otherwise
        """
        self.current_point_index += 1
        self.samples_collected = 0

        if self.current_point_index >= 9:
            # Calibration complete
            self._compute_calibration_map()
            self.speak("Calibration complete")
            return True

        # Announce next point
        region_names = [
            "top left", "top center", "top right",
            "middle left", "center", "middle right",
            "bottom left", "bottom center", "bottom right"
        ]
        self.speak(f"Point {self.current_point_index + 1}. {region_names[self.current_point_index]}")

        return False

    def _compute_calibration_map(self):
        """Compute calibration map from collected samples."""
        self.calibration_map = {}

        for region_id, samples in self.calibration_data.items():
            if len(samples) > 0:
                # Average all samples for this region
                avg_x = np.mean([s[0] for s in samples])
                avg_y = np.mean([s[1] for s in samples])
                self.calibration_map[region_id] = (avg_x, avg_y)

        self.is_calibrated = True
        self._compute_calibration_quality()

    def _compute_calibration_quality(self):
        """
        Compute calibration quality score (0.0 - 1.0).
        Based on variance of samples and coverage of regions.
        """
        if not self.calibration_map:
            self.calibration_quality = 0.0
            return

        total_variance = 0.0
        num_regions = 0

        for region_id, samples in self.calibration_data.items():
            if len(samples) < 5:
                continue

            avg_pos = self.calibration_map[region_id]

            # Calculate variance
            variance_x = np.var([s[0] - avg_pos[0] for s in samples])
            variance_y = np.var([s[1] - avg_pos[1] for s in samples])
            total_variance += (variance_x + variance_y)
            num_regions += 1

        if num_regions == 0:
            self.calibration_quality = 0.0
            return

        # Lower variance = higher quality
        avg_variance = total_variance / num_regions

        # Map variance to quality (0-1)
        # Good: variance < 100, Poor: variance > 500
        quality = max(0.0, min(1.0, 1.0 - (avg_variance / 500.0)))

        # Penalty for missing regions
        coverage = num_regions / 9.0
        quality *= coverage

        self.calibration_quality = quality

    def map_pupil_to_region(self, pupil_position: Tuple[int, int]) -> Optional[int]:
        """
        Map pupil position to region ID (0-8).

        Args:
            pupil_position: (x, y) pupil coordinates

        Returns:
            Region ID (0-8) or None if not calibrated
        """
        if not self.is_calibrated or pupil_position is None:
            return None

        # Find closest calibration point using Euclidean distance
        min_distance = float('inf')
        closest_region = None

        for region_id, calibrated_pos in self.calibration_map.items():
            dx = pupil_position[0] - calibrated_pos[0]
            dy = pupil_position[1] - calibrated_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)

            if distance < min_distance:
                min_distance = distance
                closest_region = region_id

        return closest_region

    def get_region_confidence(self, pupil_position: Tuple[int, int],
                             region_id: int) -> float:
        """
        Get confidence score for region assignment (0.0 - 1.0).

        Args:
            pupil_position: (x, y) pupil coordinates
            region_id: Region ID to check

        Returns:
            Confidence score (higher = more confident)
        """
        if not self.is_calibrated or region_id not in self.calibration_map:
            return 0.0

        calibrated_pos = self.calibration_map[region_id]

        # Calculate distance
        dx = pupil_position[0] - calibrated_pos[0]
        dy = pupil_position[1] - calibrated_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)

        # Convert distance to confidence
        # Close: confidence = 1.0, Far: confidence = 0.0
        # Threshold: 100 pixels = full confidence, 300 pixels = no confidence
        confidence = max(0.0, min(1.0, 1.0 - (distance / 300.0)))

        return confidence

    def draw_calibration_screen(self, frame: np.ndarray,
                                pupil_position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Draw calibration interface.

        Args:
            frame: Camera frame to overlay on
            pupil_position: Current pupil position

        Returns:
            Annotated frame
        """
        # Create black background
        display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # Draw all calibration points (small dots)
        for i, point in enumerate(self.calibration_points):
            if i < self.current_point_index:
                # Completed points (green)
                cv2.circle(display, point, 15, (0, 255, 0), -1)
            elif i == self.current_point_index:
                # Current point (large yellow target with animation)
                # Pulsing effect
                pulse = int(20 * np.sin(time.time() * 3) + 60)
                cv2.circle(display, point, pulse, self.target_color, 5)
                cv2.circle(display, point, 10, self.target_color, -1)
            else:
                # Future points (gray)
                cv2.circle(display, point, 10, (100, 100, 100), 2)

        # Draw progress bar
        progress = (self.current_point_index * self.samples_needed + self.samples_collected) / (9 * self.samples_needed)
        bar_width = self.screen_width - 200
        bar_height = 30
        bar_x = 100
        bar_y = self.screen_height - 100

        # Progress bar background
        cv2.rectangle(display, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)

        # Progress bar fill
        fill_width = int(bar_width * progress)
        cv2.rectangle(display, (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height),
                     (0, 255, 0), -1)

        # Progress text
        progress_text = f"Progress: {int(progress * 100)}%"
        cv2.putText(display, progress_text, (bar_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)

        # Current point info (large text for accessibility)
        current_point = self.calibration_points[self.current_point_index]
        region_names = [
            "TOP LEFT", "TOP CENTER", "TOP RIGHT",
            "MIDDLE LEFT", "CENTER", "MIDDLE RIGHT",
            "BOTTOM LEFT", "BOTTOM CENTER", "BOTTOM RIGHT"
        ]

        info_text = f"Point {self.current_point_index + 1}/9: {region_names[self.current_point_index]}"
        cv2.putText(display, info_text, (50, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.text_color, 3)

        # Samples collected
        samples_text = f"Samples: {self.samples_collected}/{self.samples_needed}"
        cv2.putText(display, samples_text, (50, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.text_color, 2)

        # Draw pupil position if available
        if pupil_position:
            cv2.circle(display, pupil_position, 8, (255, 0, 255), -1)
            cv2.circle(display, pupil_position, 15, (255, 0, 255), 2)

        # Instructions (large, high-contrast)
        instructions = [
            "Look at the yellow circle",
            "Hold your gaze steady",
            "Press SPACE to skip point",
            "Press ESC to cancel"
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(display, instruction, (50, self.screen_height - 200 + i * 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        return display

    def save_calibration(self, student_name: str,
                        calibration_dir: str = "calibrations") -> bool:
        """
        Save calibration profile to file.

        Args:
            student_name: Name of student
            calibration_dir: Directory to save calibrations

        Returns:
            True if saved successfully
        """
        if not self.is_calibrated:
            return False

        # Create directory if needed
        os.makedirs(calibration_dir, exist_ok=True)

        # Prepare data
        calibration_data = {
            'student_name': student_name,
            'timestamp': datetime.now().isoformat(),
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'calibration_map': {
                str(k): v for k, v in self.calibration_map.items()
            },
            'calibration_quality': self.calibration_quality,
            'calibration_points': self.calibration_points
        }

        # Save to file
        filename = os.path.join(calibration_dir, f"{student_name}_calibration.json")

        try:
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)

            print(f"Calibration saved: {filename}")
            self.speak("Calibration saved")
            return True

        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False

    def load_calibration(self, student_name: str,
                        calibration_dir: str = "calibrations") -> bool:
        """
        Load calibration profile from file.

        Args:
            student_name: Name of student
            calibration_dir: Directory containing calibrations

        Returns:
            True if loaded successfully
        """
        filename = os.path.join(calibration_dir, f"{student_name}_calibration.json")

        if not os.path.exists(filename):
            print(f"Calibration file not found: {filename}")
            return False

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # Verify screen dimensions match
            if (data['screen_width'] != self.screen_width or
                data['screen_height'] != self.screen_height):
                print("Warning: Screen dimensions don't match calibration")
                print(f"Calibration: {data['screen_width']}x{data['screen_height']}")
                print(f"Current: {self.screen_width}x{self.screen_height}")
                return False

            # Load calibration map
            self.calibration_map = {
                int(k): tuple(v) for k, v in data['calibration_map'].items()
            }
            self.calibration_quality = data['calibration_quality']
            self.is_calibrated = True

            print(f"Calibration loaded: {student_name}")
            print(f"Quality: {self.calibration_quality:.2%}")
            print(f"Date: {data['timestamp']}")

            self.speak(f"Calibration loaded for {student_name}")
            return True

        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

    def list_calibrations(self, calibration_dir: str = "calibrations") -> List[str]:
        """
        List available calibration profiles.

        Args:
            calibration_dir: Directory containing calibrations

        Returns:
            List of student names with calibrations
        """
        if not os.path.exists(calibration_dir):
            return []

        students = []
        for filename in os.listdir(calibration_dir):
            if filename.endswith('_calibration.json'):
                student_name = filename.replace('_calibration.json', '')
                students.append(student_name)

        return students

    def get_quality_description(self) -> str:
        """
        Get human-readable calibration quality description.

        Returns:
            Quality description string
        """
        if not self.is_calibrated:
            return "Not calibrated"

        if self.calibration_quality >= 0.85:
            return "Excellent"
        elif self.calibration_quality >= 0.70:
            return "Good"
        elif self.calibration_quality >= 0.50:
            return "Fair"
        else:
            return "Poor - Recalibrate recommended"


if __name__ == "__main__":
    """Test calibration system."""
    import sys

    print("="*60)
    print("9-REGION CALIBRATION SYSTEM TEST")
    print("="*60)
    print("\nThis test demonstrates the calibration process.")
    print("For production use, integrate with gaze_9region.py")
    print("\nPress any key to start...")
    input()

    # Create calibration system
    cal = CalibrationSystem(screen_width=1280, screen_height=720, audio_feedback=True)

    print("\nCalibration system initialized")
    print(f"TTS enabled: {cal.audio_feedback}")
    print(f"Calibration points: {len(cal.calibration_points)}")
    print(f"Samples per point: {cal.samples_needed}")

    # Test: simulate calibration data
    print("\n" + "="*60)
    print("SIMULATING CALIBRATION")
    print("="*60)
    print("(In production, this uses real pupil tracking)")

    cal.start_calibration()

    # Simulate collecting samples for each point
    for point_idx in range(9):
        print(f"\nPoint {point_idx + 1}/9: {cal.calibration_points[point_idx]}")

        # Simulate pupil positions near calibration point
        base_x, base_y = cal.calibration_points[point_idx]

        for sample in range(cal.samples_needed):
            # Add small random offset to simulate tracking jitter
            offset_x = np.random.randint(-10, 10)
            offset_y = np.random.randint(-10, 10)
            pupil_pos = (base_x + offset_x, base_y + offset_y)

            completed = cal.add_sample(pupil_pos)

            if completed:
                is_done = cal.next_point()
                if is_done:
                    break

    # Show results
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    print(f"Calibrated: {cal.is_calibrated}")
    print(f"Quality: {cal.calibration_quality:.2%} ({cal.get_quality_description()})")

    # Test save/load
    print("\n" + "="*60)
    print("TESTING SAVE/LOAD")
    print("="*60)

    student_name = "test_student"
    if cal.save_calibration(student_name):
        print("Save successful!")

        # Create new calibration system and load
        cal2 = CalibrationSystem(screen_width=1280, screen_height=720)
        if cal2.load_calibration(student_name):
            print("Load successful!")
            print(f"Loaded quality: {cal2.calibration_quality:.2%}")

    # Test region mapping
    print("\n" + "="*60)
    print("TESTING REGION MAPPING")
    print("="*60)

    test_positions = [
        (200, 120, "Top Left"),
        (640, 120, "Top Center"),
        (1080, 120, "Top Right"),
        (640, 360, "Center"),
    ]

    for x, y, expected in test_positions:
        region = cal.map_pupil_to_region((x, y))
        confidence = cal.get_region_confidence((x, y), region) if region is not None else 0.0
        print(f"Position ({x}, {y}) -> Region {region} ({expected}) - Confidence: {confidence:.2%}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Integrate with gaze_9region.py for real pupil tracking")
    print("2. Use calibration in demo_aac_board.py")
    print("3. Create student calibration profiles")
