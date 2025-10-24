"""
CVI-Optimized 5-Point Calibration System
Designed for students with Cortical Visual Impairment

Features:
- 5 calibration points instead of 9
- High contrast (black background, red/yellow targets)
- Large, simple target shapes
- Extra time per point
- Red pulsing animation for attention
- Audio feedback optimized for CVI students
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


class CVICalibration:
    """
    5-point calibration system optimized for CVI students.
    """

    def __init__(self, screen_width: int = 1280, screen_height: int = 720,
                 audio_feedback: bool = True):
        """
        Initialize CVI calibration system.

        Args:
            screen_width: Display width
            screen_height: Display height
            audio_feedback: Enable TTS
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.audio_feedback = audio_feedback and TTS_AVAILABLE

        # TTS
        self.tts_engine = None
        if self.audio_feedback:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 130)  # Slower
                self.tts_engine.setProperty('volume', 1.0)
            except:
                self.audio_feedback = False

        # 5 calibration points
        self.calibration_points = self._generate_5point_calibration()

        # Calibration data
        self.calibration_data: Dict[int, List[Tuple[int, int]]] = {
            i: [] for i in range(5)
        }

        # Calibration map (5 points)
        self.calibration_map: Optional[Dict[int, Tuple[float, float]]] = None

        # State
        self.current_point_index = 0
        self.samples_collected = 0
        self.samples_needed = 40  # More samples for CVI (extra time)
        self.is_calibrated = False
        self.calibration_quality = 0.0

        # CVI Colors
        self.bg_color = (0, 0, 0)  # Pure black
        self.target_color = (0, 0, 255)  # Red (primary color for CVI)
        self.text_color = (255, 255, 255)  # White

    def _generate_5point_calibration(self) -> List[Tuple[int, int]]:
        """
        Generate 5 calibration points.

        Layout:
        [0]         [1]
             [2]
        [3]         [4]

        Returns:
            List of (x, y) positions
        """
        margin_x = self.screen_width // 5
        margin_y = self.screen_height // 5

        points = [
            # Top-left
            (margin_x, margin_y),
            # Top-right
            (self.screen_width - margin_x, margin_y),
            # Center
            (self.screen_width // 2, self.screen_height // 2),
            # Bottom-left
            (margin_x, self.screen_height - margin_y),
            # Bottom-right
            (self.screen_width - margin_x, self.screen_height - margin_y),
        ]

        return points

    def speak(self, text: str):
        """Speak text."""
        if self.audio_feedback and self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass

    def start_calibration(self):
        """Start calibration."""
        self.current_point_index = 0
        self.samples_collected = 0
        self.calibration_data = {i: [] for i in range(5)}
        self.is_calibrated = False
        self.calibration_quality = 0.0

        self.speak("Starting calibration. Look at the red circle.")

    def add_sample(self, pupil_position: Tuple[int, int]) -> bool:
        """
        Add calibration sample.

        Args:
            pupil_position: (x, y) pupil coordinates

        Returns:
            True if point completed
        """
        if pupil_position is None:
            return False

        region_id = self.current_point_index
        self.calibration_data[region_id].append(pupil_position)
        self.samples_collected += 1

        if self.samples_collected >= self.samples_needed:
            return True

        return False

    def next_point(self) -> bool:
        """
        Move to next point.

        Returns:
            True if calibration complete
        """
        self.current_point_index += 1
        self.samples_collected = 0

        if self.current_point_index >= 5:
            self._compute_calibration_map()
            self.speak("Calibration complete")
            return True

        # Announce next point
        point_names = [
            "top left",
            "top right",
            "center",
            "bottom left",
            "bottom right"
        ]
        self.speak(f"Point {self.current_point_index + 1}. {point_names[self.current_point_index]}")

        return False

    def _compute_calibration_map(self):
        """Compute calibration map."""
        self.calibration_map = {}

        for region_id, samples in self.calibration_data.items():
            if len(samples) > 0:
                avg_x = np.mean([s[0] for s in samples])
                avg_y = np.mean([s[1] for s in samples])
                self.calibration_map[region_id] = (avg_x, avg_y)

        self.is_calibrated = True
        self._compute_calibration_quality()

    def _compute_calibration_quality(self):
        """Compute quality score."""
        if not self.calibration_map:
            self.calibration_quality = 0.0
            return

        total_variance = 0.0
        num_regions = 0

        for region_id, samples in self.calibration_data.items():
            if len(samples) < 5:
                continue

            avg_pos = self.calibration_map[region_id]

            variance_x = np.var([s[0] - avg_pos[0] for s in samples])
            variance_y = np.var([s[1] - avg_pos[1] for s in samples])
            total_variance += (variance_x + variance_y)
            num_regions += 1

        if num_regions == 0:
            self.calibration_quality = 0.0
            return

        avg_variance = total_variance / num_regions
        quality = max(0.0, min(1.0, 1.0 - (avg_variance / 500.0)))
        coverage = num_regions / 5.0
        quality *= coverage

        self.calibration_quality = quality

    def draw_calibration_screen(self, frame: np.ndarray,
                                pupil_position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Draw CVI-optimized calibration screen.

        Args:
            frame: Camera frame
            pupil_position: Current pupil position

        Returns:
            Calibration display
        """
        # Pure black background
        display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # Draw completed points (small white dots)
        for i in range(self.current_point_index):
            cv2.circle(display, self.calibration_points[i], 20, (255, 255, 255), -1)

        # Draw current target with pulsing red animation
        if self.current_point_index < 5:
            current_point = self.calibration_points[self.current_point_index]

            # Pulsing red circle (large for CVI)
            pulse = int(40 * abs(np.sin(time.time() * 2)) + 60)

            # Outer pulsing ring
            cv2.circle(display, current_point, pulse, self.target_color, 8)

            # Inner solid circle
            cv2.circle(display, current_point, 30, self.target_color, -1)

            # White center dot
            cv2.circle(display, current_point, 10, (255, 255, 255), -1)

        # Draw future points (dim gray)
        for i in range(self.current_point_index + 1, 5):
            cv2.circle(display, self.calibration_points[i], 15, (100, 100, 100), 3)

        # Progress bar
        progress = (self.current_point_index * self.samples_needed + self.samples_collected) / (5 * self.samples_needed)
        bar_width = self.screen_width - 200
        bar_height = 40
        bar_x = 100
        bar_y = self.screen_height - 120

        # Background
        cv2.rectangle(display, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)

        # Red fill
        fill_width = int(bar_width * progress)
        cv2.rectangle(display, (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height),
                     (0, 0, 255), -1)

        # White border
        cv2.rectangle(display, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (255, 255, 255), 3)

        # Progress text (LARGE WHITE TEXT)
        progress_text = f"POINT {self.current_point_index + 1} OF 5"
        text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        text_y = 80

        cv2.putText(display, progress_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

        # Samples count
        samples_text = f"{self.samples_collected}/{self.samples_needed}"
        cv2.putText(display, samples_text, (bar_x, bar_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Draw pupil if available
        if pupil_position:
            # Red circle for pupil (CVI friendly)
            cv2.circle(display, pupil_position, 12, (0, 0, 255), 3)
            cv2.circle(display, pupil_position, 4, (255, 255, 255), -1)

        # Instructions (SIMPLE, LARGE TEXT)
        instructions = [
            "LOOK AT RED CIRCLE",
            "HOLD STEADY"
        ]

        y_offset = self.screen_height - 220
        for instruction in instructions:
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (self.screen_width - text_size[0]) // 2

            cv2.putText(display, instruction, (text_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            y_offset += 50

        return display

    def save_calibration(self, student_name: str, calibration_dir: str = "calibrations") -> bool:
        """Save calibration (5-point version)."""
        if not self.is_calibrated:
            return False

        os.makedirs(calibration_dir, exist_ok=True)

        calibration_data = {
            'student_name': student_name,
            'timestamp': datetime.now().isoformat(),
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'calibration_type': 'CVI_5point',
            'calibration_map': {
                str(k): v for k, v in self.calibration_map.items()
            },
            'calibration_quality': self.calibration_quality,
            'calibration_points': self.calibration_points
        }

        filename = os.path.join(calibration_dir, f"{student_name}_cvi_calibration.json")

        try:
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)

            print(f"CVI Calibration saved: {filename}")
            self.speak("Calibration saved")
            return True

        except Exception as e:
            print(f"Error saving: {e}")
            return False

    def load_calibration(self, student_name: str, calibration_dir: str = "calibrations") -> bool:
        """Load CVI calibration."""
        filename = os.path.join(calibration_dir, f"{student_name}_cvi_calibration.json")

        if not os.path.exists(filename):
            print(f"Calibration not found: {filename}")
            return False

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            if (data['screen_width'] != self.screen_width or
                data['screen_height'] != self.screen_height):
                print("Warning: Screen dimensions don't match")
                return False

            self.calibration_map = {
                int(k): tuple(v) for k, v in data['calibration_map'].items()
            }
            self.calibration_quality = data['calibration_quality']
            self.is_calibrated = True

            print(f"CVI Calibration loaded: {student_name}")
            print(f"Quality: {self.calibration_quality:.2%}")

            self.speak(f"Calibration loaded")
            return True

        except Exception as e:
            print(f"Error loading: {e}")
            return False

    def map_pupil_to_region(self, pupil_position: Tuple[int, int]) -> Optional[int]:
        """Map pupil to 5-region."""
        if not self.is_calibrated or pupil_position is None:
            return None

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
        """Get confidence for region."""
        if not self.is_calibrated or region_id not in self.calibration_map:
            return 0.0

        calibrated_pos = self.calibration_map[region_id]

        dx = pupil_position[0] - calibrated_pos[0]
        dy = pupil_position[1] - calibrated_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)

        # More forgiving for CVI students
        confidence = max(0.0, min(1.0, 1.0 - (distance / 350.0)))

        return confidence

    def get_quality_description(self) -> str:
        """Get quality description."""
        if not self.is_calibrated:
            return "Not calibrated"

        if self.calibration_quality >= 0.80:
            return "Excellent"
        elif self.calibration_quality >= 0.65:
            return "Good"
        elif self.calibration_quality >= 0.45:
            return "Fair"
        else:
            return "Poor - Recalibrate"


if __name__ == "__main__":
    """Test CVI calibration."""
    print("="*60)
    print("CVI 5-POINT CALIBRATION TEST")
    print("="*60)

    cal = CVICalibration(screen_width=1280, screen_height=720, audio_feedback=True)

    print(f"\nCalibration points: {len(cal.calibration_points)}")
    print(f"Samples per point: {cal.samples_needed}")
    print(f"Total samples: {len(cal.calibration_points) * cal.samples_needed}")

    print("\nSimulating calibration...")
    cal.start_calibration()

    for point_idx in range(5):
        print(f"\nPoint {point_idx + 1}/5")
        base_x, base_y = cal.calibration_points[point_idx]

        for sample in range(cal.samples_needed):
            offset_x = np.random.randint(-15, 15)
            offset_y = np.random.randint(-15, 15)
            pupil_pos = (base_x + offset_x, base_y + offset_y)

            completed = cal.add_sample(pupil_pos)
            if completed:
                is_done = cal.next_point()
                if is_done:
                    break

    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    print(f"Quality: {cal.calibration_quality:.2%} ({cal.get_quality_description()})")

    # Test save/load
    if cal.save_calibration("cvi_test_student"):
        cal2 = CVICalibration()
        if cal2.load_calibration("cvi_test_student"):
            print(f"Loaded quality: {cal2.calibration_quality:.2%}")

    print("\nCVI calibration test complete!")
