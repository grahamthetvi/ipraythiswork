"""
CVI-Optimized AAC Communication Board
Designed specifically for students with Cortical Visual Impairment (CVI)

CVI Design Requirements:
- High contrast: Black background, white bold text
- Red bubbling/outlines for visual attention
- Simple, large icons with red outlines
- Maximum 5 items per screen
- Minimal visual clutter
- Large target areas
- Movement/animation to attract attention

Features:
- 5-region eye gaze grid (instead of 9)
- Pulsing red borders for attention
- Simple icon support
- Large, clear buttons
- CVI-friendly color scheme
"""

import cv2
import numpy as np
import time
from typing import Optional, List
from datetime import datetime

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Install for audio: pip install pyttsx3")

from gaze_9region import Gaze9Region, GazeEvent


class CVIAACBoard:
    """
    CVI-Optimized AAC Communication Board.

    5-region layout designed for students with Cortical Visual Impairment.
    """

    # CVI-optimized board layouts (max 5 items)
    BOARDS = {
        'main': {
            'name': 'Main Board',
            'items': [
                ('YES', '✓'),
                ('NO', '✗'),
                ('HELP', '!'),
                ('MORE', '→'),
                ('I NEED', '☝')
            ]
        },
        'needs': {
            'name': 'I Need',
            'items': [
                ('WATER', '💧'),
                ('FOOD', '🍎'),
                ('BATHROOM', '🚻'),
                ('BREAK', '⏸'),
                ('BACK', '←')
            ]
        },
        'feelings': {
            'name': 'How I Feel',
            'items': [
                ('HAPPY', '😊'),
                ('SAD', '😢'),
                ('TIRED', '😴'),
                ('OKAY', '👍'),
                ('BACK', '←')
            ]
        }
    }

    # CVI Color Scheme
    COLORS = {
        'background': (0, 0, 0),          # Pure black
        'text': (255, 255, 255),          # Pure white
        'button_bg': (30, 30, 30),        # Very dark gray
        'button_hover': (50, 50, 50),     # Dark gray
        'red_outline': (0, 0, 255),       # Pure red (BGR format)
        'red_pulse': (0, 0, 200),         # Darker red for pulse
        'message_bar': (20, 20, 20),      # Almost black
    }

    def __init__(self, tracker: Gaze9Region, enable_tts: bool = True):
        """
        Initialize CVI-optimized AAC board.

        Args:
            tracker: Gaze9Region tracker (will be adapted for 5 regions)
            enable_tts: Enable text-to-speech
        """
        self.tracker = tracker
        self.enable_tts = enable_tts and TTS_AVAILABLE

        # Text-to-speech
        self.tts_engine = None
        if self.enable_tts:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 140)  # Slower for clarity
                self.tts_engine.setProperty('volume', 1.0)
            except:
                self.enable_tts = False
                print("Warning: Could not initialize TTS")

        # Current state
        self.current_board = 'main'
        self.message_buffer: List[str] = []
        self.last_selection_time = 0
        self.selection_cooldown = 1.0  # Longer cooldown for CVI

        # Visual settings
        self.screen_width = tracker.screen_width
        self.screen_height = tracker.screen_height

        # 5-region layout (adjusted from 9-region tracker)
        self.num_regions = 5
        self.region_map = self._create_5region_map()

        # Animation
        self.animation_time = time.time()

    def _create_5region_map(self):
        """
        Map 9-region gaze positions to 5-region layout.

        Layout:
        ┌─────┬─────┬─────┐
        │  0  │     │  1  │  Top row: 0, 1
        ├─────┼─────┼─────┤
        │     │  2  │     │  Middle: center (2)
        ├─────┼─────┼─────┤
        │  3  │     │  4  │  Bottom row: 3, 4
        └─────┴─────┴─────┘

        9-region to 5-region mapping:
        9-region: 0 1 2 3 4 5 6 7 8
        5-region: 0 1 1 0 2 1 3 3 4
        """
        # Map 9-region indices to 5-region indices
        mapping = {
            0: 0,  # Top-left -> Region 0
            1: 1,  # Top-center -> Region 1
            2: 1,  # Top-right -> Region 1
            3: 0,  # Middle-left -> Region 0
            4: 2,  # Center -> Region 2
            5: 1,  # Middle-right -> Region 1
            6: 3,  # Bottom-left -> Region 3
            7: 3,  # Bottom-center -> Region 3
            8: 4,  # Bottom-right -> Region 4
        }
        return mapping

    def map_gaze_to_5region(self, gaze_event: GazeEvent) -> Optional[int]:
        """
        Map 9-region gaze event to 5-region layout.

        Args:
            gaze_event: Event from 9-region tracker

        Returns:
            5-region index (0-4)
        """
        if gaze_event.region_id in self.region_map:
            return self.region_map[gaze_event.region_id]
        return None

    def speak(self, text: str):
        """Speak text using TTS."""
        if self.enable_tts and self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass

    def handle_selection(self, region_5: int):
        """
        Handle 5-region selection.

        Args:
            region_5: Selected region (0-4)
        """
        # Cooldown check
        current_time = time.time()
        if current_time - self.last_selection_time < self.selection_cooldown:
            return

        self.last_selection_time = current_time

        # Get current board
        board = self.BOARDS[self.current_board]

        # Check if region is valid
        if region_5 >= len(board['items']):
            return

        # Get selected item
        selected_text, icon = board['items'][region_5]

        print(f"\nSelected: {selected_text}")

        # Handle special items
        if selected_text == 'BACK':
            self.current_board = 'main'
            self.speak("Main board")

        elif selected_text == 'MORE':
            # Cycle through boards
            board_names = list(self.BOARDS.keys())
            current_index = board_names.index(self.current_board)
            next_index = (current_index + 1) % len(board_names)
            self.current_board = board_names[next_index]
            self.speak(self.BOARDS[self.current_board]['name'])

        elif selected_text == 'I NEED':
            self.current_board = 'needs'
            self.speak("I need")

        else:
            # Add to message buffer and speak
            self.message_buffer.append(selected_text)
            self.speak(selected_text)

            # Show complete message
            if len(self.message_buffer) >= 1:
                sentence = ' '.join(self.message_buffer)
                print(f"Message: {sentence}")

    def get_pulsing_red(self) -> tuple:
        """
        Get pulsing red color for CVI attention.

        Returns:
            BGR color tuple with pulsing intensity
        """
        # Pulse between 180 and 255 (red channel)
        pulse = int(180 + 75 * abs(np.sin(time.time() * 2)))
        return (0, 0, pulse)  # BGR format

    def draw_board(self, region_9: Optional[int] = None,
                  confidence: float = 0.0,
                  dwell_progress: float = 0.0,
                  camera_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw CVI-optimized AAC board.

        Args:
            region_9: Current 9-region gaze (will be mapped to 5)
            confidence: Confidence score
            dwell_progress: Dwell time progress (0.0-1.0)
            camera_frame: Optional camera frame

        Returns:
            Board display image
        """
        # Map 9-region to 5-region
        region_5 = None
        if region_9 is not None and region_9 in self.region_map:
            region_5 = self.region_map[region_9]

        # Create pure black background
        display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # Draw message bar
        self._draw_message_bar(display)

        # Get current board
        board = self.BOARDS[self.current_board]
        num_items = len(board['items'])

        # Calculate button positions (5-region layout)
        positions = self._get_5region_positions()

        # Draw each button
        for i in range(min(num_items, 5)):
            text, icon = board['items'][i]
            x1, y1, x2, y2 = positions[i]

            # Determine if this button is being gazed at
            is_gazing = (i == region_5)

            # Draw button
            self._draw_cvi_button(display, text, icon, x1, y1, x2, y2,
                                is_gazing, dwell_progress if is_gazing else 0.0)

        # Draw board title
        cv2.putText(display, board['name'].upper(), (30, self.screen_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.COLORS['text'], 3)

        # Draw camera preview if provided
        if camera_frame is not None:
            self._draw_camera_preview(display, camera_frame)

        return display

    def _get_5region_positions(self) -> List[tuple]:
        """
        Get button positions for 5-region layout.

        Returns:
            List of (x1, y1, x2, y2) tuples for each region
        """
        # Layout:
        # [0]         [1]
        #      [2]
        # [3]         [4]

        margin = 40
        message_height = 140
        footer_height = 80

        available_height = self.screen_height - message_height - footer_height
        available_width = self.screen_width - 2 * margin

        # Button size (LARGE for CVI)
        button_w = (available_width - margin) // 2
        button_h = (available_height - 2 * margin) // 3

        positions = []

        # Region 0: Top-left
        x1 = margin
        y1 = message_height + margin
        positions.append((x1, y1, x1 + button_w, y1 + button_h))

        # Region 1: Top-right
        x1 = margin + button_w + margin
        y1 = message_height + margin
        positions.append((x1, y1, x1 + button_w, y1 + button_h))

        # Region 2: Center
        x1 = margin + button_w // 2
        y1 = message_height + margin + button_h + margin
        positions.append((x1, y1, x1 + button_w, y1 + button_h))

        # Region 3: Bottom-left
        x1 = margin
        y1 = message_height + margin + 2 * (button_h + margin)
        positions.append((x1, y1, x1 + button_w, y1 + button_h))

        # Region 4: Bottom-right
        x1 = margin + button_w + margin
        y1 = message_height + margin + 2 * (button_h + margin)
        positions.append((x1, y1, x1 + button_w, y1 + button_h))

        return positions

    def _draw_cvi_button(self, display: np.ndarray, text: str, icon: str,
                        x1: int, y1: int, x2: int, y2: int,
                        is_gazing: bool, dwell_progress: float):
        """
        Draw CVI-optimized button with red bubbling.

        Args:
            display: Display image
            text: Button text
            icon: Simple icon/emoji
            x1, y1, x2, y2: Button coordinates
            is_gazing: Whether user is gazing at this button
            dwell_progress: Dwell time progress (0.0-1.0)
        """
        button_w = x2 - x1
        button_h = y2 - y1

        # Button background (dark gray)
        bg_color = self.COLORS['button_hover'] if is_gazing else self.COLORS['button_bg']
        cv2.rectangle(display, (x1, y1), (x2, y2), bg_color, -1)

        # RED BUBBLING BORDER (pulsing for attention)
        if is_gazing:
            # Thick pulsing red border
            pulse_color = self.get_pulsing_red()
            border_thickness = int(8 + 6 * abs(np.sin(time.time() * 4)))
        else:
            # Static red border
            pulse_color = self.COLORS['red_outline']
            border_thickness = 6

        cv2.rectangle(display, (x1, y1), (x2, y2), pulse_color, border_thickness)

        # Dwell progress indicator (red fill from bottom)
        if is_gazing and dwell_progress > 0:
            progress_height = int(button_h * dwell_progress)
            progress_y = y2 - progress_height

            overlay = display.copy()
            cv2.rectangle(overlay, (x1 + 10, progress_y),
                        (x2 - 10, y2 - 10), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        # Draw LARGE icon at top (CVI needs simple, large visuals)
        icon_y = y1 + button_h // 3

        # Use larger font for icon
        cv2.putText(display, icon, (x1 + button_w // 2 - 40, icon_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, self.COLORS['text'], 6)

        # Draw WHITE BOLD text at bottom
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 4)[0]
        text_x = x1 + (button_w - text_size[0]) // 2
        text_y = y2 - button_h // 4

        # Draw text with outline for extra contrast
        # Black outline
        cv2.putText(display, text, (text_x - 2, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        # White text
        cv2.putText(display, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.COLORS['text'], 4)

    def _draw_message_bar(self, display: np.ndarray):
        """Draw message bar with high contrast."""
        # Pure black background for message bar
        cv2.rectangle(display, (0, 0), (self.screen_width, 120),
                     self.COLORS['background'], -1)

        # White border
        cv2.rectangle(display, (0, 0), (self.screen_width, 120),
                     self.COLORS['text'], 3)

        # Message text (LARGE and WHITE)
        if self.message_buffer:
            message = ' '.join(self.message_buffer)
        else:
            message = "LOOK AT BUTTON TO SELECT"

        # Center text
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        text_y = 70

        # Draw with outline for maximum contrast
        cv2.putText(display, message, (text_x - 2, text_y - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
        cv2.putText(display, message, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.COLORS['text'], 4)

    def _draw_camera_preview(self, display: np.ndarray, camera_frame: np.ndarray):
        """Draw small camera preview (CVI-optimized)."""
        preview_w, preview_h = 200, 150

        # Resize camera frame
        camera_small = cv2.resize(camera_frame, (preview_w, preview_h))

        # Draw pupil detection if available
        if len(self.tracker.pupil_history) > 0:
            pupil = self.tracker.pupil_history[-1]
            scale_x = preview_w / camera_frame.shape[1]
            scale_y = preview_h / camera_frame.shape[0]
            pupil_small = (int(pupil[0] * scale_x), int(pupil[1] * scale_y))

            # RED circle for CVI
            cv2.circle(camera_small, pupil_small, 10, self.COLORS['red_outline'], 3)
            cv2.circle(camera_small, pupil_small, 2, self.COLORS['text'], -1)

        # Position in top-right corner
        x_offset = self.screen_width - preview_w - 20
        y_offset = 130

        # RED border
        cv2.rectangle(display, (x_offset - 3, y_offset - 3),
                     (x_offset + preview_w + 3, y_offset + preview_h + 3),
                     self.COLORS['red_outline'], 3)

        # Overlay
        display[y_offset:y_offset+preview_h, x_offset:x_offset+preview_w] = camera_small

    def clear_message(self):
        """Clear message buffer."""
        self.message_buffer.clear()
        print("\nMessage cleared")
        self.speak("Cleared")

    def run(self):
        """Run CVI-optimized AAC board."""
        print("\n" + "="*60)
        print("CVI-OPTIMIZED AAC COMMUNICATION BOARD")
        print("="*60)
        print("\nDesigned for students with Cortical Visual Impairment")
        print("Features:")
        print("  • High contrast (black background, white text)")
        print("  • Red bubbling borders for visual attention")
        print("  • 5 large buttons (max)")
        print("  • Simple icons")
        print("  • Minimal visual clutter")
        print(f"\nTTS Enabled: {self.enable_tts}")
        print(f"Dwell Time: {self.tracker.dwell_time}s")
        print("\nControls:")
        print("  ESC - Quit")
        print("  'c' - Clear message")
        print("  '+' - Increase dwell time")
        print("  '-' - Decrease dwell time")
        print("\nStarting in 3 seconds...")
        time.sleep(3)

        # Create window
        cv2.namedWindow('CVI AAC Board', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CVI AAC Board', self.screen_width, self.screen_height)

        # Welcome message
        self.speak("AAC Board ready")

        running = True

        while running:
            ret, frame = self.tracker.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Detect gaze region (9-region from tracker)
            result = self.tracker.detect_region(frame)

            region_9 = None
            confidence = 0.0

            if result is not None:
                region_9, confidence = result

            # Calculate dwell progress
            dwell_progress = 0.0
            if (region_9 == self.tracker.current_region and
                self.tracker.region_start_time is not None):
                elapsed = time.time() - self.tracker.region_start_time
                dwell_progress = min(1.0, elapsed / self.tracker.dwell_time)

            # Process dwell time (using 9-region tracker)
            gaze_event = self.tracker.process_dwell_time(region_9, confidence)

            if gaze_event is not None:
                # Map to 5-region and handle
                region_5 = self.map_gaze_to_5region(gaze_event)
                if region_5 is not None:
                    self.handle_selection(region_5)

            # Update metrics
            self.tracker.update_fps()

            # Draw CVI-optimized board
            display = self.draw_board(region_9, confidence, dwell_progress, camera_frame=frame)

            cv2.imshow('CVI AAC Board', display)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                running = False
            elif key == ord('c'):
                self.clear_message()
            elif key == ord('+'):
                self.tracker.dwell_time = min(5.0, self.tracker.dwell_time + 0.2)
                print(f"Dwell time: {self.tracker.dwell_time:.1f}s")
            elif key == ord('-'):
                self.tracker.dwell_time = max(0.8, self.tracker.dwell_time - 0.2)
                print(f"Dwell time: {self.tracker.dwell_time:.1f}s")

        cv2.destroyAllWindows()
        self.speak("Goodbye")

        print("\n" + "="*60)
        print("CVI AAC BOARD CLOSED")
        print("="*60)


def main():
    """Main application for CVI AAC board."""
    print("="*60)
    print("CVI-OPTIMIZED AAC BOARD")
    print("="*60)
    print("\nDesigned specifically for students with")
    print("Cortical Visual Impairment (CVI)")

    # Camera selection
    print("\n" + "="*60)
    print("CAMERA SELECTION")
    print("="*60)

    select_camera = input("\nSelect camera interactively? (y/n, default=n): ").strip().lower()
    camera_index = 0

    if select_camera == 'y':
        from camera_selector import CameraSelector
        selector = CameraSelector()
        selected = selector.run_interactive_selection()
        if selected is not None:
            camera_index = selected
        else:
            print("\nUsing default camera 0")

    # Create tracker (using 9-region tracker but will map to 5)
    print("\nInitializing gaze tracker...")
    tracker = Gaze9Region(
        screen_width=1280,
        screen_height=720,
        camera_index=camera_index,
        dwell_time=2.0,  # Longer for CVI students
        confidence_threshold=0.70,  # Slightly lower for accessibility
        smoothing_window=7  # More smoothing for stability
    )

    if not tracker.start_camera():
        print("✗ Failed to start camera")
        return

    # Calibration
    print("\n" + "="*60)
    print("CALIBRATION")
    print("="*60)

    students = tracker.calibration.list_calibrations()

    if students:
        print("\nAvailable calibrations:")
        for i, name in enumerate(students):
            print(f"  {i+1}. {name}")

        choice = input("\nLoad calibration? Enter name or number (or press Enter to skip): ").strip()

        if choice:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(students):
                    tracker.calibration.load_calibration(students[idx])
            except:
                tracker.calibration.load_calibration(choice)

    if not tracker.calibration.is_calibrated:
        print("\nNo calibration loaded.")
        calibrate = input("Run calibration now? (y/n): ").strip().lower()

        if calibrate == 'y':
            tracker._run_calibration_interactive()

            save = input("\nSave calibration? (y/n): ").strip().lower()
            if save == 'y':
                name = input("Student name: ").strip()
                if name:
                    tracker.calibration.save_calibration(name)
        else:
            print("\nWarning: Running without calibration")

    # Create and run CVI AAC board
    cvi_board = CVIAACBoard(tracker, enable_tts=True)
    cvi_board.run()

    # Cleanup
    tracker.stop_camera()

    print("\n" + "="*60)
    print("CVI AAC SYSTEM SHUTDOWN")
    print("="*60)


if __name__ == "__main__":
    main()
