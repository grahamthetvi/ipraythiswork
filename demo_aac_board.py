"""
Demo AAC Communication Board
Practical demonstration of 9-region eye gaze system for AAC

Features:
- 9-region communication board with common phrases
- Text-to-speech output
- Message building
- Visual feedback
- High-contrast, accessible design
- Simple keyboard controls
- Multiple board layouts (greetings, needs, feelings, etc.)

This demonstrates how students can use eye gaze for communication.
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


class AACBoard:
    """
    AAC Communication Board with 9 regions.
    """

    # Predefined board layouts
    BOARDS = {
        'main': {
            'name': 'Main Board',
            'items': [
                'YES', 'NO', 'HELP',
                'I NEED', 'I WANT', 'I FEEL',
                'THANK YOU', 'PLEASE', 'MORE BOARDS'
            ]
        },
        'needs': {
            'name': 'I Need...',
            'items': [
                'WATER', 'FOOD', 'RESTROOM',
                'BREAK', 'HELP', 'QUIET',
                'MOVE', 'ADJUST', 'BACK'
            ]
        },
        'feelings': {
            'name': 'I Feel...',
            'items': [
                'HAPPY', 'SAD', 'TIRED',
                'EXCITED', 'WORRIED', 'CALM',
                'FRUSTRATED', 'COMFORTABLE', 'BACK'
            ]
        },
        'questions': {
            'name': 'Questions',
            'items': [
                'WHAT?', 'WHERE?', 'WHEN?',
                'WHO?', 'WHY?', 'HOW?',
                'CAN I?', 'MAY I?', 'BACK'
            ]
        },
        'social': {
            'name': 'Social',
            'items': [
                'HELLO', 'GOODBYE', 'HOW ARE YOU?',
                'THANK YOU', 'PLEASE', "I'M SORRY",
                'EXCUSE ME', "YOU'RE WELCOME", 'BACK'
            ]
        }
    }

    def __init__(self, tracker: Gaze9Region, enable_tts: bool = True):
        """
        Initialize AAC board.

        Args:
            tracker: Gaze9Region tracker instance
            enable_tts: Enable text-to-speech
        """
        self.tracker = tracker
        self.enable_tts = enable_tts and TTS_AVAILABLE

        # Text-to-speech
        self.tts_engine = None
        if self.enable_tts:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 1.0)
            except:
                self.enable_tts = False
                print("Warning: Could not initialize TTS")

        # Current state
        self.current_board = 'main'
        self.message_buffer: List[str] = []
        self.last_selection_time = 0
        self.selection_cooldown = 0.5  # Prevent double selections

        # Visual settings
        self.screen_width = tracker.screen_width
        self.screen_height = tracker.screen_height

        # Colors (high contrast for accessibility)
        self.bg_color = (40, 40, 40)
        self.button_color = (80, 80, 120)
        self.button_hover_color = (120, 120, 180)
        self.button_active_color = (0, 200, 0)
        self.text_color = (255, 255, 255)
        self.message_bg_color = (20, 20, 60)

    def speak(self, text: str):
        """
        Speak text using TTS.

        Args:
            text: Text to speak
        """
        if self.enable_tts and self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass

    def handle_selection(self, event: GazeEvent):
        """
        Handle region selection event.

        Args:
            event: GazeEvent from gaze tracker
        """
        # Cooldown check
        current_time = time.time()
        if current_time - self.last_selection_time < self.selection_cooldown:
            return

        self.last_selection_time = current_time

        # Get selected item
        board = self.BOARDS[self.current_board]
        selected_item = board['items'][event.region_id]

        print(f"\nSelected: {selected_item}")

        # Handle special items
        if selected_item == 'BACK':
            self.current_board = 'main'
            self.speak("Main board")

        elif selected_item == 'MORE BOARDS':
            # Cycle through boards
            board_names = list(self.BOARDS.keys())
            current_index = board_names.index(self.current_board)
            next_index = (current_index + 1) % len(board_names)
            self.current_board = board_names[next_index]
            self.speak(self.BOARDS[self.current_board]['name'])

        elif selected_item in ['I NEED', 'I WANT', 'I FEEL']:
            # Navigate to specific board
            if selected_item == 'I NEED':
                self.current_board = 'needs'
                self.speak("I need")
            elif selected_item == 'I WANT':
                self.current_board = 'needs'  # Same board for now
                self.speak("I want")
            elif selected_item == 'I FEEL':
                self.current_board = 'feelings'
                self.speak("I feel")

        else:
            # Add to message buffer and speak
            self.message_buffer.append(selected_item)
            self.speak(selected_item)

            # Auto-speak complete sentence if it makes sense
            if len(self.message_buffer) >= 2:
                sentence = ' '.join(self.message_buffer)
                print(f"Message: {sentence}")

    def draw_board(self, region_id: Optional[int] = None,
                  confidence: float = 0.0,
                  dwell_progress: float = 0.0,
                  camera_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw AAC board interface.

        Args:
            region_id: Currently gazed region
            confidence: Confidence score
            dwell_progress: Dwell time progress (0.0-1.0)
            camera_frame: Optional camera frame for pupil preview

        Returns:
            Board display image
        """
        # Create display
        display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        display[:] = self.bg_color

        # Draw message bar at top
        self._draw_message_bar(display)

        # Draw camera feed if provided
        if camera_frame is not None:
            self._draw_camera_preview(display, camera_frame)

        # Calculate board area (below message bar)
        board_y_start = 120
        board_height = self.screen_height - board_y_start - 60
        board_width = self.screen_width

        # Draw 9 buttons (3x3 grid)
        button_w = board_width // 3
        button_h = board_height // 3

        board = self.BOARDS[self.current_board]

        for i in range(9):
            row = i // 3
            col = i % 3

            x1 = col * button_w
            y1 = board_y_start + row * button_h
            x2 = x1 + button_w
            y2 = y1 + button_h

            # Button color based on state
            if i == region_id:
                # Currently gazing at this button
                button_color = self.button_hover_color

                # Add dwell progress indicator
                if dwell_progress > 0:
                    # Progress bar
                    progress_height = int(button_h * dwell_progress)
                    progress_y = y2 - progress_height

                    overlay = display.copy()
                    cv2.rectangle(overlay, (x1 + 5, progress_y),
                                (x2 - 5, y2 - 5), self.button_active_color, -1)
                    cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
            else:
                button_color = self.button_color

            # Draw button background
            cv2.rectangle(display, (x1 + 5, y1 + 5), (x2 - 5, y2 - 5),
                        button_color, -1)

            # Draw button border
            border_color = (200, 200, 200) if i == region_id else (150, 150, 150)
            border_thickness = 4 if i == region_id else 2
            cv2.rectangle(display, (x1 + 5, y1 + 5), (x2 - 5, y2 - 5),
                        border_color, border_thickness)

            # Draw button text (large for accessibility)
            text = board['items'][i]
            self._draw_centered_text(display, text,
                                    (x1 + button_w // 2, y1 + button_h // 2),
                                    font_scale=0.8, thickness=2)

        # Draw board title
        cv2.putText(display, f"Board: {board['name']}", (20, self.screen_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # Draw confidence indicator
        if region_id is not None:
            conf_text = f"Confidence: {confidence:.2f}"
            cv2.putText(display, conf_text, (self.screen_width - 300, self.screen_height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        return display

    def _draw_message_bar(self, display: np.ndarray):
        """Draw message bar showing current message."""
        # Message bar background
        cv2.rectangle(display, (0, 0), (self.screen_width, 100),
                     self.message_bg_color, -1)

        # Message text
        if self.message_buffer:
            message = ' '.join(self.message_buffer)
        else:
            message = "Look at a button to select it..."

        # Draw message (large text for visibility)
        self._draw_centered_text(display, message, (self.screen_width // 2, 50),
                                font_scale=1.2, thickness=3)

        # Clear button indicator
        if self.message_buffer:
            cv2.putText(display, "Press 'C' to clear", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    def _draw_camera_preview(self, display: np.ndarray, camera_frame: np.ndarray):
        """Draw camera feed with pupil detection in top-right corner."""
        # Camera preview size
        preview_w, preview_h = 240, 180

        # Resize camera frame
        camera_small = cv2.resize(camera_frame, (preview_w, preview_h))

        # Draw pupil detection if available
        if len(self.tracker.pupil_history) > 0:
            pupil = self.tracker.pupil_history[-1]
            # Scale pupil coordinates
            scale_x = preview_w / camera_frame.shape[1]
            scale_y = preview_h / camera_frame.shape[0]
            pupil_small = (int(pupil[0] * scale_x), int(pupil[1] * scale_y))

            # Draw pupil indicator
            cv2.circle(camera_small, pupil_small, 8, (0, 255, 0), 2)
            cv2.circle(camera_small, pupil_small, 2, (0, 0, 255), -1)

            # Crosshair
            cv2.line(camera_small, (pupil_small[0] - 12, pupil_small[1]),
                    (pupil_small[0] + 12, pupil_small[1]), (0, 255, 0), 1)
            cv2.line(camera_small, (pupil_small[0], pupil_small[1] - 12),
                    (pupil_small[0], pupil_small[1] + 12), (0, 255, 0), 1)

        # Position in top-right corner (below message bar)
        x_offset = self.screen_width - preview_w - 15
        y_offset = 110

        # Border
        cv2.rectangle(display, (x_offset - 2, y_offset - 2),
                     (x_offset + preview_w + 2, y_offset + preview_h + 2),
                     (150, 150, 150), 2)

        # Overlay camera feed
        display[y_offset:y_offset+preview_h, x_offset:x_offset+preview_w] = camera_small

        # Label
        cv2.putText(display, "Camera", (x_offset, y_offset - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _draw_centered_text(self, display: np.ndarray, text: str,
                           center: tuple, font_scale: float = 1.0,
                           thickness: int = 2):
        """Draw text centered at position."""
        # Handle multi-line text
        if len(text) > 15:
            # Split into multiple lines
            words = text.split()
            lines = []
            current_line = []

            for word in words:
                current_line.append(word)
                line_text = ' '.join(current_line)
                text_size = cv2.getTextSize(line_text, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, thickness)[0]
                if text_size[0] > 250:  # Max width
                    if len(current_line) > 1:
                        current_line.pop()
                        lines.append(' '.join(current_line))
                        current_line = [word]

            if current_line:
                lines.append(' '.join(current_line))

            # Draw lines
            line_height = 30
            start_y = center[1] - (len(lines) - 1) * line_height // 2

            for i, line in enumerate(lines):
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, thickness)[0]
                text_x = center[0] - text_size[0] // 2
                text_y = start_y + i * line_height

                cv2.putText(display, line, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                          self.text_color, thickness)
        else:
            # Single line
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, thickness)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2

            cv2.putText(display, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       self.text_color, thickness)

    def clear_message(self):
        """Clear message buffer."""
        self.message_buffer.clear()
        print("\nMessage cleared")

    def run(self):
        """Run AAC board application."""
        print("\n" + "="*60)
        print("AAC COMMUNICATION BOARD")
        print("="*60)
        print(f"\nTTS Enabled: {self.enable_tts}")
        print(f"Screen: {self.screen_width}x{self.screen_height}")
        print(f"Dwell Time: {self.tracker.dwell_time}s")
        print("\nControls:")
        print("  ESC - Quit")
        print("  'c' - Clear message")
        print("  'b' - Change board")
        print("  '+' - Increase dwell time")
        print("  '-' - Decrease dwell time")
        print("  's' - Speak current message")
        print("\nStarting in 3 seconds...")
        time.sleep(3)

        # Create display window
        cv2.namedWindow('AAC Board', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AAC Board', self.screen_width, self.screen_height)

        # Welcome message
        self.speak("AAC Board ready. Look at a button to select it.")

        running = True

        while running:
            ret, frame = self.tracker.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Detect gaze region
            result = self.tracker.detect_region(frame)

            region_id = None
            confidence = 0.0

            if result is not None:
                region_id, confidence = result

            # Calculate dwell progress
            dwell_progress = 0.0
            if (region_id == self.tracker.current_region and
                self.tracker.region_start_time is not None):
                elapsed = time.time() - self.tracker.region_start_time
                dwell_progress = min(1.0, elapsed / self.tracker.dwell_time)

            # Process dwell time
            gaze_event = self.tracker.process_dwell_time(region_id, confidence)

            if gaze_event is not None:
                self.handle_selection(gaze_event)

            # Update tracker metrics
            self.tracker.update_fps()

            # Draw AAC board
            display = self.draw_board(region_id, confidence, dwell_progress, camera_frame=frame)

            cv2.imshow('AAC Board', display)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                running = False
            elif key == ord('c'):
                self.clear_message()
            elif key == ord('b'):
                # Cycle boards
                board_names = list(self.BOARDS.keys())
                current_index = board_names.index(self.current_board)
                next_index = (current_index + 1) % len(board_names)
                self.current_board = board_names[next_index]
                self.speak(self.BOARDS[self.current_board]['name'])
            elif key == ord('s'):
                # Speak current message
                if self.message_buffer:
                    message = ' '.join(self.message_buffer)
                    self.speak(message)
            elif key == ord('+'):
                self.tracker.dwell_time = min(5.0, self.tracker.dwell_time + 0.1)
                print(f"Dwell time: {self.tracker.dwell_time:.1f}s")
            elif key == ord('-'):
                self.tracker.dwell_time = max(0.5, self.tracker.dwell_time - 0.1)
                print(f"Dwell time: {self.tracker.dwell_time:.1f}s")

        cv2.destroyAllWindows()
        self.speak("Goodbye")

        print("\n" + "="*60)
        print("AAC BOARD CLOSED")
        print("="*60)


def main():
    """Main application."""
    print("="*60)
    print("AAC COMMUNICATION BOARD DEMO")
    print("="*60)
    print("\nThis demonstrates practical use of eye gaze tracking")
    print("for Augmentative and Alternative Communication (AAC)")

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

    # Create tracker
    print("\nInitializing gaze tracker...")
    tracker = Gaze9Region(
        screen_width=1280,
        screen_height=720,
        camera_index=camera_index,
        dwell_time=1.5,  # Slightly longer for AAC to prevent accidents
        confidence_threshold=0.75,
        smoothing_window=5
    )

    if not tracker.start_camera():
        print("✗ Failed to start camera")
        return

    # Check calibration
    print("\n" + "="*60)
    print("CALIBRATION CHECK")
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
            print("\nWarning: Running without calibration (accuracy will be poor)")

    # Create AAC board
    aac_board = AACBoard(tracker, enable_tts=True)

    # Run application
    aac_board.run()

    # Cleanup
    tracker.stop_camera()

    print("\n" + "="*60)
    print("APPLICATION CLOSED")
    print("="*60)


if __name__ == "__main__":
    main()
