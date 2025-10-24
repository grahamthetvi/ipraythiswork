# 9-Region Eye Gaze AAC System - Setup Guide

## For Teachers, TVIs, and Educational Staff

This is a production-ready eye gaze tracking system designed specifically for students with visual impairments who need Augmentative and Alternative Communication (AAC) support.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. [Camera Setup](#camera-setup)
6. [Calibration Guide](#calibration-guide)
7. [Using the AAC Board](#using-the-aac-board)
8. [Testing & Validation](#testing--validation)
9. [Troubleshooting](#troubleshooting)
10. [Accessibility Considerations](#accessibility-considerations)
11. [Best Practices](#best-practices)
12. [Technical Support](#technical-support)

---

## System Overview

### What is this system?

This eye gaze tracking system allows students to:
- Control a computer using only their eyes
- Communicate by selecting words/phrases on an AAC board
- Select from 9 screen regions arranged in a 3x3 grid
- Build and speak messages using text-to-speech

### Key Features

- **9-Region Grid**: Simple 3x3 layout that's easy to learn
- **Multiple Detection Methods**: Automatic fallback ensures reliability
- **Calibration System**: Personalized for each student
- **High-Contrast Design**: Optimized for low vision
- **Audio Feedback**: Speaks selections aloud
- **Dwell Time Selection**: Look at region to select (no clicking needed)
- **Performance Metrics**: Monitor accuracy and speed

### Technical Specifications

- **Accuracy Target**: 85%+ after calibration
- **Frame Rate**: 30+ FPS
- **Latency**: <100ms from eye movement to detection
- **Camera Support**: Standard webcam (minimum), IR camera (optimal)

---

## System Requirements

### Hardware Requirements

**Minimum:**
- Computer with Windows 10/11, macOS 10.14+, or Linux
- Intel Core i5 or equivalent
- 8GB RAM
- USB webcam (720p minimum)
- Display: 1280x720 minimum resolution

**Recommended:**
- Intel Core i7 or equivalent
- 16GB RAM
- IR camera (e.g., The Eye Tribe, Tobii Eye Tracker)
- Display: 1920x1080 resolution
- External speakers for better audio feedback

### Software Requirements

- Python 3.8 or higher
- OpenCV (cv2)
- NumPy
- pyttsx3 (for text-to-speech)
- GazeTracking library (optional but recommended)

---

## Installation

### Step 1: Install Python

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer
3. **IMPORTANT**: Check "Add Python to PATH"
4. Click "Install Now"

**macOS:**
```bash
brew install python3
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip
```

### Step 2: Install Dependencies

Open terminal/command prompt and run:

```bash
# Core dependencies
pip install opencv-python numpy pyttsx3

# Optional: GazeTracking library (recommended)
git clone https://github.com/antoinelame/GazeTracking.git
cd GazeTracking
pip install -r requirements.txt
# Copy gaze_tracking folder to your project directory
```

### Step 3: Download the System

```bash
git clone [repository-url]
cd ipraythiswork
```

### Step 4: Test Installation

```bash
python gaze_9region.py
```

If you see the tracker window, installation is successful!

---

## Quick Start Guide

### First-Time Setup (30 minutes)

1. **Connect Camera**
   - Plug in USB webcam or IR camera
   - Position at eye level, 18-24 inches from student

2. **Run Demo AAC Board**
   ```bash
   python demo_aac_board.py
   ```

3. **Calibrate for Student**
   - When prompted, start calibration
   - Student looks at 9 yellow circles (one at a time)
   - Takes about 5 minutes
   - Save calibration with student's name

4. **Test Communication**
   - Student looks at buttons to select them
   - System speaks selections aloud
   - Build messages by selecting multiple words

### Daily Use (5 minutes)

1. **Start System**
   ```bash
   python demo_aac_board.py
   ```

2. **Load Student Profile**
   - Select student's name from list
   - System loads their calibration

3. **Begin Communication**
   - Student uses eye gaze to select phrases
   - System speaks messages aloud

---

## Camera Setup

### Camera Positioning

**Critical for Success!**

```
                    [CAMERA]
                       |
                   18-24 inches
                       |
                 [STUDENT'S EYES]
                       |
                   12-18 inches
                       |
                    [SCREEN]
```

### Positioning Guidelines

1. **Height**: Camera should be at eye level with student
2. **Distance**: 18-24 inches from student's face
3. **Angle**: Camera points straight at student (not tilted)
4. **Lighting**:
   - Avoid backlighting (windows behind student)
   - Use diffuse, even lighting
   - No direct light in student's eyes or on screen

### Camera Settings

For standard webcam:
- **Brightness**: Medium (auto-adjust enabled)
- **Exposure**: Auto or manual -5
- **Resolution**: 640x480 (good balance of speed/quality)
- **Frame Rate**: 30 FPS

For IR camera:
- Follow manufacturer's guidelines
- Usually requires specific drivers
- Often provides better tracking in varied lighting

### Testing Camera Position

Run the test program:
```bash
python gazetracking_wrapper.py
```

You should see:
- Clear view of both eyes
- Pupils detected (green circles)
- Smooth tracking as eyes move

**Adjust if:**
- Pupils not detected: Move camera closer
- Jittery tracking: Improve lighting
- No face detected: Check camera angle

---

## Calibration Guide

### Why Calibrate?

Each student's eyes are different. Calibration teaches the system where YOUR student looks for each region.

### When to Calibrate

- **First time** using system
- **Weekly** for best accuracy
- **Whenever** accuracy drops
- **After** changing camera position
- **If** student's seating position changes

### Calibration Process

#### Preparation (5 minutes)

1. Position student comfortably
2. Adjust camera (see Camera Setup)
3. Check lighting
4. Ensure student can see screen clearly
5. Explain process to student

#### Running Calibration (5-10 minutes)

1. **Start Calibration**
   ```bash
   python demo_aac_board.py
   # When prompted, choose "Run calibration"
   ```

2. **Follow Yellow Circle**
   - Large yellow circle appears
   - Audio says location: "Point 1. Top left"
   - Student looks at circle
   - Hold gaze steady (3 seconds)
   - Circle moves automatically

3. **Complete All 9 Points**
   - Top row: left, center, right
   - Middle row: left, center, right
   - Bottom row: left, center, right

4. **Review Quality**
   - System shows calibration quality
   - **Excellent** (85%+): Perfect!
   - **Good** (70-84%): Acceptable
   - **Fair** (50-69%): Try again
   - **Poor** (<50%): Check camera/lighting

5. **Save Profile**
   - Enter student's name
   - System saves calibration
   - Can reload anytime

#### Tips for Successful Calibration

**For Students:**
- Sit still and comfortable
- Look directly at yellow circle
- Hold gaze steady (don't move eyes)
- Blink normally between points
- Take breaks if needed

**For Staff:**
- Give verbal encouragement
- Use audio cues ("Look at the circle")
- Monitor student's attention
- Repeat if student loses focus
- Multiple short sessions better than one long session

### Calibration Quality Indicators

| Quality | Accuracy | Action |
|---------|----------|--------|
| Excellent | 85-100% | Perfect! Use system |
| Good | 70-84% | Acceptable, monitor performance |
| Fair | 50-69% | Recalibrate recommended |
| Poor | <50% | Must recalibrate or check setup |

---

## Using the AAC Board

### Starting the AAC Board

```bash
python demo_aac_board.py
```

### Interface Overview

The screen shows:
- **9 large buttons** in 3x3 grid
- **Message bar** at top (shows selected words)
- **Current board name** at bottom
- **Confidence indicator** (shows tracking quality)

### How to Select

1. **Look** at desired button
2. **Hold gaze** for dwell time (default 1.5 seconds)
3. **Progress indicator** shows circular timer
4. **Speaks** selection when timer completes
5. **Adds** to message bar

### Available Boards

**Main Board:**
- YES, NO, HELP
- I NEED, I WANT, I FEEL
- THANK YOU, PLEASE, MORE BOARDS

**I Need... Board:**
- WATER, FOOD, RESTROOM
- BREAK, HELP, QUIET
- MOVE, ADJUST, BACK

**I Feel... Board:**
- HAPPY, SAD, TIRED
- EXCITED, WORRIED, CALM
- FRUSTRATED, COMFORTABLE, BACK

**Questions Board:**
- WHAT?, WHERE?, WHEN?
- WHO?, WHY?, HOW?
- CAN I?, MAY I?, BACK

**Social Board:**
- HELLO, GOODBYE, HOW ARE YOU?
- THANK YOU, PLEASE, I'M SORRY
- EXCUSE ME, YOU'RE WELCOME, BACK

### Keyboard Controls

| Key | Action |
|-----|--------|
| ESC | Quit application |
| C | Clear current message |
| B | Change board |
| S | Speak current message again |
| + | Increase dwell time (slower) |
| - | Decrease dwell time (faster) |

### Building Messages

1. Select words in sequence
2. Message builds in top bar
3. Each word is spoken
4. Complete sentence is shown
5. Press 'S' to repeat message
6. Press 'C' to start new message

**Example:**
1. Look at "I NEED"
2. Look at "HELP"
3. Message shows: "I NEED HELP"
4. System speaks: "I need help"

### Adjusting Dwell Time

**Too fast?** (accidental selections)
- Press '+' to increase time
- Recommended: 1.5-2.0 seconds

**Too slow?** (student gets frustrated)
- Press '-' to decrease time
- Minimum: 0.5 seconds

**Finding the right time:**
- Start with 1.5 seconds
- Observe student
- Adjust based on their control

---

## Testing & Validation

### Why Test?

Testing ensures the system is working accurately before daily use.

### Running Accuracy Test

```bash
python gaze_tester.py
# Select option 1: Run Accuracy Test
```

### What It Tests

- **Per-Region Accuracy**: How well each region is detected
- **Overall Accuracy**: System-wide performance
- **Latency**: Speed of detection
- **Confidence**: Reliability of detections

### Understanding Results

**Target Accuracy: 85%+**

| Result | Meaning | Action |
|--------|---------|--------|
| 85-100% | Excellent | System ready for use |
| 75-84% | Good | Acceptable, monitor |
| 60-74% | Fair | Recalibrate recommended |
| <60% | Poor | Must troubleshoot |

### Performance Benchmark

```bash
python gaze_tester.py
# Select option 2: Run Performance Benchmark
```

Tests:
- **FPS** (frames per second): Should be 30+
- **Latency**: Should be <100ms
- **Detection Rate**: Should be 90%+

### Exporting Test Results

Results can be exported to CSV for:
- Progress monitoring
- IEP documentation
- Troubleshooting
- Research/analysis

Files generated:
- `gaze_test_results_[date].csv` - Detailed results
- `gaze_test_summary_[date].csv` - Summary statistics

---

## Troubleshooting

### Problem: Low Accuracy (<70%)

**Possible Causes & Solutions:**

1. **Poor Calibration**
   - Solution: Recalibrate (see Calibration Guide)
   - Check: Student was focused during calibration

2. **Camera Position Changed**
   - Solution: Reposition camera
   - Check: Camera at eye level, 18-24" away

3. **Lighting Issues**
   - Solution: Improve lighting
   - Check: No backlighting, even diffuse light
   - Check: No glare on student's glasses

4. **Student Position Changed**
   - Solution: Reposition student or recalibrate
   - Check: Student sitting upright, eyes level with camera

### Problem: Pupil Not Detected

**Solutions:**

1. **Adjust Brightness**
   - Press 'B' to brighten
   - Try different lighting

2. **Move Camera Closer**
   - Optimal: 18-24 inches from face

3. **Check Camera Settings**
   - Run: `python gazetracking_wrapper.py`
   - Adjust exposure with 'B' and 'D' keys

4. **Try Different Tracker**
   - System auto-selects best method
   - GazeTracking usually best
   - Hough Circle good for IR cameras

### Problem: Jittery/Unstable Tracking

**Solutions:**

1. **Increase Smoothing**
   - Edit `gaze_9region.py`
   - Change `smoothing_window=5` to `smoothing_window=10`

2. **Improve Lighting**
   - Use diffuse, even lighting
   - Avoid shadows on face

3. **Stabilize Camera**
   - Mount camera securely
   - Avoid vibration

### Problem: Accidental Selections

**Solutions:**

1. **Increase Dwell Time**
   - Press '+' key
   - Recommended: 1.5-2.0 seconds

2. **Increase Confidence Threshold**
   - Edit `gaze_9region.py`
   - Change `confidence_threshold=0.75` to `0.85`

### Problem: System Too Slow

**Solutions:**

1. **Lower Resolution**
   - Edit camera settings to 640x480

2. **Close Other Programs**
   - Free up CPU resources

3. **Reduce Smoothing**
   - Lower `smoothing_window` value

### Problem: No Audio/TTS Not Working

**Solutions:**

1. **Install pyttsx3**
   ```bash
   pip install pyttsx3
   ```

2. **Check System Audio**
   - Verify speakers connected
   - Check volume settings

3. **Test TTS Separately**
   ```python
   import pyttsx3
   engine = pyttsx3.init()
   engine.say("Hello")
   engine.runAndWait()
   ```

### Problem: Camera Not Found

**Solutions:**

1. **Check Connection**
   - Replug USB cable
   - Try different USB port

2. **Check Camera Index**
   - Edit code: Change `camera_index=0` to `1` or `2`

3. **List Available Cameras**
   - Run: `python gazetracking_wrapper.py`
   - Shows all detected cameras

### Getting Help

If problems persist:

1. **Check Logs**
   - Look for error messages in terminal
   - Note which component fails

2. **Test Components**
   ```bash
   # Test camera
   python gazetracking_wrapper.py

   # Test calibration
   python calibration_system.py

   # Test tracker
   python gaze_9region.py
   ```

3. **Document Issue**
   - Camera model
   - Error messages
   - When problem occurs
   - Steps already tried

4. **Contact Support**
   - Include test results
   - Include system specifications
   - Include screenshots if possible

---

## Accessibility Considerations

### Visual Impairments

This system is specifically designed for students with visual impairments:

**Design Features:**
- **High Contrast**: Yellow on black, white text
- **Large Buttons**: Easy to see and target
- **Large Text**: 1.2x-2.0x normal size
- **Audio Feedback**: All selections spoken aloud
- **Progress Indicators**: Visual dwell time display
- **Adjustable**: All settings can be modified

### Customization Options

**For Low Vision:**
- Increase button size (edit `demo_aac_board.py`)
- Change color scheme (high contrast options)
- Increase text size
- Enable screen magnification

**For Photosensitivity:**
- Reduce animation speed
- Disable pulsing effects
- Use solid colors

**For Motor Control:**
- Increase dwell time (more time to hold gaze)
- Increase smoothing (reduce jitter)
- Larger target areas

### Cognitive Load Considerations

**Simplify Interface:**
- Start with Main board only
- Add boards gradually
- Use familiar words/phrases
- Consistent layout

**Reduce Distractions:**
- Minimize background elements
- Clear, simple design
- Focus on communication goals

### Physical Positioning

**Wheelchair Users:**
- Adjustable camera mount essential
- Consider mounting options
- Test multiple positions

**Limited Head Control:**
- Secure head position (if safe)
- Adjust camera to student
- May need custom mount

---

## Best Practices

### Daily Use Routine

**Start of Session (5 min):**
1. Position student comfortably
2. Check camera position
3. Load student's calibration
4. Quick accuracy check
5. Adjust dwell time if needed

**During Session:**
- Monitor student's fatigue
- Take breaks every 15-20 minutes
- Watch for tracking quality indicators
- Encourage and support student

**End of Session:**
- Save any new calibrations
- Note any issues for next time
- Clean camera lens if needed

### Maintenance

**Daily:**
- Check camera position
- Verify calibration loads correctly
- Monitor tracking quality

**Weekly:**
- Recalibrate for best accuracy
- Test system performance
- Clean camera lens
- Update software if needed

**Monthly:**
- Export test results for records
- Review student progress
- Check for software updates
- Backup calibration profiles

### Student Training

**Week 1: Introduction**
- Explain eye gaze tracking
- Demo with simple selections
- Practice calibration process
- Build comfort with system

**Week 2: Basic Communication**
- Practice Main board
- Build simple messages
- Learn YES/NO/HELP
- Short sessions (10-15 min)

**Week 3: Expanding Skills**
- Introduce new boards
- Practice message building
- Increase session length
- Encourage independence

**Week 4+: Mastery**
- Full board access
- Complex message building
- Natural conversation flow
- Student-led communication

### Documentation

**Keep Records:**
- Calibration dates and quality
- Accuracy test results
- Dwell time settings
- Student progress notes
- Issues and solutions

**IEP Goals:**
- Use test data for objective measures
- Export CSV results
- Track improvement over time
- Document accommodations

---

## Technical Support

### System Files

**Core Components:**
- `gaze_9region.py` - Main tracker
- `calibration_system.py` - Calibration
- `demo_aac_board.py` - AAC application
- `gaze_tester.py` - Testing tool

**Support Files:**
- `gazetracking_wrapper.py` - GazeTracking integration
- `hough_tracker.py` - Fallback tracker
- `dlib_tracker.py` - Alternative tracker

### Configuration Files

**Calibration Profiles:**
- Located in: `calibrations/` folder
- Format: `[student_name]_calibration.json`
- Backup regularly!

**Test Results:**
- Format: `gaze_test_results_[date].csv`
- Format: `gaze_test_summary_[date].csv`
- Save for records

### System Requirements Verification

```bash
# Check Python version
python --version  # Should be 3.8+

# Check installed packages
pip list | grep opencv
pip list | grep numpy
pip list | grep pyttsx3

# Test camera
python -c "import cv2; print(cv2.__version__)"
```

### Performance Optimization

**For Better FPS:**
- Lower camera resolution
- Close unnecessary programs
- Use wired camera (not wireless)
- Reduce smoothing window

**For Better Accuracy:**
- Better lighting
- IR camera if possible
- Frequent recalibration
- Proper camera positioning

**For Lower Latency:**
- Reduce buffer size
- Faster computer
- Close background apps
- Use USB 3.0 port

### Updates and Maintenance

**Check for Updates:**
```bash
git pull
pip install --upgrade opencv-python numpy pyttsx3
```

**Backup Important Files:**
- Calibration profiles
- Test results
- Modified settings

### Additional Resources

**Documentation:**
- This README
- Code comments in each file
- GazeTracking library docs

**Community:**
- GitHub issues for bug reports
- Share configurations
- Request features

**Training:**
- Video tutorials (if available)
- Sample sessions
- Best practices guides

---

## Contact Information

For technical support, feature requests, or questions:

- **GitHub Issues**: [Repository URL]/issues
- **Email**: [Support email if available]
- **Documentation**: This README and code comments

---

## License and Credits

**Eye Gaze Tracking System for AAC**
- Designed for students with visual impairments
- Production-ready and tested
- Open source

**Credits:**
- GazeTracking library by Antoine Lamé
- OpenCV community
- AAC professionals and educators

**License:**
- [Your license here]

---

## Quick Reference

### Common Commands

```bash
# Start AAC Board
python demo_aac_board.py

# Run accuracy test
python gaze_tester.py

# Test camera
python gazetracking_wrapper.py

# Run calibration only
python calibration_system.py
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| ESC | Exit/Quit |
| C | Clear message |
| B | Change board |
| S | Speak message |
| + | Increase dwell time |
| - | Decrease dwell time |

### Recommended Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| Dwell Time | 1.5s | Balance speed/accuracy |
| Confidence | 0.75 | Good reliability |
| Smoothing | 5 | Reduce jitter |
| Camera Distance | 18-24" | Optimal tracking |
| Resolution | 640x480 | Good performance |

---

## Appendix: Technical Details

### Detection Methods

1. **GazeTracking** (Primary)
   - Uses facial landmarks
   - Best overall accuracy
   - Requires good lighting

2. **Hough Circle** (Fallback)
   - Detects circular pupils
   - Great with IR cameras
   - Fast and robust

3. **Auto-Detection**
   - System chooses best method
   - Automatic fallback
   - Transparent to user

### Performance Targets

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Accuracy | 85%+ | 75%+ | <75% |
| FPS | 30+ | 20+ | <20 |
| Latency | <100ms | <150ms | >150ms |
| Detection | 90%+ | 80%+ | <80% |

### Calibration Algorithm

- Collects 30 samples per region
- Averages pupil positions
- Creates mapping function
- Computes quality score
- Saves to JSON profile

### Region Mapping

```
Region Layout (3x3 grid):
┌───┬───┬───┐
│ 0 │ 1 │ 2 │  (Top row)
├───┼───┼───┤
│ 3 │ 4 │ 5 │  (Middle row)
├───┼───┼───┤
│ 6 │ 7 │ 8 │  (Bottom row)
└───┴───┴───┘
```

---

**Last Updated**: [Date]
**Version**: 1.0
**For**: Teachers, TVIs, Educational Staff
**Purpose**: AAC for Students with Visual Impairments
