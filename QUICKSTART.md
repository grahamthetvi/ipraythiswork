# Quick Start Guide - Eye Tracking with GUIs

## Step-by-Step Workflow

### Step 1: Test Pupil Detection (Live Preview)

**Run the live preview to test your camera and lighting:**

```bash
python preview_pupil_detection.py
```

**What you'll see:**
- Live camera feed
- Green circle and crosshair on detected pupil
- Detection rate percentage
- Real-time feedback

**Controls:**
- `+` - Increase sensitivity (if pupil not detected)
- `-` - Decrease sensitivity (if false detections)
- `R` - Reset to default
- `ESC` - Exit

**Goal:** Get 70%+ detection rate before proceeding to calibration.

**Tips:**
- Adjust room lighting (diffuse, not too bright)
- Position camera 18-24 inches from face
- Remove glasses if glare is an issue
- Try different camera if detection poor (use `camera_selector.py`)

---

### Step 2: Calibrate with Visual GUI

**Once pupil detection is working well, run calibration:**

```bash
python calibrate_with_gui.py
```

**What you'll see:**

1. **Two windows**:
   - `Calibration` - Fullscreen with red target points
   - `Pupil Detection` - Camera feed showing pupil tracking

2. **Calibration process**:
   - Red targets appear one at a time
   - Look directly at each red target
   - Green progress bar shows collection progress
   - System collects 40 samples per point

3. **After calibration**:
   - Option to save calibration profile
   - Saved as `.json` file for reuse

**Configuration:**
- Screen dimensions (1920x1080 default)
- Number of points:
  - `5 points` - Faster, CVI-friendly
  - `9 points` - More accurate (recommended)
- Camera index (0 or 1)

---

### Step 3: Test Accuracy

**Run accuracy test to measure how well calibration worked:**

```bash
python accuracy_tester.py
```

**What you'll see:**

1. **Fullscreen test targets**:
   - Red circles at precise locations
   - Progress bar
   - Live gaze point (green dot)

2. **Results**:
   - Mean error in pixels
   - Accuracy percentage (±100px)
   - Success rate

**Good results:**
- Mean error < 100 pixels ✓
- Accuracy > 70% ✓
- Success rate > 85% ✓

**Poor results?**
- Recalibrate more carefully
- Try different camera
- Improve lighting
- Run comparison test (next step)

---

### Step 4: Compare Cameras (Optional but Recommended)

**If you have multiple cameras (webcam + IR camera), compare them:**

```bash
python tracker_comparison.py
```

**Configuration:**
- Test Hough Circle: Yes
- Camera index first test: 0 (webcam)
- Camera index second test: 1 (IR camera)

**What you'll see:**
- Same tests for each camera
- Side-by-side comparison table
- Ranked by performance

**Example output:**
```
TRACKER COMPARISON RESULTS
================================================================
Tracker Name              Mean Error  Accuracy  Success Rate
----------------------------------------------------------------
Hough Circle (IR cam)        78.2      79.3%     91.2%     #1
Hough Circle (webcam)       124.5      61.8%     84.5%     #2
================================================================
```

**Result:** Now you know which camera works best!

---

### Step 5: View Detailed Results (Optional)

**Analyze test results with visualizations:**

```bash
python view_test_results.py
```

**Visualizations:**
1. **Error Vectors** - See target vs measured gaze
2. **Heat Map** - Spatial accuracy distribution (red=bad, blue=good)
3. **Distribution** - Histogram of errors
4. **By Region** - Accuracy across 3×3 screen grid

**Use this to:**
- Understand where tracker is less accurate
- Identify if certain screen regions are problematic
- Compare multiple test sessions

---

## Visual Guide

### Preview Tool (Step 1)
```
┌─────────────────────────────────────┐
│  Pupil Detection Preview            │
│                                     │
│  ╭─────────────────╮                │
│  │                 │                │
│  │      ◉ ← Green  │  Your eye      │
│  │    ╋╋╋  circle  │                │
│  │                 │                │
│  ╰─────────────────╯                │
│                                     │
│  Detection rate: 87.3%              │
│  param2: 30 (sensitivity)           │
│                                     │
└─────────────────────────────────────┘
```

### Calibration Tool (Step 2)
```
Main Screen (Fullscreen):          Pupil Detection Window:
┌──────────────────────┐           ┌─────────────────────┐
│                      │           │  ╭──────────────╮   │
│   ●                  │           │  │     ◉        │   │
│         ●            │           │  │   ╋╋╋        │   │
│             ●        │           │  ╰──────────────╯   │
│                      │           │  Point 3/9          │
│   ●     ⊗     ●      │ ← Current │  Samples: 25/40     │
│                      │   target  │  Pupil: (124, 203)  │
│   ●     ●     ●      │           └─────────────────────┘
│                      │
│ ▓▓▓▓▓▓▓▓░░░░░░       │ ← Progress
└──────────────────────┘
```

### Accuracy Test (Step 3)
```
┌──────────────────────────────┐
│                              │
│   ◉  ← Target (red)          │
│     ·  ← Your gaze (green)   │
│                              │
│          ◉                   │
│           ·                  │
│                              │
│  ▓▓▓▓▓▓░░░░░░░                │ ← Progress
│  Point 2/9 | Samples: 18/30  │
└──────────────────────────────┘
```

---

## Troubleshooting

### "Pupil: Not detected" in preview

**Try:**
1. Press `+` to increase sensitivity
2. Improve lighting (add diffuse light source)
3. Adjust camera angle (should point at face)
4. Move closer/farther from camera (18-24" ideal)
5. Remove glasses if causing glare
6. Try different camera with `camera_selector.py`

### Calibration collects very slowly

**Causes:**
- Pupil detection poor (see above)
- Looking away from target
- Head moving

**Solutions:**
- Ensure preview shows >70% detection rate first
- Focus directly on red target
- Keep head still (use headrest if needed)

### High error in accuracy test (>150px)

**Causes:**
- Poor calibration
- User moved after calibration
- Wrong camera
- Incorrect screen dimensions

**Solutions:**
- Recalibrate more carefully
- Stay in same position
- Verify screen dimensions correct
- Try comparison test with different camera

### One corner of screen has high error

**This is normal!** Accuracy typically decreases toward edges.

**Solutions:**
- Use more calibration points (9 instead of 5)
- Add extra calibration points in corners
- For AAC, place important items toward screen center

---

## Complete Example Session

```bash
# 1. Test pupil detection
$ python preview_pupil_detection.py
# Camera index: 0
# Detection rate: 85.2% ✓ Good!
# ESC to exit

# 2. Calibrate
$ python calibrate_with_gui.py
# Screen: 1920x1080
# Points: 9
# Camera: 0
# [Look at each red target]
# Save calibration? y
# Filename: my_calibration.json

# 3. Test accuracy
$ python accuracy_tester.py
# Screen: 1920x1080
# Points: 9
# Tracker: 1 (Hough Circle)
# Camera: 0
# Load calibration? y
# File: my_calibration.json
# [Look at targets]
# Mean error: 87.3 pixels ✓
# Accuracy: 74.2% ✓
# Success rate: 89.1% ✓

# 4. Compare with IR camera
$ python tracker_comparison.py
# [Test camera 0, then camera 1]
# Winner: Camera 0 (webcam) with 87.3px error

# 5. Ready to use!
# Use camera 0 for your AAC application
```

---

## Next Steps

### For AAC System:

**CVI-Optimized (High contrast, 5 regions):**
```bash
python cvi_aac_board.py
```

**Standard (9 regions):**
```bash
python demo_aac_board.py
```

Both support loading your calibration profile!

### For Custom Application:

```python
from trackers.hough_tracker_gaze import HoughGazeTracker

# Initialize
tracker = HoughGazeTracker(1920, 1080)
tracker.start(camera_index=0)

# Load your calibration
tracker.load_calibration("my_calibration.json")

# Get gaze coordinates
gaze = tracker.get_gaze_point()
if gaze:
    x, y = gaze.x, gaze.y
    print(f"Looking at: ({x}, {y})")
```

---

## Files Overview

### GUI Tools (Use These!)
- `preview_pupil_detection.py` - Test camera and pupil detection
- `calibrate_with_gui.py` - Visual calibration with feedback
- `accuracy_tester.py` - Test accuracy with visual targets
- `tracker_comparison.py` - Compare multiple systems
- `view_test_results.py` - Visualize test results

### Utilities
- `camera_selector.py` - Find best camera
- `test_install.py` - Verify installation

### AAC Applications
- `cvi_aac_board.py` - CVI-optimized AAC
- `demo_aac_board.py` - Standard AAC

### Documentation
- `README_TESTING.md` - Complete framework docs
- `QUICKSTART.md` - This file
- `README_CVI.md` - CVI-specific info
- `INSTALL_WINDOWS.md` - Windows setup

---

## Tips for Best Results

1. **Lighting**: Diffuse, even lighting. Not too bright, not too dim.

2. **Camera position**: Eye level, 18-24 inches away, centered on face.

3. **Head position**: Stay still during calibration and use. Consider headrest.

4. **Calibration**: Look directly at CENTER of each target. Take your time.

5. **Multiple cameras**: If you have both webcam and IR camera, test both!

6. **Save calibrations**: Save your calibration profile and reuse it.

7. **Recalibrate**: If you move camera or change position, recalibrate.

8. **Test first**: Always run preview before calibration to verify detection.

---

## Questions?

- Framework details: See `README_TESTING.md`
- CVI requirements: See `README_CVI.md`
- Installation: See `INSTALL_WINDOWS.md`
- Issues: Check troubleshooting section above

**You're ready to go! Start with Step 1 (preview) and work through the steps.**
