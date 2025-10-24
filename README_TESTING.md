# Eye Tracking Testing Framework

## Overview

This is a **unified testing framework** for comparing multiple eye tracking systems' ability to accurately map eye gaze to screen coordinates.

**Primary Goal**: Determine which eye tracking system works best with your specific hardware (webcam vs IR camera) by objectively measuring gaze-to-screen coordinate accuracy.

---

## The Problem This Solves

Traditional eye tracking implementations often:
- Detect pupils but don't map to screen coordinates accurately
- Work differently with webcams vs IR cameras
- Lack objective accuracy measurements
- Make it hard to compare different tracking systems

This framework provides:
- ✅ **Standardized testing** - Same test points for all trackers
- ✅ **Objective metrics** - Mean error, accuracy percentages, success rates
- ✅ **Visual validation** - See exactly where the tracker thinks you're looking
- ✅ **Comparative analysis** - Direct side-by-side comparisons
- ✅ **Calibration quality** - Perspective transformation for accurate gaze-to-screen mapping

---

## Architecture

### Core Components

```
eye_tracker_framework.py
    ↓ (defines interface)
├── trackers/hough_tracker_gaze.py      (OpenCV-based, any camera)
├── trackers/webgazer_tracker.py        (Browser-based, webcam)
└── trackers/pupil_labs_tracker.py      (Professional, special hardware)
    ↓ (tested by)
├── accuracy_tester.py                  (Single tracker testing)
├── tracker_comparison.py               (Multi-tracker comparison)
└── view_test_results.py                (Results visualization)
```

### Abstract Base Class: `EyeTrackerBase`

All trackers implement this interface:

```python
class EyeTrackerBase(abc.ABC):
    @abstractmethod
    def start(self, camera_index: int = 0) -> bool:
        """Start the eye tracker."""

    @abstractmethod
    def stop(self):
        """Stop the eye tracker."""

    @abstractmethod
    def get_gaze_point(self) -> Optional[GazePoint]:
        """Get current gaze point on screen (x, y coordinates)."""

    @abstractmethod
    def calibrate(self, calibration_points: List[Tuple[int, int]]) -> bool:
        """Calibrate using screen coordinates."""

    @abstractmethod
    def get_calibration_quality(self) -> float:
        """Get calibration quality score (0.0-1.0)."""
```

**Key Improvement**: All trackers return actual **screen coordinates** (x, y pixels), not just regions. This is achieved through perspective transformation calibration.

---

## Available Eye Trackers

### 1. Hough Circle Tracker (`hough_tracker_gaze.py`)

**Technology**: OpenCV Hough Circle Transform + Perspective transformation

**Pros**:
- ✅ Works with **any webcam or IR camera**
- ✅ No special software required (just OpenCV)
- ✅ No compilation needed (pure Python)
- ✅ Fast startup

**Cons**:
- ❌ Requires good pupil detection (lighting dependent)
- ❌ Moderate accuracy (~50-150 pixel error typical)
- ❌ Can lose tracking if pupils not clearly visible

**Hardware**: Standard webcam or IR camera

**Setup**:
```python
from trackers.hough_tracker_gaze import HoughGazeTracker

tracker = HoughGazeTracker(screen_width=1920, screen_height=1080)
tracker.start(camera_index=0)  # 0=default, 1=second camera
tracker.calibrate(calibration_points)  # 5-9 points
gaze = tracker.get_gaze_point()  # Returns (x, y) coordinates
```

**Calibration**:
- Uses `cv2.findHomography()` to compute perspective transformation
- Maps pupil position in camera frame → screen coordinates
- Requires 4+ calibration points (9 recommended)

---

### 2. WebGazer Tracker (`webgazer_tracker.py`)

**Technology**: WebGazer.js (browser-based ML eye tracking)

**Pros**:
- ✅ No installation (runs in browser)
- ✅ Machine learning-based
- ✅ Works with standard webcam
- ✅ Self-calibrating with use

**Cons**:
- ❌ Requires browser window open
- ❌ JavaScript must be enabled
- ❌ Camera must be accessible to browser
- ❌ Network latency (Flask bridge to Python)

**Hardware**: Webcam (accessed via browser)

**Setup**:
```python
from trackers.webgazer_tracker import WebGazerTracker

tracker = WebGazerTracker(screen_width=1920, screen_height=1080)
tracker.start()  # Opens browser to localhost:5000
# Complete calibration in browser (9-point)
# Browser sends gaze data to Python via Flask
gaze = tracker.get_gaze_point()
```

**Architecture**:
- Flask server runs on `localhost:5000`
- Browser loads `templates/webgazer.html`
- JavaScript tracks gaze with WebGazer.js
- Gaze coordinates sent to Python via POST requests
- Python receives coordinates via `/gaze` endpoint

---

### 3. Pupil Labs Tracker (`pupil_labs_tracker.py`)

**Technology**: Pupil Labs Core (research-grade eye tracking)

**Pros**:
- ✅ **High accuracy** (research/clinical grade)
- ✅ Dedicated eye cameras
- ✅ Professional calibration
- ✅ Rich gaze data

**Cons**:
- ❌ **Expensive** (~$2000+ for hardware)
- ❌ Requires Pupil Capture software running
- ❌ Special hardware setup
- ❌ More complex workflow

**Hardware**: Pupil Labs Core or Invisible eye tracking glasses

**Setup**:
```python
from trackers.pupil_labs_tracker import PupilLabsTracker

tracker = PupilLabsTracker(screen_width=1920, screen_height=1080)
tracker.start()  # Connects to Pupil Capture via ZMQ
# Calibrate in Pupil Capture software
gaze = tracker.get_gaze_point()  # Normalized coordinates from Pupil
```

**Requirements**:
- Pupil Capture software running
- ZMQ connection to `localhost:50020`
- Calibration done in Pupil Capture UI

---

## Testing Tools

### 1. Accuracy Tester (`accuracy_tester.py`)

**Purpose**: Test a single tracker's accuracy

**What it does**:
1. Displays visual targets at precise screen coordinates
2. User looks at each target
3. Collects 30 gaze samples per target
4. Calculates error distance (pixels from target)
5. Generates accuracy metrics

**Usage**:
```bash
python accuracy_tester.py
```

**Interactive prompts**:
- Screen dimensions
- Number of test points (5/9/13)
- Tracker selection
- Camera index (for Hough Circle)
- Calibration

**Output**:
- CSV file: `accuracy_test_[tracker]_[timestamp].csv`
- Console: Mean/median error, accuracy %, success rate
- Optional: Visual error vectors

**Metrics**:
- **Mean Error**: Average distance from target (pixels)
- **Median Error**: Middle value of error distribution
- **Std Dev**: Error consistency
- **Accuracy (±100px)**: % of samples within 100 pixels
- **Success Rate**: % of valid gaze measurements

---

### 2. Tracker Comparison (`tracker_comparison.py`)

**Purpose**: Compare multiple trackers side-by-side

**What it does**:
1. Tests multiple trackers with **identical test points**
2. Runs same accuracy test for each
3. Generates comparative statistics
4. Ranks trackers by performance

**Usage**:
```bash
python tracker_comparison.py
```

**Interactive workflow**:
1. Select which trackers to test
2. Configure each tracker
3. Calibrate each tracker
4. Run identical accuracy tests
5. View comparison table
6. Save results

**Output**:
- CSV file: `tracker_comparison_[timestamp].csv`
- Console: Comparison table with rankings
- Optional: Bar chart visualization

**Example Output**:
```
TRACKER COMPARISON RESULTS
========================================================================
Tracker Name              Mean Error  Median  Std Dev  Accuracy  Success Rate
                          (pixels)    (px)    (px)     (±100px)  (%)
------------------------------------------------------------------------
Pupil Labs Core              42.3     38.1     15.2      95.2%     98.5%     #1
Hough Circle (Calibrated)    87.6     79.3     28.7      72.4%     89.1%     #2
WebGazer                    124.8    118.5     45.3      58.7%     82.3%     #3
========================================================================

🏆 BEST PERFORMER: Pupil Labs Core
   Mean error: 42.3 pixels
   Accuracy (±100px): 95.2%
```

---

### 3. Results Viewer (`view_test_results.py`)

**Purpose**: Visualize test results from CSV files

**What it does**:
- Loads accuracy test or comparison CSV files
- Generates multiple visualizations
- Provides statistical summaries

**Usage**:
```bash
python view_test_results.py
```

**Visualizations**:

1. **Error Vectors**
   - Shows targets (red) vs measured gaze (green)
   - Yellow arrows show error direction/magnitude
   - See spatial accuracy patterns

2. **Error Heat Map**
   - Color-coded spatial accuracy
   - Red = high error areas
   - Blue = low error areas
   - Identifies problematic screen regions

3. **Error Distribution Histogram**
   - Shows how errors are distributed
   - Compare multiple trackers
   - See if errors are consistent or variable

4. **Accuracy by Region**
   - Divides screen into 3×3 grid
   - Shows mean error per region
   - Identifies if tracker works better in certain areas

---

## Complete Testing Workflow

### Step 1: Setup

1. **Install dependencies**:
   ```bash
   pip install opencv-python numpy pyttsx3
   # For WebGazer: pip install flask
   # For Pupil Labs: pip install pyzmq msgpack
   ```

2. **Test installation**:
   ```bash
   python test_install.py
   ```

3. **Select best camera** (if using Hough Circle):
   ```bash
   python camera_selector.py
   ```
   - Tests all available cameras
   - Shows live pupil detection
   - Recommends best camera index

---

### Step 2: Single Tracker Test

Test one tracker to validate it works:

```bash
python accuracy_tester.py
```

**Example session** (Hough Circle with webcam):
1. Enter screen dimensions: `1920 x 1080`
2. Select test points: `9`
3. Select tracker: `1` (Hough Circle)
4. Camera index: `0`
5. Calibrate: `y`
6. Look at 9 calibration points (30 samples each)
7. Look at 9 test points (30 samples each)
8. View results summary
9. Optionally visualize error vectors

**Expected output**:
```
TEST SUMMARY
============================================================
Error Distance (pixels):
  Mean:   87.6
  Median: 79.3
  Std:    28.7
  Min:    45.2
  Max:    156.8

Accuracy (±100px):
  Mean:   72.4%

Valid measurements: 9/9 (100.0%)
============================================================

✓ Results saved to: accuracy_test_Hough_Circle_20250124_143022.csv
```

---

### Step 3: Multi-Tracker Comparison

Compare multiple trackers:

```bash
python tracker_comparison.py
```

**Recommended comparison**:
1. Test **Hough Circle with webcam** (camera_index=0)
2. Test **Hough Circle with IR camera** (camera_index=1)
3. Optionally test WebGazer
4. Compare results

This directly answers: **"Which camera works better with Hough Circle?"**

**Tips**:
- Use same number of test points for fair comparison
- Complete calibration carefully for each tracker
- Take breaks between tests (avoid fatigue)
- Run tests in consistent lighting conditions

---

### Step 4: Analyze Results

```bash
python view_test_results.py
```

Load the CSV files from previous tests and visualize:

1. **Error vectors** - See where tracker is off
2. **Heat map** - Find problematic screen areas
3. **Distribution** - Understand error patterns
4. **By region** - See if accuracy varies by location

**Questions to answer**:
- Is the tracker consistently off in one direction? → Check calibration
- High error in screen corners? → Normal, may need more calibration points
- Random scattered errors? → Poor pupil detection, try better camera/lighting
- One region terrible? → May need region-specific calibration

---

## Interpreting Results

### Good Performance

**Hough Circle (baseline)**:
- Mean error: < 100 pixels
- Accuracy (±100px): > 70%
- Success rate: > 85%

**WebGazer**:
- Mean error: < 120 pixels
- Accuracy (±100px): > 60%
- Success rate: > 80%

**Pupil Labs** (professional):
- Mean error: < 50 pixels
- Accuracy (±100px): > 90%
- Success rate: > 95%

### Poor Performance Indicators

- ❌ Mean error > 200 pixels → Likely calibration issue or wrong camera
- ❌ Success rate < 70% → Poor pupil detection, change lighting or camera
- ❌ High std dev (> 50px) → Unstable tracking, needs better hardware or lighting
- ❌ Accuracy < 50% → System not usable for AAC, troubleshoot

---

## Troubleshooting

### Issue: High Mean Error (>150px)

**Possible causes**:
1. Poor calibration
2. User moved after calibration
3. Wrong camera selected
4. Screen dimensions incorrect

**Solutions**:
- Recalibrate more carefully
- Ensure user stays still
- Try different camera (`camera_selector.py`)
- Verify screen dimensions match actual display

---

### Issue: Low Success Rate (<80%)

**Possible causes**:
1. Pupil detection failing
2. Poor lighting
3. Glasses causing glare
4. Camera position/angle

**Solutions**:
- Run `camera_selector.py` to verify pupil detection
- Improve lighting (diffuse, even, not too bright)
- Remove glasses or adjust angle
- Position camera at eye level, 18-24 inches away

---

### Issue: Inconsistent Results (High Std Dev)

**Possible causes**:
1. User not looking directly at targets
2. Head movement during test
3. Variable pupil detection
4. Camera auto-adjust interfering

**Solutions**:
- Use verbal cues to direct attention
- Stabilize head position (headrest if needed)
- Lock camera settings (disable auto-exposure, auto-focus)
- Collect more samples per point

---

### Issue: One Tracker Much Worse

**This is expected!** Different systems have different strengths:

- **Hough Circle**: Sensitive to lighting and camera quality
  - Try IR camera if webcam fails
  - Adjust lighting
  - Check pupil detection with `camera_selector.py`

- **WebGazer**: Needs webcam with good resolution
  - Ensure camera accessible to browser
  - Complete full calibration (all 9 points)
  - Let it adapt (accuracy improves with use)

- **Pupil Labs**: Requires proper hardware setup
  - Verify Pupil Capture running
  - Complete calibration in Pupil Capture first
  - Check ZMQ connection

---

## Next Steps After Testing

### Once you've identified the best tracker:

1. **For AAC System**: Integrate best tracker with AAC boards
   - Use `cvi_aac_board.py` (CVI-optimized, 5 regions)
   - Or `demo_aac_board.py` (standard, 9 regions)
   - Both support pluggable trackers via `EyeTrackerBase`

2. **For Custom Application**: Import the tracker
   ```python
   from trackers.hough_tracker_gaze import HoughGazeTracker

   tracker = HoughGazeTracker(1920, 1080)
   tracker.start(camera_index=0)
   tracker.calibrate(points)

   while True:
       gaze = tracker.get_gaze_point()
       if gaze:
           x, y = gaze.x, gaze.y
           # Use coordinates for your application
   ```

3. **For Research**: Use CSV data for analysis
   - Compare conditions (lighting, distance, etc.)
   - Validate custom tracker implementations
   - Document system performance

---

## Advanced: Adding New Trackers

To add a new eye tracking system:

1. **Create tracker file**: `trackers/my_tracker.py`

2. **Inherit from base class**:
   ```python
   from eye_tracker_framework import EyeTrackerBase, GazePoint

   class MyTracker(EyeTrackerBase):
       def __init__(self, screen_width, screen_height):
           super().__init__(screen_width, screen_height)
           self.tracker_name = "My Custom Tracker"

       def start(self, camera_index=0) -> bool:
           # Initialize your tracker
           return True

       def stop(self):
           # Cleanup
           pass

       def get_gaze_point(self) -> Optional[GazePoint]:
           # Return gaze coordinates
           return GazePoint(x=..., y=..., confidence=..., timestamp=...)

       def calibrate(self, calibration_points) -> bool:
           # Implement calibration
           return True

       def get_calibration_quality(self) -> float:
           return 0.8
   ```

3. **Test with framework**:
   ```python
   from trackers.my_tracker import MyTracker
   from accuracy_tester import AccuracyTester

   tracker = MyTracker(1920, 1080)
   tracker.start()
   tracker.calibrate(points)

   tester = AccuracyTester(1920, 1080)
   results = tester.run_accuracy_test(tracker, num_points=9)
   ```

4. **Add to comparison**: Edit `tracker_comparison.py` to include your tracker

---

## Key Concepts

### Perspective Transformation Calibration

**Problem**: Pupil position in camera frame ≠ gaze position on screen

**Solution**: Perspective transformation matrix

```python
# During calibration:
# Collect pupil positions (camera coords) for known screen positions
src_points = [(pupil_x, pupil_y), ...]  # Camera coordinates
dst_points = [(screen_x, screen_y), ...]  # Screen coordinates

# Compute transformation matrix
matrix, _ = cv2.findHomography(src_points, dst_points)

# During use:
# Map pupil to screen
pupil_point = np.array([[pupil_x, pupil_y]])
screen_point = cv2.perspectiveTransform(pupil_point.reshape(1,1,2), matrix)
screen_x, screen_y = screen_point[0][0]
```

This is **critical** for accurate gaze-to-screen mapping. Without it, you only get pupil position in camera frame, which doesn't translate to screen coordinates.

---

### Accuracy Metrics

**Mean Error**: Average Euclidean distance from target
```python
error = sqrt((gaze_x - target_x)² + (gaze_y - target_y)²)
mean_error = average(all_errors)
```

**Accuracy Percentage**: Samples within tolerance
```python
accurate = (error <= 100 pixels)
accuracy = (accurate_samples / total_samples) * 100
```

**Success Rate**: Valid measurements
```python
success_rate = (valid_samples / total_attempts) * 100
```

**Lower mean error = Better**
**Higher accuracy % = Better**
**Higher success rate = More reliable**

---

## File Reference

### Core Framework
- `eye_tracker_framework.py` - Abstract base class and data structures
- `trackers/hough_tracker_gaze.py` - Hough Circle with perspective transform
- `trackers/webgazer_tracker.py` - WebGazer.js bridge
- `trackers/pupil_labs_tracker.py` - Pupil Labs ZMQ integration
- `templates/webgazer.html` - WebGazer frontend

### Testing Tools
- `accuracy_tester.py` - Single tracker accuracy testing
- `tracker_comparison.py` - Multi-tracker comparison
- `view_test_results.py` - Results visualization

### Utilities
- `camera_selector.py` - Interactive camera selection
- `test_install.py` - Verify installation

### AAC Applications
- `cvi_aac_board.py` - CVI-optimized AAC (5 regions)
- `demo_aac_board.py` - Standard AAC (9 regions)

### Documentation
- `README_TESTING.md` - This file
- `README_CVI.md` - CVI-specific information
- `README_SETUP.md` - General setup guide
- `INSTALL_WINDOWS.md` - Windows installation

---

## Summary

This testing framework provides:

1. **Unified interface** - All trackers work the same way
2. **Objective testing** - Measure actual accuracy, not guesses
3. **Direct comparison** - See which system works best for your hardware
4. **Visual validation** - See exactly where tracking fails
5. **Production-ready** - Integrate best tracker into AAC or custom apps

**The key innovation**: Perspective transformation calibration that maps pupil position to accurate screen coordinates, solving the "eye tracking to eye gaze to screen pinpoint location" problem.

**Your next step**: Run `tracker_comparison.py` to test Hough Circle with both your webcam and IR camera, then use the best performing system for your AAC application.

---

**Version**: 1.0
**Last Updated**: 2025-01-24
**For**: Unified eye tracking testing and comparison
