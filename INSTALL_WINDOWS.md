# Installation Guide for Windows

## Quick Install (No C++ Compiler Needed)

This guide gets you running the CVI AAC system **without** needing to install dlib or Visual Studio Build Tools.

---

## Step 1: Install Python

1. Download Python 3.9 or later from https://www.python.org/downloads/
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   ```

---

## Step 2: Install Core Dependencies

Open Command Prompt and run:

```cmd
pip install opencv-python numpy pyttsx3
```

That's it! These three packages are all you need for the CVI system.

---

## Step 3: Download the Code

```cmd
git clone [your-repo-url]
cd ipraythiswork
```

Or download ZIP from GitHub and extract.

---

## Step 4: Run the CVI AAC System

```cmd
python cvi_aac_board.py
```

The system will use **Hough Circle tracking** (built into OpenCV) which works great and doesn't need dlib!

---

## Troubleshooting

### "opencv-python not found"

```cmd
pip install --upgrade pip
pip install opencv-python
```

### "pyttsx3 not working"

```cmd
pip install --upgrade pyttsx3
```

If TTS still doesn't work, the system will run without audio (visual only).

### Camera not detected

- Make sure your webcam is plugged in
- Close other programs using the camera (Zoom, Teams, etc.)
- Try running camera_selector.py first

---

## About Dlib/GazeTracking (Optional)

The error you saw is because **dlib requires a C++ compiler** which most users don't have.

**Good news**: The CVI AAC system **doesn't need dlib**! It uses:
- **Hough Circle Tracker** (built into OpenCV)
- Works perfectly for CVI students
- No compilation needed
- Faster installation

### If You Really Want GazeTracking (Advanced)

Only follow this if you specifically want the GazeTracking library:

1. **Install Visual Studio Build Tools**
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Select "Desktop development with C++"
   - Size: ~6GB

2. **Install CMake**
   - Download from: https://cmake.org/download/
   - Add to PATH

3. **Install dlib from pre-built wheel** (easier):
   ```cmd
   pip install dlib-binary
   ```

4. **Or compile from source** (if wheel doesn't work):
   ```cmd
   pip install dlib
   ```

But again, **this is not needed** for the CVI system!

---

## Verify Installation

Test that everything works:

```cmd
# Test camera selector
python camera_selector.py

# Test CVI AAC board
python cvi_aac_board.py
```

You should see:
- Camera feed with pupil detection (red circles)
- No errors about missing dlib

---

## System Check

Run this to verify your setup:

```python
# test_install.py
import sys
print(f"Python: {sys.version}")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except:
    print("✗ OpenCV not installed - run: pip install opencv-python")

try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__}")
except:
    print("✗ NumPy not installed - run: pip install numpy")

try:
    import pyttsx3
    print("✓ pyttsx3: installed")
except:
    print("⚠ pyttsx3 not installed - audio won't work (visual will work)")

try:
    import dlib
    print(f"✓ dlib: installed (optional)")
except:
    print("⚠ dlib not installed (not needed for CVI system)")

print("\nYour system is ready if OpenCV and NumPy show ✓")
```

Save as `test_install.py` and run:
```cmd
python test_install.py
```

---

## What You Need vs What's Optional

### Required (Must Have):
- ✅ Python 3.8+
- ✅ opencv-python
- ✅ numpy
- ✅ pyttsx3 (for audio, optional)

### Optional (Nice to Have):
- ⚪ dlib (only if you want GazeTracking library)
- ⚪ GazeTracking library (not needed for CVI system)

The CVI AAC system works great with just the required packages!

---

## Summary

**To get running quickly:**

```cmd
# 1. Install dependencies (no compiler needed)
pip install opencv-python numpy pyttsx3

# 2. Run the system
python cvi_aac_board.py
```

**That's it!** No need to fight with dlib/CMake/Visual Studio.

---

## Still Getting Errors?

### Common Issues:

**"pip is not recognized"**
- Reinstall Python, check "Add to PATH"
- Or use: `python -m pip install opencv-python numpy pyttsx3`

**"Permission denied"**
- Run Command Prompt as Administrator
- Or use: `pip install --user opencv-python numpy pyttsx3`

**"No module named cv2"**
- Make sure you installed `opencv-python` not `opencv-contrib-python`
- Try: `pip uninstall opencv-python` then `pip install opencv-python`

**Camera not working**
- Check Device Manager (Windows key + X → Device Manager → Cameras)
- Try different camera index in code (0, 1, 2...)
- Use camera_selector.py to find working camera

---

## Next Steps

Once installed:

1. **Test camera**: `python camera_selector.py`
2. **Run CVI system**: `python cvi_aac_board.py`
3. **Calibrate**: Follow on-screen instructions (5 points)
4. **Communicate**: Look at buttons to select

---

**Need help?** Check README_CVI.md for CVI-specific setup tips.
