"""
Installation Test Script
Checks if all required packages are installed for CVI AAC system
"""

import sys

print("="*60)
print("CVI AAC SYSTEM - INSTALLATION CHECK")
print("="*60)

print(f"\nPython Version: {sys.version}")
print(f"Python Path: {sys.executable}")

print("\n" + "-"*60)
print("CHECKING DEPENDENCIES")
print("-"*60)

all_good = True

# Check OpenCV (REQUIRED)
try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__} (REQUIRED)")
except ImportError:
    print("✗ OpenCV: NOT INSTALLED (REQUIRED)")
    print("  Install: pip install opencv-python")
    all_good = False

# Check NumPy (REQUIRED)
try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__} (REQUIRED)")
except ImportError:
    print("✗ NumPy: NOT INSTALLED (REQUIRED)")
    print("  Install: pip install numpy")
    all_good = False

# Check pyttsx3 (OPTIONAL but recommended)
try:
    import pyttsx3
    print("✓ pyttsx3: installed (text-to-speech)")
except ImportError:
    print("⚠ pyttsx3: NOT INSTALLED (optional)")
    print("  Install for audio: pip install pyttsx3")
    print("  System will work without audio")

# Check dlib (OPTIONAL - not needed for CVI system)
try:
    import dlib
    print("✓ dlib: installed (optional, for GazeTracking)")
except ImportError:
    print("⚠ dlib: NOT INSTALLED (not needed for CVI system)")
    print("  CVI system uses Hough Circle tracker (no dlib needed)")

print("\n" + "-"*60)
print("CAMERA CHECK")
print("-"*60)

try:
    import cv2
    # Try to open camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("✓ Camera 0: Working")
            h, w = frame.shape[:2]
            print(f"  Resolution: {w}x{h}")
        else:
            print("⚠ Camera 0: Opens but can't read frames")
        cap.release()
    else:
        print("⚠ Camera 0: Cannot open")
        print("  Try: python camera_selector.py")
except Exception as e:
    print(f"✗ Camera test failed: {e}")

print("\n" + "="*60)
print("RESULT")
print("="*60)

if all_good:
    print("\n✓ ALL REQUIRED PACKAGES INSTALLED")
    print("\nYou're ready to run:")
    print("  python cvi_aac_board.py")
else:
    print("\n✗ MISSING REQUIRED PACKAGES")
    print("\nInstall missing packages:")
    print("  pip install opencv-python numpy pyttsx3")

print("\n" + "="*60)
print("\nNOTE: dlib is NOT required for the CVI AAC system")
print("The system uses Hough Circle tracking (built into OpenCV)")
print("="*60)
