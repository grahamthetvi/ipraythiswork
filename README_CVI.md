# CVI-Optimized AAC System

## Designed for Students with Cortical Visual Impairment

This specialized version of the eye gaze AAC system is specifically designed for students with **Cortical Visual Impairment (CVI)** based on research-backed design principles.

---

## What is CVI?

**Cortical Visual Impairment** is a brain-based visual impairment where the eyes may work fine, but the brain has difficulty processing visual information. Students with CVI need specially designed visual materials.

---

## CVI-Specific Design Features

### 1. **High Contrast**
- ✅ **Pure black background** (#000000)
- ✅ **Pure white bold text** for maximum readability
- ✅ **No patterns or textures** that could confuse visual processing
- ✅ **Large, clear fonts** (1.5x - 3.0x normal size)

### 2. **Red Bubbling/Outlines**
- ✅ **Red is the primary color** - Research shows CVI students often see red first
- ✅ **Pulsing red borders** (6-14px thickness) that animate to attract attention
- ✅ **Red progress indicators** for dwell time
- ✅ **Red pupil detection markers** in camera view

### 3. **Simplified Layout**
- ✅ **Maximum 5 items** per screen (not 9)
- ✅ **Large button areas** - each button occupies significant screen space
- ✅ **Minimal visual clutter** - nothing extra to distract
- ✅ **Predictable layout** - consistent placement

### 4. **Simple Icons**
- ✅ **Large, simple symbols** (emojis/unicode characters)
- ✅ **High contrast** against background
- ✅ **Paired with text** for clarity
- ✅ **No complex images** or photographs

### 5. **Movement for Attention**
- ✅ **Pulsing animations** on current target
- ✅ **Red bubbling effect** draws eyes to active elements
- ✅ **Progress filling** from bottom to top (visual feedback)

---

## Files for CVI Students

| File | Purpose |
|------|---------|
| **cvi_aac_board.py** | Main CVI-optimized AAC communication board (5 regions) |
| **cvi_calibration.py** | 5-point calibration system (instead of 9-point) |

---

## How to Use

### Quick Start

```bash
# Run the CVI-optimized AAC board
python cvi_aac_board.py
```

### Complete Setup

1. **Camera Selection** (optional)
   ```bash
   python camera_selector.py
   # Find best camera with pupil detection
   ```

2. **Run CVI AAC Board**
   ```bash
   python cvi_aac_board.py
   # Select camera? y/n
   ```

3. **Calibration**
   - 5 points instead of 9 (simpler, faster)
   - Large red pulsing circles
   - Extra time per point (40 samples vs 30)
   - Audio cues: "Point 1, top left", etc.

4. **Communication**
   - Look at large buttons with red borders
   - Red pulsing shows which button you're looking at
   - Progress fills from bottom (visual dwell time)
   - Selection speaks aloud

---

## CVI Design Principles Applied

### Based on CVI Research:

1. **Color Preference**: Red → Yellow → High contrast
   - ✅ Red used for all primary elements
   - ✅ Yellow could be added as alternative

2. **Visual Complexity**: Simple > Complex
   - ✅ Only 5 buttons maximum
   - ✅ Large, simple shapes
   - ✅ No background patterns

3. **Movement**: Moving objects attract attention
   - ✅ Pulsing red borders (2-4 Hz)
   - ✅ Filling progress indicator
   - ✅ Smooth animations

4. **Single Object Presentation**: Fewer items better
   - ✅ 5 items vs typical 9
   - ✅ Large spacing between items
   - ✅ One item can be focused at a time

5. **Latency**: Longer processing time needed
   - ✅ Default dwell time: 2.0 seconds (vs 1.5)
   - ✅ More samples during calibration
   - ✅ Extra smoothing for stability

6. **Visual Field Preferences**: Many CVI students prefer lower visual field
   - ✅ 5-region layout accommodates this
   - ✅ Buttons well-distributed across screen

---

## 5-Region Layout

```
┌───────────────────────────────┐
│  [0]              [1]          │  Top row: 2 buttons
│   YES              NO          │
│                                │
│         [2]                    │  Center: 1 button
│        HELP                    │
│                                │
│  [3]              [4]          │  Bottom row: 2 buttons
│  I NEED           MORE         │
└───────────────────────────────┘
```

**Regions:**
- **0**: Top-left
- **1**: Top-right
- **2**: Center
- **3**: Bottom-left
- **4**: Bottom-right

**Why 5 instead of 9?**
- Reduces visual complexity
- Larger buttons (easier targets)
- Faster to learn
- Less cognitive load
- Better for students with limited visual field

---

## Available Boards

### Main Board (5 items)
1. **YES** (✓) - Affirmative response
2. **NO** (✗) - Negative response
3. **HELP** (!) - Request assistance
4. **MORE** (→) - Navigate to more boards
5. **I NEED** (☝) - Express needs

### I Need Board (5 items)
1. **WATER** (💧) - Request water
2. **FOOD** (🍎) - Request food
3. **BATHROOM** (🚻) - Bathroom break
4. **BREAK** (⏸) - Rest/break
5. **BACK** (←) - Return to main

### How I Feel Board (5 items)
1. **HAPPY** (😊) - Positive emotion
2. **SAD** (😢) - Negative emotion
3. **TIRED** (😴) - Fatigue
4. **OKAY** (👍) - Neutral/fine
5. **BACK** (←) - Return to main

**Note**: Icons are simple Unicode characters, not complex images. They provide visual cues while maintaining high contrast.

---

## Calibration for CVI Students

### 5-Point Calibration

**Points:**
1. Top-left
2. Top-right
3. Center
4. Bottom-left
5. Bottom-right

**Features:**
- **Large red pulsing circles** (60-100px diameter)
- **40 samples per point** (vs 30 for standard)
- **Audio cues** announcing each point
- **White dots** show completed points
- **Pure black background**
- **Large progress bar** with red fill

**Tips:**
- Allow extra time per point
- Ensure good lighting (but not too bright)
- Student should be comfortable and stable
- Use verbal encouragement
- Can repeat if needed

---

## Controls

### During Use:
- **ESC**: Quit application
- **C**: Clear current message
- **+**: Increase dwell time (slower)
- **-**: Decrease dwell time (faster)

### Recommended Settings for CVI:
- **Dwell time**: 2.0-3.0 seconds (longer is better)
- **Confidence threshold**: 0.70 (more forgiving)
- **Smoothing**: 7 frames (more stable)

---

## Comparison: Standard vs CVI

| Feature | Standard System | CVI-Optimized |
|---------|----------------|---------------|
| **Regions** | 9 (3x3 grid) | 5 (custom layout) |
| **Background** | Dark gray | Pure black |
| **Borders** | Gray/multi-color | Red pulsing |
| **Items/screen** | Up to 9 | Maximum 5 |
| **Button size** | Medium | Extra large |
| **Calibration** | 9 points | 5 points |
| **Samples/point** | 30 | 40 |
| **Dwell time** | 1.5s | 2.0s |
| **Text size** | 1.0x | 1.5-3.0x |
| **Animations** | Minimal | Pulsing red |
| **Icons** | Optional | Simple, large |

---

## CVI Characteristics Addressed

### Common CVI Characteristics:

1. **Color Preference** (often red/yellow first)
   - ✅ Red as primary color throughout

2. **Visual Complexity** (simple better)
   - ✅ Maximum 5 items
   - ✅ No background patterns

3. **Movement** (attracts attention)
   - ✅ Pulsing animations
   - ✅ Progress indicators

4. **Visual Latency** (needs more time)
   - ✅ Longer dwell time
   - ✅ More calibration samples

5. **Visual Field Preferences** (lower field often better)
   - ✅ Flexible 5-region layout
   - ✅ Buttons throughout screen

6. **Light Sensitivity** (varies)
   - ✅ Black background reduces glare
   - ✅ Adjustable camera brightness

7. **Distance Viewing** (often better up close)
   - ✅ Large targets visible close-up
   - ✅ Camera positioning flexible

---

## Setup for CVI Students

### Environment:
1. **Lighting**: Even, diffuse light (avoid glare)
2. **Background**: Simple, uncluttered area
3. **Position**: Student comfortable, stable
4. **Distance**: Camera 18-24 inches from eyes
5. **Screen**: At comfortable viewing distance

### Camera:
1. **Use camera selector** to find best camera
2. **Verify pupil detection** (should see red circles)
3. **Adjust brightness** if needed
4. **Position at eye level**

### Calibration:
1. **Explain process** simply
2. **Give extra time** for each point
3. **Use verbal encouragement**
4. **Watch for fatigue** (short sessions better)
5. **Can repeat** if needed

### During Use:
1. **Monitor engagement**
2. **Watch for visual fatigue**
3. **Adjust dwell time** as needed
4. **Encourage verbally**
5. **Celebrate successes**

---

## Troubleshooting for CVI

### Issue: Student not looking at targets

**Possible causes:**
- Targets too small → They're already large, but verify screen size
- Color not visible → Red should be most visible for CVI
- Too much visual complexity → CVI system has minimal clutter
- Visual fatigue → Take breaks, shorter sessions

**Solutions:**
- Ensure black background, red borders visible
- Check lighting (not too bright/dark)
- Reduce session length
- Use verbal cues to direct attention

### Issue: Pupil not detected

**Solutions:**
- Use camera selector to find best camera
- Adjust lighting
- Check camera positioning
- Ensure glasses are not causing glare

### Issue: Selections inaccurate

**Solutions:**
- Recalibrate (5-point calibration)
- Increase dwell time (+key)
- Check student positioning
- Verify calibration quality

### Issue: Student seems frustrated

**Solutions:**
- Increase dwell time (more time to select)
- Reduce number of boards (keep it simple)
- Take frequent breaks
- Ensure system is responding (watch pupil detection)

---

## Research-Based Design

This system incorporates CVI research from:

- **Dr. Christine Roman-Lantzy** (CVI expert, author of "Cortical Visual Impairment: An Approach to Assessment and Intervention")
- **Perkins School for the Blind** CVI resources
- **CVI characteristics** identified in educational literature
- **Color preference studies** in CVI population
- **Visual complexity research** for CVI learners

---

## Technical Details

### Color Specifications:
- **Background**: RGB(0, 0, 0) - Pure black
- **Text**: RGB(255, 255, 255) - Pure white
- **Primary outlines**: RGB(255, 0, 0) - Pure red
- **Button background**: RGB(30, 30, 30) - Very dark gray

### Animation:
- **Pulse frequency**: 2-4 Hz (visible but not jarring)
- **Border thickness**: 6-14px (varies with pulse)
- **Progress fill**: Bottom-to-top (predictable direction)

### Layout:
- **Button spacing**: Minimum 40px between buttons
- **Button size**: ~25-30% of screen area each
- **Text size**: 1.5x-3.0x standard (24-48pt equivalent)

---

## Future Enhancements

Potential additions for CVI students:

1. **Yellow mode**: Switch primary color from red to yellow
2. **Contrast adjuster**: Fine-tune black/white levels
3. **Simplified boards**: Even fewer items (2-3)
4. **Position memory**: Remember last successful positions
5. **Fatigue detection**: Monitor and suggest breaks
6. **Custom icons**: Upload student-specific simple images

---

## Questions?

### For CVI-specific questions:
- Consult with student's Teacher of the Visually Impaired (TVI)
- Review student's CVI Range score
- Consider individual CVI characteristics
- Adjust system to student's specific needs

### For technical questions:
- See README_SETUP.md for general troubleshooting
- Use camera_selector.py to verify pupil detection
- Check calibration quality scores

---

## Summary

The CVI-optimized AAC system provides:

✅ **High contrast**: Black background, white text, red outlines
✅ **Simple layout**: Maximum 5 items instead of 9
✅ **Red bubbling**: Pulsing red borders for visual attention
✅ **Large targets**: Extra-large buttons for easier access
✅ **Simple icons**: High-contrast symbols, not complex images
✅ **5-point calibration**: Faster, simpler than 9-point
✅ **Extra time**: Longer dwell time, more samples
✅ **Movement**: Pulsing animations to attract attention
✅ **Minimal clutter**: Clean, simple interface

**This system is specifically designed for students with CVI based on research and best practices in the field.**

---

**File**: README_CVI.md
**Version**: 1.0
**For**: Students with Cortical Visual Impairment
**Last Updated**: 2025-01-24
