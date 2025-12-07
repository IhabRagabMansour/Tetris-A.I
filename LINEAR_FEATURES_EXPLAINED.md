# Linear Neural Network Features Explained

This document explains the 6 hand-engineered features used by the **Linear Neural Network** architecture for Tetris AI decision-making. Unlike the CNN which learns features automatically from the raw board, the Linear model relies on these carefully designed features to understand the game state.

## Table of Contents
- [Overview](#overview)
- [The 6 Features](#the-6-features)
- [Visual Examples](#visual-examples)
- [Enabling Feature Visualization](#enabling-feature-visualization)
- [How Features Influence Rewards](#how-features-influence-rewards)

---

## Overview

The Linear NN receives a **1D vector of 6 numerical values** as input, representing:
1. Total Heights
2. Bumpiness
3. Lines Removed
4. Holes
5. Y Position (Max Height)
6. Pillar (Binary Flag)

**Input shape:** `(batch_size, 6)`
**Example:** `[78.0, 12.0, 1.0, 3.0, 15.0, 0.0]`

These features compress the entire 20×10 board state into just 6 numbers, making training faster but relying on human domain knowledge.

---

## The 6 Features

### 1. **Total Heights**
- **Definition:** Sum of all column heights across the 10 columns
- **Range:** 0 to 200 (theoretical max: 20 × 10 = 200)
- **Meaning:**
  - Low values (0-50): Board is mostly empty, early game
  - Medium values (50-100): Mid-game, moderate stack height
  - High values (100+): Dangerous, board is filling up
- **Calculation:** `sum([h1, h2, h3, ..., h10])`

**Example:**
```
Column heights: [5, 8, 7, 10, 12, 9, 6, 4, 3, 7]
Total Heights = 5 + 8 + 7 + 10 + 12 + 9 + 6 + 4 + 3 + 7 = 71
```

**Visual Indicator:** Numbers above each column (color-coded):
- 🟢 Green: Short columns (< 8)
- 🟡 Yellow: Medium columns (8-15)
- 🔴 Red: Tall columns (> 15)

---

### 2. **Bumpiness**
- **Definition:** Total absolute height difference between adjacent columns
- **Range:** 0 to 180 (very uneven board)
- **Meaning:**
  - Low values (0-5): Flat, smooth surface (GOOD)
  - Medium values (6-15): Some variation (OK)
  - High values (16+): Very uneven, hard to fill (BAD)
- **Calculation:** `|h1-h2| + |h2-h3| + |h3-h4| + ... + |h9-h10|`

**Example:**
```
Column heights: [5, 8, 7, 10, 12, 9, 6, 4, 3, 7]
Bumpiness = |5-8| + |8-7| + |7-10| + |10-12| + |12-9| + |9-6| + |6-4| + |4-3| + |3-7|
         = 3 + 1 + 3 + 2 + 3 + 3 + 2 + 1 + 4
         = 22
```

**Visual Indicator:** Orange/yellow lines between columns with height difference labeled

---

### 3. **Lines Removed**
- **Definition:** Number of lines cleared in the current move
- **Range:** 0 to 4
- **Meaning:**
  - 0: No lines cleared
  - 1-3: Good progress
  - 4: Tetris! (Best case, highest reward)
- **Purpose:** Immediate reward signal for successful line clearing

**Example:**
```
Before move:  ████ █████  After move:  (1 line cleared)
              ██████████              ████ █████
              ██████████
              ██████████

Lines Removed = 1
```

---

### 4. **Holes**
- **Definition:** Number of empty cells that have at least one block above them
- **Range:** 0 to 200 (theoretical max)
- **Meaning:**
  - 0: Perfect, no holes (IDEAL)
  - 1-3: Acceptable, minor issues
  - 4+: Bad, blocks are trapped and hard to clear
- **Calculation:** Count empty cells with blocks above in the same column

**Example:**
```
Board visualization:
Col 0: [0, 0, 1, 0, 1]  ← 1 hole (row 3 is empty, has blocks above)
Col 1: [0, 1, 1, 1, 1]  ← 0 holes
Col 2: [0, 1, 0, 0, 1]  ← 2 holes (rows 2 and 3 are empty)

Total Holes = 1 + 0 + 2 = 3
```

**Visual Indicator:** Red circles inside empty cells that have blocks above them

---

### 5. **Y Position (Max Height)**
- **Definition:** The maximum height of the currently falling tetromino piece
- **Range:** 0 to 20
- **Meaning:**
  - Low values (0-10): Piece placed low (GOOD)
  - High values (15+): Piece placed high, danger zone (BAD)
- **Purpose:** Encourages placing pieces as low as possible

**Example:**
```
Falling piece at row 5:
█
███  ← Max Y position = 5
```

---

### 6. **Pillar**
- **Definition:** Binary flag indicating if there's a dangerous "spike" column
- **Range:** 0 (no pillar) or 1 (pillar exists)
- **Meaning:**
  - 0: No dangerous spikes, relatively even surface
  - 1: At least one column is ≥3 blocks higher than both neighbors (DANGEROUS)
- **Calculation:** Check if any column satisfies:
  - `height[i-1] - height[i] >= 3` AND `height[i+1] - height[i] >= 3`
  - OR edge columns: `height[1] - height[0] >= 3` or `height[-2] - height[-1] >= 3`

**Example:**
```
Column heights: [10, 10, 6, 10, 10]  ← Column 2 is a valley (pillar detected)
                 ██  ██  ██  ██  ██
                 ██  ██      ██  ██
                 ██  ██  __  ██  ██  ← 4-block difference!

Pillar = 1 (column 2 has a spike of -4 relative to neighbors)
```

**Visual Indicator:** Red rectangle border highlighting the problematic column

---

## Visual Examples

### Example 1: Empty Board (Beginning of Game)
```
Board:
____________________
____________________
____________________
____________________

Features:
- Total Heights: 0
- Bumpiness: 0
- Lines Removed: 0
- Holes: 0
- Y Position: 0
- Pillar: 0

State vector: [0, 0, 0, 0, 0, 0]
```

---

### Example 2: Clean, Flat Board (Good State)
```
Board:
____________________
____________________
____________________
████████████████████
████████████████████
████████████████████

Column Heights: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

Features:
- Total Heights: 30
- Bumpiness: 0 (perfectly flat)
- Lines Removed: 0
- Holes: 0
- Y Position: 3
- Pillar: 0

State vector: [30, 0, 0, 0, 3, 0]
✅ GOOD STATE: Low bumpiness, no holes
```

---

### Example 3: Uneven Board with Holes (Bad State)
```
Board:
____██______________
____██______________
██__████____████____
████████____████████
████████__██████████

Column Heights: [4, 4, 2, 5, 5, 5, 2, 2, 4, 4]
Holes: 3 (columns 2, 6, 7 have empty cells below blocks)

Features:
- Total Heights: 37
- Bumpiness: 14 (lots of height variation)
- Lines Removed: 0
- Holes: 3
- Y Position: 5
- Pillar: 1 (column 2 and 6 are deep valleys)

State vector: [37, 14, 0, 3, 5, 1]
❌ BAD STATE: High bumpiness, holes present, pillar detected
```

---

### Example 4: Tetris Clear (Best Case)
```
Before Move:
████████__
████████__
████████__
██████████

After Move (4 lines cleared):
__________
__________
__________
__________

Features:
- Total Heights: 0 (after clearing)
- Bumpiness: 0
- Lines Removed: 4 ← TETRIS!
- Holes: 0
- Y Position: 4
- Pillar: 0

State vector: [0, 0, 4, 0, 4, 0]
⭐ EXCELLENT: 4-line Tetris clear!
```

---

## Enabling Feature Visualization

When training with **Linear architecture** and **RENDER = True**:

### Side Panel Display
A feature panel automatically appears on the right side showing:
- Real-time values for all 6 features
- Color-coded indicators (red = bad, green = good)
- Feature descriptions

### Visual Overlays (Press 'F' to Toggle)
Press the **'F' key** during gameplay to toggle visual overlays on the board:

1. **Column Heights:** Numbers above each column
   - 🟢 Green: < 8 blocks
   - 🟡 Yellow: 8-15 blocks
   - 🔴 Red: > 15 blocks

2. **Holes:** Red circles inside empty cells under blocks

3. **Bumpiness:** Orange lines between columns showing height differences

4. **Pillars:** Red border highlighting dangerous spike columns

**Note:** Feature visualization only works when:
- `architecture = "Linear"`
- `RENDER = True` in `settings.py`

---

## How Features Influence Rewards

The Linear NN learns to predict rewards based on these features. The reward function typically penalizes:

- ❌ **High Total Heights:** Dangerous, close to game over
- ❌ **High Bumpiness:** Uneven surface, hard to place pieces
- ❌ **Holes:** Trapped blocks, difficult to clear
- ❌ **Pillars:** Creates deep valleys, limits placement options
- ❌ **High Y Position:** Pieces stacked too high

And rewards:
- ✅ **Lines Removed:** Primary goal, especially 4-line Tetris clears
- ✅ **Low Heights:** Safe board state, more room to maneuver
- ✅ **Flat Surface (low bumpiness):** Easier to place future pieces

---

## Comparing Linear vs CNN

| Aspect | Linear NN | CNN |
|--------|-----------|-----|
| **Input** | 6 features | 20×10 raw board |
| **Feature Engineering** | Manual (human-designed) | Automatic (learned) |
| **Input Size** | 6 values | 200 values |
| **What It Sees** | Compressed statistics | Full spatial patterns |
| **Training Speed** | Faster (smaller input) | Slower (larger input) |
| **Flexibility** | Limited to designed features | Can discover novel patterns |
| **Interpretability** | High (we know what it uses) | Low (black box) |

---

## Summary

The Linear NN uses these 6 carefully crafted features to make decisions:

1. **Total Heights** → Overall board fullness
2. **Bumpiness** → Surface smoothness
3. **Lines Removed** → Immediate scoring
4. **Holes** → Trapped blocks penalty
5. **Y Position** → Piece placement height
6. **Pillar** → Dangerous spike detection

By understanding these features, you can better interpret why the Linear model makes certain decisions and compare its behavior to the CNN architecture.

---

**Enable visualization by:**
1. Setting `RENDER = True` in `settings.py`
2. Running with `architecture="Linear"`
3. Pressing **'F'** to toggle board overlays

Happy training! 🎮🤖
