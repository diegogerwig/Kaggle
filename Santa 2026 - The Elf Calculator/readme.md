# Santa 2026 - The Elf Calculator 🎅🧮

Kaggle Competition: [Santa 2026 - The Elf Calculator](https://www.kaggle.com/competitions/santa-2026)  
Write arithmetic programs to build large target numbers from small starting values in the fewest total steps.

---

## 🏆 Overview & Prizes

Santa's workshop toy calculator can store up to **48 values** simultaneously and only perform one basic arithmetic operation at a time (`+`, `-`, `*`, `/`). The objective is to produce all target numbers for 1,000 distinct puzzles in as few instructions as possible.

### Prize Pool ($50,000 Total)
* **1st Place:** $12,000
* **2nd Place:** $10,000
* **3rd Place:** $10,000
* **4th Place:** $8,000
* **Rudolph Prize:** $10,000 (Awarded to the team holding 1st place on the leaderboard for the longest cumulative duration between Nov 18, 2026 and Jan 28, 2027).

---

## 📅 Timeline

* **November 18, 2026:** Competition Start Date
* **January 21, 2027:** Entry & Team Merger Deadline
* **January 28, 2027:** Final Submission Deadline (11:59 PM UTC)

---

## 📏 Constraints & Rules

| Parameter | Limit / Rule |
| :--- | :--- |
| **Registers per puzzle** | 48 (indices `0` through `47`) |
| **Max instructions per puzzle** | 4,096 |
| **Max cell size in CSV** | 131,072 bytes (128 KiB) |
| **Puzzles evaluated** | 1,000 (`scored_0001` through `scored_1000`) |
| **Allowed operators** | `+`, `-`, `*`, `/` |
| **Division (`/`)** | Must be **exact** integer division ($a \pmod b = 0$, $b \ne 0$) |
| **Subtraction (`-`)** | Result must be a **strictly positive integer** ($> 0$) |
| **Register values** | Positive integer $\le \text{value\_limit}$ |
| **Scoring Metric** | Total instructions summed across all 1,000 puzzles (**Lower is better**) |
| **Leaderboard Split** | None (100% of the 1,000 puzzles are evaluated directly on the leaderboard) |

### Instruction Format
Each instruction is a 4-element JSON array:
```json
["op", src1, src2, dest]
```
- `op`: `"+"` | `"-"` | `"*"` | `"/"`
- `src1`, `src2`: Source registers (`0`..`47`). Both must hold defined values before execution.
- `dest`: Destination register (`0`..`47`). Destination can be the same as `src1` or `src2`.
- Target condition: Every target number in a puzzle must be present in at least one register during or at the end of the program execution.

---

## 📁 Repository Structure

```text
Santa 2026 - The Elf Calculator/
├── data/
│   ├── raw/                 # Competition datasets (puzzles.csv, sample_submission.csv)
│   └── processed/           # Processed chains and lookup tables
├── src/
│   ├── __init__.py
│   ├── simulator.py         # Complete Elf Calculator simulator & instruction validator
│   ├── puzzle.py            # Dataset loading and puzzle data structures
│   └── solver/              # Optimization and search algorithms (local / protected)
├── tests/
│   └── test_simulator.py    # Unit tests for simulator correctness
├── scripts/
│   ├── run_solver.py        # Pipeline runner to generate submissions
│   └── validate_submission.py # Strict submission verifier
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python -m unittest tests/test_simulator.py
```

### 3. Generate Submission
```bash
python scripts/run_solver.py --puzzles data/raw/puzzles.csv --output submissions/submission.csv
```

### 4. Validate Submission
```bash
python scripts/validate_submission.py submissions/submission.csv --puzzles data/raw/puzzles.csv
```
