# Santa 2026 - The Elf Calculator 🎅🧮

Kaggle Competition: [Santa 2026 - The Elf Calculator](https://www.kaggle.com/competitions/santa-2026)  
*Sponsored by Google LLC (Mountain View, CA)*

Write arithmetic programs to build large target numbers from small starting values in the fewest total steps using a 48-register arithmetic calculator.

---

## 🏆 Overview & Prizes ($50,000 Total)

The workshop's calculator performs basic arithmetic operations (`+`, `-`, `*`, `/`) across **48 storage registers** (indices `0` through `47`). The competition score is the **sum of all instructions executed across 1,000 scored puzzles** (`scored_0001` to `scored_1000`). Lower score is better.

| Award | Amount | Description |
| :--- | :--- | :--- |
| **1st Place** | **$12,000** | Lowest cumulative instructions across 1,000 puzzles |
| **2nd Place** | **$10,000** | 2nd lowest cumulative instructions |
| **3rd Place** | **$10,000** | 3rd lowest cumulative instructions |
| **4th Place** | **$8,000** | 4th lowest cumulative instructions |
| **Rudolph Prize** | **$10,000** | Team holding 1st place on the leaderboard for the longest cumulative duration between Nov 17, 2026 12:00 AM UTC and Jan 28, 2027 11:59 PM UTC |

*Points and medals are awarded for this competition.*

---

## 📅 Official Competition Timeline

* **November 18, 2026:** Competition Start Date
* **January 21, 2027 (11:59 PM UTC):** Entry Deadline & Team Merger Deadline
* **January 28, 2027 (11:59 PM UTC):** Final Submission Deadline

---

## ⚖️ Official Competition Rules & Compliance Summary

### 1. Teams & Submissions
* **Max Team Size:** Up to 5 members.
* **Daily Submissions:** Maximum **100 submissions per day**.
* **Final Submission Selection:** Up to **2 submissions** selected for final judging.
* **Private Sharing Prohibition:** Private sharing of code or data outside of teams is strictly prohibited. Any code shared must be made public to all participants on the Kaggle forums.

### 2. Data Security & Open Source Licensing
* **Data Security (Rule B.4.b):** Participants must prevent unauthorized distribution of Competition Data to non-participants. Competition data is strictly kept out of public repositories via `.gitignore`.
* **Winner's License (Rule B.5.a):** Winning solutions must be licensed under Open Source License **Apache 2.0** or **CC BY 4.0**, with full reproducible code and documentation.
* **External Data / Tools (Rule B.6):** Publicly available and equally accessible tools / data at no cost or adhering to the Reasonableness Standard are permitted.

---

## 📏 Technical Limits & Engine Rules

| Specification | Limit / Requirement |
| :--- | :--- |
| **Puzzles** | Exactly 1,000 puzzles (`scored_0001` through `scored_1000`) |
| **Storage Registers** | Exactly 48 registers per puzzle (indexed `0` through `47`) |
| **Initial State** | Registers `0..len(starting_values)-1` pre-loaded with puzzle `starting_values` |
| **Instruction Limit** | Maximum **4,096 instructions** per program |
| **Cell Size Limit** | Maximum **131,072 bytes (128 KiB)** UTF-8 per CSV program cell |
| **Allowed Operators** | `"+"` (Addition), `"-"` (Subtraction), `"*"` (Multiplication), `"/"` (Division) |
| **Division Rule (`/`)** | Must be **exact integer division** (`src1 % src2 == 0`, `src2 != 0`) |
| **Subtraction Rule (`-`)** | Result must be a **strictly positive integer** (`> 0`) |
| **Register Value Limits** | Positive integer ≤ `value_limit` at every intermediate execution step |
| **Instruction Format** | JSON array of 4 elements: `["op", src1, src2, dest]` |
| **Execution Order** | Sources are read before destination is overwritten (`dest` may equal `src1` or `src2`) |
| **Success Condition** | Every target number in a puzzle must be present in at least one register during execution |

---

## 📁 Repository Structure

```text
Santa 2026 - The Elf Calculator/
├── data/                    # (Excluded by .gitignore) Competition datasets
├── src/
│   ├── __init__.py
│   ├── simulator.py         # Full Elf Calculator simulator with 100% rule validation
│   ├── puzzle.py            # Data model parser for puzzles.csv
│   └── solver/              # (Excluded by .gitignore) Private optimization solvers
├── tests/
│   └── test_simulator.py    # Unit tests for calculator rules and edge cases
├── scripts/
│   └── validate_submission.py # Strict submission verifier
├── submissions/             # Generated submission CSVs (Excluded by .gitignore)
├── requirements.txt         # Dependencies
└── README.md
```

---

## 🛠️ Usage & Verification

### 1. Run Simulator Unit Tests
Verifies that all simulator operations, edge cases (inexact division, non-positive subtractions, value limits) behave according to official rules:
```bash
python -m unittest tests/test_simulator.py
```

### 2. Strict Pre-Submission Validation
Always validate before submitting to Kaggle to guarantee 100% compliance:
```bash
python scripts/validate_submission.py submissions/submission.csv --puzzles data/raw/puzzles.csv
```
The validator verifies:
- Exactly two columns: `id,program`
- All 1,000 IDs (`scored_0001` to `scored_1000`) present without duplicates or omissions
- UTF-8 byte size ≤ 128 KiB per cell
- Instruction count ≤ 4,096 per puzzle
- Simulates every instruction to guarantee zero division remainders, positive integers, values ≤ `value_limit`, and 100% target coverage.
