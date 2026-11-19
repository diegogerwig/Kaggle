# Santa 2026 - The Elf Calculator 🎅🧮

Kaggle Competition: [Santa 2026 - The Elf Calculator](https://www.kaggle.com/competitions/santa-2026)

Workspace and simulator setup for the Santa 2026 Kaggle Challenge.

---

## 📁 Repository Structure

```text
Santa 2026 - The Elf Calculator/
├── data/                    # (Excluded) Competition datasets (puzzles.csv, etc.)
├── src/
│   ├── __init__.py
│   ├── simulator.py         # Elf Calculator simulator and validator
│   └── puzzle.py            # Data models and puzzle loaders
├── tests/
│   └── test_simulator.py    # Unit tests for simulator operations
├── scripts/
│   └── validate_submission.py # Submission format and constraint validator
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Verification

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python -m unittest tests/test_simulator.py
```

### 3. Validate Submission
```bash
python scripts/validate_submission.py path/to/submission.csv --puzzles data/raw/puzzles.csv
```
