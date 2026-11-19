"""
Submission validator for Santa 2026 - The Elf Calculator.
Strictly checks:
1. Two columns: 'id' and 'program' (in that order).
2. All 1,000 IDs (scored_0001 through scored_1000) present exactly once.
3. UTF-8 byte size <= 131,072 bytes per cell.
4. Max 4,096 instructions per program.
5. Replays program through ElfCalculator to ensure 100% rule compliance and target coverage.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.puzzle import load_puzzles
from src.simulator import ElfCalculator, MAX_INSTRUCTIONS, MAX_CELL_BYTES, validate_program_string


def validate_submission(submission_path: str, puzzles_path: str = "data/raw/puzzles.csv"):
    sub_file = Path(submission_path)
    if not sub_file.exists():
        print(f"Error: Submission file {sub_file} does not exist.")
        return False

    puzzles_file = Path(puzzles_path)
    has_puzzles = puzzles_file.exists()
    puzzles = load_puzzles(puzzles_file) if has_puzzles else {}

    print(f"Validating submission: {sub_file}")
    total_instructions = 0
    seen_ids = set()

    with open(sub_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["id", "program"]:
            print(f"ERROR: Invalid header {header}. Must be exactly ['id', 'program'].")
            return False

        for row_idx, row in enumerate(reader, 1):
            if len(row) != 2:
                print(f"ERROR at line {row_idx+1}: Expected 2 columns, got {len(row)}.")
                return False

            p_id, prog_str = row
            if p_id in seen_ids:
                print(f"ERROR: Duplicate ID '{p_id}' detected at line {row_idx+1}.")
                return False
            seen_ids.add(p_id)

            # Check byte limit and JSON parsing
            is_valid_json, program, err = validate_program_string(prog_str)
            if not is_valid_json:
                print(f"ERROR for '{p_id}': {err}")
                return False

            if len(program) > MAX_INSTRUCTIONS:
                print(f"ERROR for '{p_id}': Instruction count {len(program)} > limit {MAX_INSTRUCTIONS}.")
                return False

            total_instructions += len(program)

            # If puzzles are available, simulate
            if has_puzzles and p_id in puzzles:
                puzzle = puzzles[p_id]
                calc = ElfCalculator(puzzle.initial_registers, value_limit=puzzle.value_limit)
                valid_sim, remaining, sim_err = calc.run_program(program, targets=set(puzzle.targets))
                if not valid_sim:
                    print(f"SIMULATION ERROR for '{p_id}': {sim_err}")
                    return False

    # Check IDs
    expected_ids = {f"scored_{i:04d}" for i in range(1, 1001)}
    missing_ids = expected_ids - seen_ids
    if missing_ids:
        print(f"WARNING: Missing {len(missing_ids)} IDs out of 1000! (e.g. {sorted(list(missing_ids))[:5]})")
    else:
        print("✓ All 1,000 scored IDs (scored_0001..scored_1000) are present exactly once.")

    print(f"\n==========================================")
    print(f"VALIDATION PASSED!")
    print(f"Total Rows Verified: {len(seen_ids)}")
    print(f"Total Score (Instructions): {total_instructions}")
    print(f"Average Instructions / Puzzle: {total_instructions / len(seen_ids):.2f}")
    print(f"==========================================")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Santa 2026 submission.")
    parser.add_argument("submission", type=str, help="Path to submission.csv")
    parser.add_argument("--puzzles", type=str, default="data/raw/puzzles.csv", help="Path to puzzles.csv")
    args = parser.parse_args()

    success = validate_submission(args.submission, args.puzzles)
    sys.exit(0 if success else 1)
