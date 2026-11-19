"""
Puzzle representations and data loaders for Santa 2026 - The Elf Calculator.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
import csv
import json
from pathlib import Path


@dataclass
class Puzzle:
    id: str
    starting_values: List[int]
    targets: List[int]
    value_limit: int
    num_registers: int = 48
    initial_registers: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.initial_registers:
            self.initial_registers = {i: v for i, v in enumerate(self.starting_values)}


Union_Path = Path | str


def load_puzzles(csv_path: Union_Path) -> Dict[str, Puzzle]:
    """
    Loads puzzles from a CSV file (e.g. data/raw/puzzles.csv).
    Handles starting_values, targets, value_limit, registers.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Puzzle file not found: {csv_file}")

    puzzles: Dict[str, Puzzle] = {}
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzle_id = row.get("id") or row.get("puzzle_id")

            # Starting values
            start_raw = row.get("starting_values") or row.get("seed") or "[3]"
            if isinstance(start_raw, str):
                try:
                    starting_values = json.loads(start_raw)
                    if isinstance(starting_values, int):
                        starting_values = [starting_values]
                except Exception:
                    starting_values = [int(x.strip()) for x in start_raw.replace("[", "").replace("]", "").replace(",", " ").split() if x.strip()]
            elif isinstance(start_raw, int):
                starting_values = [start_raw]
            else:
                starting_values = list(start_raw)

            # Value limit
            val_limit = 10**18
            if "value_limit" in row and row["value_limit"]:
                val_limit = int(row["value_limit"])
            elif "limit" in row and row["limit"]:
                val_limit = int(row["limit"])

            # Num registers
            num_regs = 48
            if "registers" in row and row["registers"]:
                num_regs = int(row["registers"])

            # Targets
            targets_raw = row.get("targets") or row.get("target") or row.get("target_numbers", "[]")
            if isinstance(targets_raw, str):
                try:
                    targets = json.loads(targets_raw)
                except Exception:
                    targets = [int(x.strip()) for x in targets_raw.replace("[", "").replace("]", "").replace(",", " ").split() if x.strip()]
            else:
                targets = list(targets_raw)

            init_regs = {i: v for i, v in enumerate(starting_values)}

            puzzles[puzzle_id] = Puzzle(
                id=puzzle_id,
                starting_values=starting_values,
                targets=targets,
                value_limit=val_limit,
                num_registers=num_regs,
                initial_registers=init_regs
            )

    return puzzles


Union_Path = Path | str
