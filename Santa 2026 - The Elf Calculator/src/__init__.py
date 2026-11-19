"""
Santa 2026 - The Elf Calculator package.
"""

from src.simulator import ElfCalculator, SimulationError, MAX_REGISTERS, MAX_INSTRUCTIONS, MAX_CELL_BYTES
from src.puzzle import Puzzle, load_puzzles

__all__ = [
    "ElfCalculator",
    "SimulationError",
    "MAX_REGISTERS",
    "MAX_INSTRUCTIONS",
    "MAX_CELL_BYTES",
    "Puzzle",
    "load_puzzles"
]
