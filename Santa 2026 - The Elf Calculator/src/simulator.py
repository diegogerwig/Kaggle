"""
Elf Calculator Simulator and Validator for Santa 2026.
Simulates execution of 48-register arithmetic programs (+, -, *, /).
"""

from typing import List, Dict, Set, Tuple, Any, Optional, Union
import json

MAX_REGISTERS = 48
MAX_INSTRUCTIONS = 4096
MAX_CELL_BYTES = 131072
ALLOWED_OPS = {"+", "-", "*", "/"}


class SimulationError(Exception):
    """Custom exception raised when an invalid operation or limit breach occurs."""
    pass


class ElfCalculator:
    """
    Executes and validates an arithmetic program on the Elf Calculator.
    """

    def __init__(self, initial_registers: Union[Dict[int, int], List[Optional[int]], int], value_limit: Optional[int] = None):
        """
        Args:
            initial_registers: Initial register values. If an int is given, it is assigned to register 0.
                               If a dict is given, maps register index -> int.
                               If a list is given, register index i gets value list[i].
            value_limit: Maximum allowed integer in any register (None if unbounded).
        """
        self.registers: List[Optional[int]] = [None] * MAX_REGISTERS
        self.value_limit = value_limit
        self.produced_values: Set[int] = set()

        if isinstance(initial_registers, int):
            self._set_register(0, initial_registers, step="initial")
        elif isinstance(initial_registers, dict):
            for reg, val in initial_registers.items():
                self._set_register(reg, val, step="initial")
        elif isinstance(initial_registers, (list, tuple)):
            for reg, val in enumerate(initial_registers):
                if val is not None:
                    self._set_register(reg, val, step="initial")
        else:
            raise ValueError(f"Unsupported initial_registers type: {type(initial_registers)}")

    def _set_register(self, reg: int, val: int, step: Union[str, int]) -> None:
        if not (0 <= reg < MAX_REGISTERS):
            raise SimulationError(f"[Step {step}] Register index {reg} out of bounds (0..{MAX_REGISTERS-1}).")
        if not isinstance(val, int) or val <= 0:
            raise SimulationError(f"[Step {step}] Value {val} must be a strictly positive integer.")
        if self.value_limit is not None and val > self.value_limit:
            raise SimulationError(
                f"[Step {step}] Value {val} exceeds value_limit {self.value_limit} in register {reg}."
            )
        self.registers[reg] = val
        self.produced_values.add(val)

    def execute_instruction(self, instruction: Union[List[Any], Tuple[Any, ...]], step: int) -> int:
        """
        Executes a single instruction: [op, src1, src2, dest]
        """
        if len(instruction) != 4:
            raise SimulationError(
                f"[Step {step}] Instruction {instruction} must have exactly 4 elements: [op, src1, src2, dest]."
            )

        op, src1, src2, dest = instruction

        if op not in ALLOWED_OPS:
            raise SimulationError(f"[Step {step}] Unsupported operator '{op}'. Must be one of {ALLOWED_OPS}.")

        for name, reg in [("src1", src1), ("src2", src2), ("dest", dest)]:
            if not isinstance(reg, int) or not (0 <= reg < MAX_REGISTERS):
                raise SimulationError(f"[Step {step}] {name}={reg} is invalid. Register indices must be ints in 0..{MAX_REGISTERS-1}.")

        val1 = self.registers[src1]
        val2 = self.registers[src2]

        if val1 is None:
            raise SimulationError(f"[Step {step}] Source register {src1} is undefined.")
        if val2 is None:
            raise SimulationError(f"[Step {step}] Source register {src2} is undefined.")

        if op == "+":
            res = val1 + val2
        elif op == "-":
            res = val1 - val2
            if res <= 0:
                raise SimulationError(f"[Step {step}] Subtraction {val1} - {val2} = {res} is not a positive integer.")
        elif op == "*":
            res = val1 * val2
        elif op == "/":
            if val2 == 0:
                raise SimulationError(f"[Step {step}] Division by zero.")
            if val1 % val2 != 0:
                raise SimulationError(f"[Step {step}] Inexact division: {val1} / {val2} leaves a remainder ({val1 % val2}).")
            res = val1 // val2
            if res <= 0:
                raise SimulationError(f"[Step {step}] Division result {res} must be a positive integer.")
        else:
            raise SimulationError(f"[Step {step}] Unrecognized op '{op}'.")

        self._set_register(dest, res, step=step)
        return res

    def run_program(
        self,
        program: List[List[Any]],
        targets: Optional[Set[int]] = None
    ) -> Tuple[bool, Set[int], Optional[str]]:
        """
        Runs a full program and checks if all targets are reached.
        Returns:
            (is_valid, remaining_targets, error_message)
        """
        if len(program) > MAX_INSTRUCTIONS:
            return False, set(targets or []), f"Program exceeds maximum instruction limit ({len(program)} > {MAX_INSTRUCTIONS})."

        remaining = set(targets) if targets else set()
        # Remove targets already present in initial registers
        remaining.difference_update(self.produced_values)

        for step, instruction in enumerate(program):
            try:
                res = self.execute_instruction(instruction, step)
                if remaining and res in remaining:
                    remaining.remove(res)
            except SimulationError as e:
                return False, remaining, str(e)

        if remaining:
            return False, remaining, f"Missing {len(remaining)} target(s): {sorted(list(remaining))[:10]}..."

        return True, set(), None


def validate_program_string(program_str: str) -> Tuple[bool, Optional[List[List[Any]]], Optional[str]]:
    """
    Validates byte size and JSON structure of a program cell string.
    """
    byte_len = len(program_str.encode("utf-8"))
    if byte_len > MAX_CELL_BYTES:
        return False, None, f"Program cell exceeds {MAX_CELL_BYTES} bytes ({byte_len} bytes)."

    try:
        parsed = json.loads(program_str)
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON in program cell: {e}"

    if not isinstance(parsed, list):
        return False, None, "Program must be a JSON list of instructions."

    return True, parsed, None
