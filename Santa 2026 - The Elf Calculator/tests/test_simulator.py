"""
Unit tests for Elf Calculator simulator and baseline solver.
Uses standard library unittest so it runs out-of-the-box without extra packages.
"""

import unittest
from src.simulator import ElfCalculator, SimulationError
from src.puzzle import Puzzle
from src.solver.baseline import BaselineSolver
from src.solver.dag_portfolio_solver import DAGPortfolioSolver


class TestElfCalculator(unittest.TestCase):

    def test_toy_example_from_prompt(self):
        """
        Tests the toy puzzle from the competition overview:
        seed = 3
        ["/", 0, 0, 1] -> 3 / 3 = 1 (reg 1)
        ["-", 0, 1, 2] -> 3 - 1 = 2 (reg 2)
        ["+", 0, 2, 3] -> 3 + 2 = 5 (reg 3)
        ["+", 0, 3, 4] -> 3 + 5 = 8 (reg 4)
        """
        calc = ElfCalculator(initial_registers={0: 3}, value_limit=1000)
        program = [
            ["/", 0, 0, 1],
            ["-", 0, 1, 2],
            ["+", 0, 2, 3],
            ["+", 0, 3, 4],
        ]
        is_valid, remaining, error = calc.run_program(program, targets={1, 2, 5, 8})
        self.assertTrue(is_valid, f"Program failed: {error}")
        self.assertEqual(len(remaining), 0)
        self.assertEqual(calc.registers[0], 3)
        self.assertEqual(calc.registers[1], 1)
        self.assertEqual(calc.registers[2], 2)
        self.assertEqual(calc.registers[3], 5)
        self.assertEqual(calc.registers[4], 8)

    def test_inexact_division_fails(self):
        calc = ElfCalculator(initial_registers={0: 5, 1: 2})
        with self.assertRaises(SimulationError) as ctx:
            calc.execute_instruction(["/", 0, 1, 2], step=0)
        self.assertIn("Inexact division", str(ctx.exception))

    def test_non_positive_subtraction_fails(self):
        calc = ElfCalculator(initial_registers={0: 3, 1: 5})
        with self.assertRaises(SimulationError) as ctx:
            calc.execute_instruction(["-", 0, 1, 2], step=0)
        self.assertIn("not a positive integer", str(ctx.exception))

        calc2 = ElfCalculator(initial_registers={0: 3})
        with self.assertRaises(SimulationError) as ctx2:
            calc2.execute_instruction(["-", 0, 0, 1], step=0)
        self.assertIn("not a positive integer", str(ctx2.exception))

    def test_baseline_solver(self):
        puzzle = Puzzle(
            id="test_0001",
            starting_values=[3],
            targets=[1, 2, 7, 13, 42],
            value_limit=100000
        )
        solver = BaselineSolver(puzzle)
        instructions = solver.solve()

        self.assertGreater(len(instructions), 0)
        calc = ElfCalculator(puzzle.initial_registers, value_limit=puzzle.value_limit)
        is_valid, remaining, error = calc.run_program(instructions, targets=set(puzzle.targets))
        self.assertTrue(is_valid, f"Baseline solver failed simulation: {error}")
        self.assertEqual(len(remaining), 0)

    def test_dag_portfolio_solver(self):
        puzzle = Puzzle(
            id="test_dag_0001",
            starting_values=[631, 889, 1205, 1563],
            targets=[1, 2, 1024, 1563, 100855253734114, 125045864139519],
            value_limit=1445817928925762
        )
        solver = DAGPortfolioSolver(puzzle)
        instructions = solver.solve()

        self.assertGreater(len(instructions), 0)
        calc = ElfCalculator(puzzle.initial_registers, value_limit=puzzle.value_limit)
        is_valid, remaining, error = calc.run_program(instructions, targets=set(puzzle.targets))
        self.assertTrue(is_valid, f"DAGPortfolioSolver failed simulation: {error}")
        self.assertEqual(len(remaining), 0)


if __name__ == "__main__":
    unittest.main()
