from typing import Optional
import unittest

import numpy as np

from factorio_sat import solver
from factorio_sat.template import CompositeTemplateParams
from factorio_sat import stringifier
from factorio_sat import blueprint


class InvalidSolutionError(AssertionError):
    def __init__(self, solution: np.ndarray) -> None:
        solution = np.vectorize(blueprint.read_tile)(solution)
        solution = stringifier.encode(solution)
        self.solution_str = solution
        super().__init__()

    def __str__(self) -> str:
        return '\n' + self.solution_str


class BaseGridTest(unittest.TestCase):
    grid: solver.Grid

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.grid = None

    def make_grid(
            self,
            width: int,
            height: int,
            colours: Optional[int] = None,
            underground_length: int = float('inf'),
            extras: CompositeTemplateParams = {},
            enforce_basic_rules=True
    ):
        self.grid = solver.Grid(width, height, colours, underground_length, extras)
        if enforce_basic_rules:
            self.enforce_basic_rules()

    def set_content(self, string_grid: str, *args, **kwargs):
        tiles = stringifier.decode([line.strip() for line in string_grid.strip().splitlines()])
        if self.grid is None:
            self.make_grid(tiles.shape[1], tiles.shape[0], *args, **kwargs)
        else:
            self.assertEqual(tiles.shape, (self.grid.height, self.grid.width), string_grid)
            assert len(args) == 0
            assert len(kwargs) == 0

        for (y, x), tile in np.ndenumerate(tiles):
            self.grid.set_tile(x, y, tile)

    def enforce_basic_rules(self):
        self.grid.prevent_intersection()
        self.grid.enforce_maximum_underground_length()
        self.grid.prevent_bad_undergrounding()
        self.grid.prevent_bad_colouring()

    def assert_sat(self):
        solution = self.grid.solve()
        self.assertIsNotNone(solution, 'Grid does not have a solution')

    def assert_unsat(self):
        solution = self.grid.solve()
        if solution is not None:
            raise InvalidSolutionError(solution)
