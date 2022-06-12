import argparse
import json
from enum import Enum
from typing import Callable

import numpy as np

from tile import BaseTile, TransformableTile


class Operation(Enum):
    NO_OP = (lambda tile: tile, lambda grid: grid)
    ROT_90 = (lambda tile: tile.rotate_90(), lambda grid: np.rot90(grid, k=1))
    ROT_180 = (lambda tile: tile.rotate_180(), lambda grid: np.rot90(grid, k=2))
    ROT_270 = (lambda tile: tile.rotate_270(), lambda grid: np.rot90(grid, k=3))
    FLIP_X = (lambda tile: tile.flip_x(), lambda grid: grid[:, ::-1])
    FLIP_Y = (lambda tile: tile.flip_y(), lambda grid: grid[::-1, :])

    def __init__(self, tile: Callable[[TransformableTile], TransformableTile], grid: Callable[[np.ndarray], np.ndarray]):
        self.tile = tile
        self.grid = grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply transformations to tile grids')
    parser.add_argument('operation', nargs='?', type=str, choices=[op.name.lower() for op in Operation], default=Operation.ROT_90.name.lower())
    args = parser.parse_args()
    operation = Operation[args.operation.upper()]

    while True:
        solution = np.array(json.loads(input()))
        solution = operation.grid(solution)
        for cell in solution.flat:
            tile = BaseTile.read(cell['tile'])
            if isinstance(tile, TransformableTile):
                cell['tile'] = operation.tile(tile).write()

        print(json.dumps(solution.tolist()))
