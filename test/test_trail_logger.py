from dataclasses import dataclass
from typing import List, Optional
import unittest

import numpy as np
from factorio_sat import stringifier
from factorio_sat import solver
from factorio_sat.direction import Direction
from factorio_sat.coord import Coord
from factorio_sat.trail_logger import PathTracer, PathLocation, find_shortcuts


@dataclass(frozen=True)
class ShortcutCase:
    string_grid: str
    path_len: Optional[int] = None
    shortcut: Optional[List[PathLocation]] = None


SHORTCUT_CASES = [
    ShortcutCase(
        '''
        ┌─────┐
        │  k  │
        │G → →│
        │↑ K  │
        │F g  │
        └─────┘
        ''',
        9,
        [
            PathLocation(Coord(1, 0), None),
            PathLocation(Coord(1, 1), Direction.DOWN),
            PathLocation(Coord(1, 2), None),
            PathLocation(Coord(1, 3), None),
            PathLocation(Coord(0, 3), None),
            PathLocation(Coord(0, 2), None),
            PathLocation(Coord(0, 1), None),
            PathLocation(Coord(1, 1), None),
        ]
    ),
    ShortcutCase(
        '''
        ┌─────┐
        │  k  │
        │G → →│
        │↑ K  │
        │F g  │
        └─────┘
        ''',
        9,
        [
            PathLocation(Coord(1, 0), None),
            PathLocation(Coord(1, 1), Direction.DOWN),
            PathLocation(Coord(1, 2), None),
            PathLocation(Coord(1, 3), None),
            PathLocation(Coord(0, 3), None),
            PathLocation(Coord(0, 2), None),
            PathLocation(Coord(0, 1), None),
            PathLocation(Coord(1, 1), None),
        ],
    ),
    ShortcutCase(
        '''
        ┌─────┐
        │  I  │
        │H ← ←│
        │↓ i  │
        │T h  │
        └─────┘
        ''',
        9,
        [
            PathLocation(Coord(1, 1), None),
            PathLocation(Coord(0, 1), None),
            PathLocation(Coord(0, 2), None),
            PathLocation(Coord(0, 3), None),
            PathLocation(Coord(1, 3), None),
            PathLocation(Coord(1, 2), None),
            PathLocation(Coord(1, 1), Direction.UP),
            PathLocation(Coord(1, 0), None),
        ]
    ),
    ShortcutCase(
        '''
        ┌───┐
        │I k│
        │? ?│
        │i K│
        │F g│
        └───┘
        ''',
        8,
        [
            PathLocation(Coord(1, 0), None),
            PathLocation(Coord(1, 1), Direction.DOWN),
            PathLocation(Coord(1, 2), None),
            PathLocation(Coord(1, 3), None),
            PathLocation(Coord(0, 3), None),
            PathLocation(Coord(0, 2), None),
            PathLocation(Coord(0, 1), Direction.UP),
            PathLocation(Coord(0, 0), None),
        ]
    ),
    ShortcutCase(
        '''
        ┌───┐
        │↑ ↓│
        │F g│
        └───┘
        ''',
        4,
        [
            PathLocation(Coord(1, 0), None),
            PathLocation(Coord(1, 1), None),
            PathLocation(Coord(0, 1), None),
            PathLocation(Coord(0, 0), None),
        ]
    ),
    ShortcutCase(
        '''
        ┌─────┐
        │I    │
        │?    │
        │i G f│
        │↑ I ↓│
        │F ← g│
        │  ?  │
        │  i  │
        └─────┘
        ''',
        14,
        [
            PathLocation(Coord(1, 3), None),
            PathLocation(Coord(1, 2), None),
            PathLocation(Coord(2, 2), None),
            PathLocation(Coord(2, 3), None),
            PathLocation(Coord(2, 4), None),
            PathLocation(Coord(1, 4), None),
            PathLocation(Coord(0, 4), None),
            PathLocation(Coord(0, 3), None),
        ]
    ),
    ShortcutCase(
        '''
        ┌─────┐
        │  I  │
        │  ?  │
        │H ← t│
        │↓ ? ↑│
        │↓ i ↑│
        │T h I│
        │     │
        │    i│
        └─────┘
        ''',
        17,
        [
            PathLocation(Coord(2, 5), None),
            PathLocation(Coord(2, 4), None),
            PathLocation(Coord(2, 3), None),
            PathLocation(Coord(2, 2), None),
            PathLocation(Coord(1, 2), None),
            PathLocation(Coord(0, 2), None),
            PathLocation(Coord(0, 3), None),
            PathLocation(Coord(0, 4), None),
            PathLocation(Coord(0, 5), None),
            PathLocation(Coord(1, 5), None),
            PathLocation(Coord(2, 6), None),
        ]
    ),
    ShortcutCase(
        '''
        ┌─────┐
        │H ← t│
        │↓ ? ↑│
        │T f I│
        │? ↓  │
        │← g  │
        │     │
        │    i│
        └─────┘
        ''',
        15,
        [
            PathLocation(Coord(2, 2), None),
            PathLocation(Coord(2, 1), None),
            PathLocation(Coord(2, 0), None),
            PathLocation(Coord(1, 0), None),
            PathLocation(Coord(0, 0), None),
            PathLocation(Coord(0, 1), None),
            PathLocation(Coord(0, 2), None),
            PathLocation(Coord(1, 2), None),
            PathLocation(Coord(2, 3), None),
        ]
    ),
    ShortcutCase(
        '''
        ┌───┐
        │t H│
        │F g│
        └───┘
        ''',
        4,
        [
            PathLocation(Coord(1, 0), None),
            PathLocation(Coord(1, 1), None),
            PathLocation(Coord(0, 1), None),
            PathLocation(Coord(0, 0), None),
        ]
    ),
    ShortcutCase(
        '''
        ┌───────────┐
        │H ← ← t    │
        │T l ? I L →│
        │           │
        │      i    │
        └───────────┘
        ''',
        shortcut=[
            PathLocation(coord=Coord(x=3, y=1), underground_dir=None),
            PathLocation(coord=Coord(x=3, y=0), underground_dir=None),
            PathLocation(coord=Coord(x=2, y=0), underground_dir=None),
            PathLocation(coord=Coord(x=1, y=0), underground_dir=None),
            PathLocation(coord=Coord(x=0, y=0), underground_dir=None),
            PathLocation(coord=Coord(x=0, y=1), underground_dir=None),
            PathLocation(coord=Coord(x=1, y=1), underground_dir=None),
            PathLocation(coord=Coord(x=2, y=1), underground_dir=Direction.RIGHT),
            PathLocation(coord=Coord(x=3, y=1), underground_dir=Direction.RIGHT),
            PathLocation(coord=Coord(x=4, y=1), underground_dir=None),
            PathLocation(coord=Coord(x=3, y=2), underground_dir=None)
        ],
    ),
    ShortcutCase(
        '''
        ┌───────┐
        │      I│
        │       │
        │H ← t  │
        │↓ ? I i│
        │↓ ?   ↑│
        │T → → h│
        │    ?  │
        │    i  │
        └───────┘
        ''',
        19,
        [
            PathLocation(Coord(2, 3), None),
            PathLocation(Coord(2, 2), None),
            PathLocation(Coord(1, 2), None),
            PathLocation(Coord(0, 2), None),
            PathLocation(Coord(0, 3), None),
            PathLocation(Coord(0, 4), None),
            PathLocation(Coord(0, 5), None),
            PathLocation(Coord(1, 5), None),
            PathLocation(Coord(2, 5), None),
            PathLocation(Coord(3, 5), None),
            PathLocation(Coord(3, 4), None),
            PathLocation(Coord(3, 3), None),
            PathLocation(Coord(2, 4), None),
            PathLocation(Coord(3, 2), None)
        ]
    ),
    ShortcutCase(
        '''
        ┌───────┐
        │  k    │
        │G l   L│
        │↑ K    │
        │F g    │
        └───────┘
        ''',
    ),
    ShortcutCase(
        '''
        ┌───────┐
        │  I    │
        │  ?    │
        │G → → f│
        │I i ? ↓│
        │  F ← g│
        │       │
        │i      │
        └───────┘
        '''
    ),
    ShortcutCase(
        '''
        ┌───┐
        │→ f│
        │  ↓│
        │← g│
        └───┘
        ''',
        5,
        [
            PathLocation(Coord(0, 0), None),
            PathLocation(Coord(1, 0), None),
            PathLocation(Coord(1, 1), None),
            PathLocation(Coord(1, 2), None),
            PathLocation(Coord(0, 2), None),
            PathLocation(Coord(0, 1), None),
        ]
    ),
    ShortcutCase(
        '''
        ┌───┐
        │I  │
        │?  │
        │G f│
        │h k│
        │? ?│
        │i K│
        │F g│
        └───┘
        ''',
        14,
        [
            PathLocation(Coord(x=0, y=2), None),
            PathLocation(Coord(x=1, y=2), None),
            PathLocation(Coord(x=1, y=3), None),
            PathLocation(Coord(x=1, y=4), Direction.DOWN),
            PathLocation(Coord(x=1, y=5), None),
            PathLocation(Coord(x=1, y=6), None),
            PathLocation(Coord(x=0, y=6), None),
            PathLocation(Coord(x=0, y=5), None),
            PathLocation(Coord(x=0, y=4), Direction.UP),
            PathLocation(Coord(x=0, y=3), Direction.UP),
            PathLocation(Coord(x=0, y=2), Direction.UP)
        ]
    ),
    ShortcutCase(
        '''
        ┌───┐
        │k  │
        │?  │
        │H t│
        │g I│
        │? ?│
        │K i│
        │T h│
        └───┘
        ''',
        14,
        [
            PathLocation(Coord(x=0, y=2), Direction.DOWN),
            PathLocation(Coord(x=0, y=3), Direction.DOWN),
            PathLocation(Coord(x=0, y=4), Direction.DOWN),
            PathLocation(Coord(x=0, y=5), None),
            PathLocation(Coord(x=0, y=6), None),
            PathLocation(Coord(x=1, y=6), None),
            PathLocation(Coord(x=1, y=5), None),
            PathLocation(Coord(x=1, y=4), Direction.UP),
            PathLocation(Coord(x=1, y=3), None),
            PathLocation(Coord(x=1, y=2), None),
            PathLocation(Coord(x=0, y=2), None)
        ]
    ),
]

NON_SHORTCUT_CASES = [
    '''
    ┌───────┐
    │  k    │
    │G l ? L│
    │↑ K    │
    │F g    │
    └───────┘
    ''',
    '''
    ┌─────┐
    │G → →│
    │↑ ? ?│
    │F ← ←│
    └─────┘
    ''',
    '''
    ┌─────┐
    │  k  │
    │  ?  │
    │G → →│
    │↑ K  │
    │F g  │
    └─────┘
    ''',
    '''
    ┌─────┐
    │H ← t│
    │↓ ? ↑│
    │T f I│
    │? ↓ ?│
    │← g  │
    │     │
    │    i│
    └─────┘
    ''',
    '''
    ┌─────┐
    │H t  │
    │↓ I  │
    │T → →│
    │     │
    │  ?  │
    │  i  │
    └─────┘
    ''',
    '''
    ┌───────┐
    │      I│
    │       │
    │H ← t ?│
    │↓ ? I i│
    │↓ ?   ↑│
    │T → → h│
    │    ?  │
    │    i  │
    └───────┘
    '''
]


class TestPathTracer(unittest.TestCase):
    def make_path_tracer(self, string_grid: str) -> PathTracer:
        lines = [line.strip() for line in string_grid.strip().splitlines()]

        unknown = np.array([[char == '?' for char in line] for line in lines])[1:-1, 1:-1:2]
        lines = [line.replace('?', ' ') for line in lines]

        tiles = stringifier.decode(lines)

        # print(unknown.shape, unknown[1:-1, 1:-1:2].shape, tiles.shape)
        tiles[unknown] = None

        underground = self.derive_underground(tiles)

        return PathTracer(tiles, underground)

    def derive_underground(self, tiles: np.ndarray) -> np.ndarray:
        grid = solver.Grid(tiles.shape[1], tiles.shape[0], colours=None, underground_length=float('inf'))
        grid.prevent_intersection()
        grid.prevent_bad_undergrounding()

        for (y, x), tile in np.ndenumerate(tiles):
            if tile is not None:
                grid.set_tile(x, y, tile)

        solution = grid.solve()
        assert solution is not None

        underground = np.empty((*tiles.shape, 4), dtype=bool)
        for idx, tile in np.ndenumerate(solution):
            underground[idx] = tile['underground']

        return underground

    def test_can_trace_multiple_paths(self):
        tracer = self.make_path_tracer('''
┌─────┐
│    I│
│→ → →│
│→ f i│
│  T h│
└─────┘
        ''')

        paths = list(tracer.trace_all_paths())

        self.assertEqual(len(paths), 2)
        paths.sort(key=lambda path: path[0])

        self.assertEqual(paths[0], [
            PathLocation(Coord(0, 1), None),
            PathLocation(Coord(1, 1), None),
            PathLocation(Coord(2, 1), None),
        ])
        self.assertEqual(paths[1], [
            PathLocation(Coord(0, 2), None),
            PathLocation(Coord(1, 2), None),
            PathLocation(Coord(1, 3), None),
            PathLocation(Coord(2, 3), None),
            PathLocation(Coord(2, 2), None),
            PathLocation(Coord(2, 1), Direction.UP),
            PathLocation(Coord(2, 0), None),
        ])

    def test_shortcut_cases(self):
        for case in SHORTCUT_CASES:
            with self.subTest(case.string_grid):
                tracer = self.make_path_tracer(case.string_grid)

                paths = list(tracer.trace_all_paths())
                self.assertEqual(len(paths), 1)
                if case.path_len is not None:
                    self.assertEqual(len(paths[0]), case.path_len)

                shortcuts = list(find_shortcuts(tracer, paths[0]))
                for shortcut in shortcuts:
                    print(shortcut)
                self.assertEqual(len(shortcuts), 1)
                if case.shortcut is not None:
                    self.assertEqual(shortcuts[0], case.shortcut)

    def test_non_shortcutable_paths(self):
        for str_grid in NON_SHORTCUT_CASES:
            with self.subTest(str_grid.strip()):
                tracer = self.make_path_tracer(str_grid)
                paths = list(tracer.trace_all_paths())
                self.assertEqual(len(paths), 1)
                shortcuts = list(find_shortcuts(tracer, paths[0]))
                for shortcut in shortcuts:
                    print(shortcut)
                self.assertEqual(len(shortcuts), 0)
