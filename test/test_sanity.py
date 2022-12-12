from factorio_sat import tile
from factorio_sat.direction import Direction
from test.grid_test_case import BaseGridTest


class TestSanity(BaseGridTest):
    def setUp(self):
        super().setUp()
        self.make_grid(5, 5)

    def test_empty_grid_sat(self):
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                self.grid.set_tile(x, y, tile.EmptyTile())
        self.assert_sat()

    def test_non_empty_grid_sat(self):
        self.grid.set_tile(2, 2, tile.Belt(Direction.RIGHT, Direction.RIGHT))
        self.assert_sat()

    def test_invalid_intersection(self):
        self.grid.set_tile(1, 2, tile.Belt(Direction.UP, Direction.UP))
        self.grid.set_tile(2, 2, tile.Belt(Direction.RIGHT, Direction.RIGHT))
        self.assert_unsat()
