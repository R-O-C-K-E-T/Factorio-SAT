from typing import Tuple
import enum


@enum.unique
class Direction(enum.Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

    def __init__(self, value: int):
        self.vec: Tuple[int, int] = [(1, 0), (0, -1), (-1, 0), (0, 1)][value]

    @staticmethod
    def from_factorio(direction: int):
        return Direction((1 - (direction // 2)) % 4)

    def add(self, rotation: int):
        assert isinstance(rotation, int)
        return Direction((self.value + rotation) % 4)

    def __index__(self):
        return self.value

    @property
    def dx(self):
        return self.vec[0]

    @property
    def dy(self):
        return self.vec[1]

    @property
    def next(self):
        return self.add(1)

    @property
    def prev(self):
        return self.add(-1)

    @property
    def reverse(self):
        return self.add(2)

    @property
    def factorio_direction(self) -> int:
        return ((1 - self.value) % 4) * 2

    @property
    def axis(self):
        return Axis(self.value % 2)


@enum.unique
class Axis(enum.Enum):
    HORIZONTAL = 0
    VERTICAL = 1

    def __index__(self):
        return self.value

    @property
    def directions(self):
        return Direction(self.value), Direction(self.value + 2)
