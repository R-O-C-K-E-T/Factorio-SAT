from typing import NamedTuple, TypeVar, Union


T = TypeVar('T')


class Coord(NamedTuple):
    x: T
    y: T

    @staticmethod
    def _as_coord(val: 'Union[Coord[T], T]') -> 'Coord[T]':
        return val if isinstance(val, Coord) else Coord(val, val)

    def __add__(self, other: 'Union[Coord[T], T]') -> 'Coord[T]':
        other = self._as_coord(other)
        return Coord(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Union[Coord[T], T]') -> 'Coord[T]':
        other = self._as_coord(other)
        return Coord(self.x - other.x, self.y - other.y)

    def __mul__(self, other: 'Union[Coord[T], T]') -> 'Coord[T]':
        other = self._as_coord(other)
        return Coord(self.x * other.x, self.y * other.y)

    def __truediv__(self, other: 'Union[Coord[T], T]') -> 'Coord[T]':
        other = self._as_coord(other)
        return Coord(self.x / other.x, self.y / other.y)

    def __mod__(self, other: 'Union[Coord[T], T]') -> 'Coord[T]':
        other = self._as_coord(other)
        return Coord(self.x % other.x, self.y % other.y)
