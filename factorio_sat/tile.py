from typing import Any, ClassVar, Dict, Optional
from dataclasses import dataclass

from .direction import Axis, Direction

TILE_TYPES = {}


class BaseTile:
    type_key: ClassVar[str]

    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'type_key'):
            if cls.type_key in TILE_TYPES:
                raise RuntimeError(f'Duplicate tile type key: {repr(cls.type_key)}')
            TILE_TYPES[cls.type_key] = cls

    def write(self) -> Dict[str, Any]:
        return {'type': self.type_key}

    @staticmethod
    def read(json_dict: Dict[str, Any]) -> 'BaseTile':
        return TILE_TYPES[json_dict['type']].read(json_dict)


class TransformableTile(BaseTile):
    def rotate_90(self) -> 'TransformableTile':
        raise NotImplementedError

    def flip_x(self) -> 'TransformableTile':
        raise NotImplementedError

    def rotate_180(self) -> 'TransformableTile':
        return self.rotate_90().rotate_90()

    def rotate_270(self) -> 'TransformableTile':
        return self.rotate_180().rotate_90()

    def flip_y(self) -> 'TransformableTile':
        return self.rotate_90().flip_x().rotate_270()


@dataclass(frozen=True)
class EmptyTile(BaseTile):
    type_key: ClassVar[str] = 'empty'

    @classmethod
    def read(cls, json_dict: Dict[str, Any]):
        return cls()


class BeltConnectedTile(BaseTile):
    input_direction: Optional[Direction]
    output_direction: Optional[Direction]


def flip_x_direction(direction: Direction):
    return direction.reverse if direction.axis == Axis.HORIZONTAL else direction


@dataclass(frozen=True)
class Belt(BeltConnectedTile, TransformableTile):
    type_key: ClassVar[str] = 'belt'

    input_direction: Direction
    output_direction: Direction

    def flip_x(self) -> 'Belt':
        return type(self)(flip_x_direction(self.input_direction), flip_x_direction(self.output_direction))

    def rotate_90(self) -> 'Belt':
        return type(self)(self.input_direction.next, self.output_direction.next)

    def __post_init__(self):
        if self.input_direction.reverse == self.output_direction:
            raise RuntimeError(f'Cannot input and output from same side ({self.input_direction}, {self.output_direction})')

    def write(self) -> Dict[str, Any]:
        return {
            **super().write(),
            'input_direction': int(self.input_direction),
            'output_direction': int(self.output_direction),
        }

    @classmethod
    def read(cls, json_dict: Dict[str, Any]):
        return cls(Direction(json_dict['input_direction']), Direction(json_dict['output_direction']))


@dataclass(frozen=True)
class UndergroundBelt(BeltConnectedTile, TransformableTile):
    type_key: ClassVar[str] = 'underground_belt'

    direction: Direction
    is_input: bool

    @property
    def input_direction(self) -> Optional[Direction]:
        return self.direction if self.is_input else None

    @property
    def output_direction(self) -> Optional[Direction]:
        return None if self.is_input else self.direction

    def flip_x(self) -> 'UndergroundBelt':
        return type(self)(flip_x_direction(self.direction), self.is_input)

    def rotate_90(self) -> 'UndergroundBelt':
        return type(self)(self.direction.next, self.is_input)

    def write(self) -> Dict[str, Any]:
        return {
            **super().write(),
            'direction': int(self.direction),
            'is_input': self.is_input,
        }

    @classmethod
    def read(cls, json_dict: Dict[str, Any]):
        return cls(Direction(json_dict['direction']), json_dict['is_input'])


@dataclass(frozen=True)
class Splitter(BeltConnectedTile, TransformableTile):
    type_key: ClassVar[str] = 'splitter'

    direction: Direction
    is_head: bool

    @property
    def input_direction(self) -> Direction:
        return self.direction

    @property
    def output_direction(self) -> Direction:
        return self.direction

    def flip_x(self) -> 'Splitter':
        return type(self)(flip_x_direction(self.direction), not self.is_head)

    def rotate_90(self) -> 'Splitter':
        return type(self)(self.direction.next, self.is_head)

    def write(self) -> Dict[str, Any]:
        return {
            **super().write(),
            'direction': int(self.direction),
            'is_head': self.is_head,
        }

    @classmethod
    def read(cls, json_dict: Dict[str, Any]):
        return cls(Direction(json_dict['direction']), json_dict['is_head'])


@dataclass(frozen=True)
class Inserter(TransformableTile):
    type_key: ClassVar[str] = 'inserter'

    direction: Direction
    type: int  # 0 -> Normal, 1 -> Long

    def flip_x(self) -> 'Inserter':
        return type(self)(flip_x_direction(self.direction), self.type)

    def rotate_90(self) -> 'Inserter':
        return type(self)(self.direction.next, self.type)

    def write(self) -> Dict[str, Any]:
        return {
            **super().write(),
            'direction': int(self.direction),
            'insert_type': self.type,
        }

    @classmethod
    def read(cls, json_dict: Dict[str, Any]):
        return cls(Direction(json_dict['direction']), json_dict['insert_type'])


@dataclass(frozen=True)
class AssemblingMachine(BaseTile):
    type_key: ClassVar[str] = 'assembling_machine'

    x: int
    y: int

    def __post_init__(self):
        assert 0 <= self.x < 3
        assert 0 <= self.y < 3

    def write(self) -> Dict[str, Any]:
        return {
            **super().write(),
            'x': self.x,
            'y': self.y,
        }

    @classmethod
    def read(cls, json_dict: Dict[str, Any]):
        return cls(json_dict['x'], json_dict['y'])
