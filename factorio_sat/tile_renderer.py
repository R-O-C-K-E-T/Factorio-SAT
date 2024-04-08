from dataclasses import dataclass
import enum
from os import path
from typing import Dict, Protocol, Type, Union, TypeVar

import dearpygui.dearpygui as dpg

from .direction import Direction
from .tile import BaseTile, Belt, Splitter, UndergroundBelt
from .coord import Coord


class RenderLayer(enum.Enum):
    BOTTOM = enum.auto()
    TOP = enum.auto()


T = TypeVar('T')
TTile = TypeVar('TTile', bound=BaseTile)


class TileRenderer(Protocol[TTile]):
    def render(self, tile: Belt, coord: Coord[int], frame: int, pixels_per_tile: int, layer: RenderLayer):
        ...


class TileRendererRegistry(TileRenderer[BaseTile]):
    def __init__(self):
        self._registry: Dict[Union[BaseTile, Type[BaseTile]], TileRenderer[BaseTile]] = {}

    def register(self, key: Union[TTile, Type[TTile]], renderer: TileRenderer[TTile]):
        assert key not in self._registry
        self._registry[key] = renderer

    def render(self, tile: Belt, coord: Coord[int], frame: int, pixels_per_tile: int, layer: RenderLayer):
        renderer = self._registry.get(tile)
        if renderer is None:
            renderer = self._registry.get(type(tile))

        if renderer is None:
            # raise KeyError(f'Failed to find renderer for {tile}')
            return

        renderer.render(tile, coord, frame, pixels_per_tile, layer)


@dataclass(frozen=True)
class Texture:
    size: Coord[int]
    tag: Union[int, str]

    def load(name: str) -> 'Texture':
        base_path = path.join(path.dirname(__file__), 'assets')
        width, height, _, data = dpg.load_image(path.join(base_path, name))

        with dpg.texture_registry():
            tag_id = dpg.add_static_texture(width, height, data)

        return Texture(
            Coord(width, height),
            tag_id
        )

    def draw(self, screen_coord: Coord[float], screen_size: Coord[float], image_coord: Coord[float], image_size: Coord[float]):
        uv_min = image_coord / self.size
        uv_max = (image_coord + image_size) / self.size

        dpg.draw_image(self.tag, screen_coord, screen_coord + screen_size, uv_min=uv_min, uv_max=uv_max)


@dataclass(frozen=True)
class Tilemap:
    texture: Texture
    entry_size: Coord[int]
    pixels_per_unit: int

    def __post_init__(self):
        assert self.texture.size % self.entry_size == Coord(0, 0)

    def size_in_screen_coord(self, screen_pixels_per_unit: int) -> Coord[float]:
        return self.entry_size * screen_pixels_per_unit / self.pixels_per_unit

    def draw(
        self,
        screen_coord: Coord[float],
        screen_pixels_per_unit: int,
        entry: Coord[int, int],
        lower: Coord[float] = Coord(0, 0),
        upper: Coord[float] = Coord(1, 1)
    ):
        self.texture.draw(
            screen_coord,
            self.size_in_screen_coord(screen_pixels_per_unit) * (upper - lower),
            (entry + lower) * self.entry_size,
            self.entry_size * (upper - lower)
        )


@dataclass(frozen=True)
class BeltRenderer(TileRenderer[Belt]):
    tilemap: Tilemap

    def render(self, tile: Belt, coord: Coord[int], frame: int, pixels_per_tile: int, layer: RenderLayer):
        if layer != RenderLayer.BOTTOM:
            return

        frame = frame % 16
        index = ((11, 8, 4, 7), (0, 2, 1, 3), (6, 10, 9, 5))[(tile.output_direction.value - tile.input_direction.value + 1) % 4][tile.input_direction]
        self.tilemap.draw(
            (coord - 0.5) * pixels_per_tile,
            pixels_per_tile,
            Coord(frame, index)
        )


@dataclass(frozen=True)
class UndergroundBeltRenderer(TileRenderer[UndergroundBelt]):
    belt_tilemap: Tilemap
    structure_tilemap: Tilemap

    def render(self, tile: UndergroundBelt, coord: Coord[int], frame: int, pixels_per_tile: int, layer: RenderLayer):
        if layer == RenderLayer.BOTTOM:
            frame = frame % 16
            index = ((14, 12, 18, 16), (19, 17, 15, 13))[tile.is_input][tile.direction]
            self.belt_tilemap.draw(
                (coord - 0.5) * pixels_per_tile,
                pixels_per_tile,
                Coord(frame, index)
            )
        elif layer == RenderLayer.TOP:
            self.structure_tilemap.draw(
                (coord - 1) * pixels_per_tile,
                pixels_per_tile,
                Coord([[3, 2, 1, 0], [1, 0, 3, 2]][tile.is_input][tile.direction], int(tile.is_input)),
            )


@dataclass(frozen=True)
class SplitterRenderer(TileRenderer[Splitter]):
    belt: Tilemap
    splitter_east: Tilemap
    splitter_east_top_patch: Tilemap
    splitter_west: Tilemap
    splitter_west_top_patch: Tilemap
    splitter_south: Tilemap
    splitter_north: Tilemap

    def render(self, tile: Splitter, coord: Coord[int], frame: int, pixels_per_tile: int, layer: RenderLayer):
        if layer == RenderLayer.BOTTOM:
            frame = frame % 16
            self.belt.draw(
                (coord - 0.5) * pixels_per_tile,
                pixels_per_tile,
                Coord(frame, (0, 2, 1, 3)[tile.direction])
            )
        else:
            index = Coord(*divmod(frame % 32, 8)[::-1])
            if tile.direction == Direction.RIGHT:
                if tile.is_head:
                    self.splitter_east.draw(
                        (coord - Coord(0, 5 / 16)) * pixels_per_tile,
                        pixels_per_tile,
                        index
                    )
                else:
                    self.splitter_east_top_patch.draw(
                        (coord - Coord(0, 8 / 16)) * pixels_per_tile,
                        pixels_per_tile,
                        index
                    )
            elif tile.direction == Direction.LEFT:
                if tile.is_head:
                    self.splitter_west_top_patch.draw(
                        (coord - Coord(0, 5 / 16)) * pixels_per_tile,
                        pixels_per_tile,
                        index
                    )
                else:
                    self.splitter_west.draw(
                        (coord - Coord(0, 5 / 16)) * pixels_per_tile,
                        pixels_per_tile,
                        index
                    )
            elif tile.direction == Direction.UP:
                if tile.is_head:
                    self.splitter_north.draw(
                        coord * pixels_per_tile,
                        pixels_per_tile,
                        index,
                        lower=Coord(13 / 32, 0)
                    )
                else:
                    self.splitter_north.draw(
                        coord * pixels_per_tile,
                        pixels_per_tile,
                        index,
                        upper=Coord(13 / 32, 1)
                    )
            elif tile.direction == Direction.DOWN:
                if tile.is_head:
                    self.splitter_south.draw(
                        (coord - Coord(4 / 32, 0)) * pixels_per_tile,
                        pixels_per_tile,
                        index,
                        upper=Coord(14 / 32, 1)
                    )
                else:
                    self.splitter_south.draw(
                        coord * pixels_per_tile,
                        pixels_per_tile,
                        index,
                        lower=Coord(14 / 32, 0)
                    )


def make_renderer() -> TileRenderer[BaseTile]:
    px_per_unit = 64
    registry = TileRendererRegistry()
    belt_tilemap = Tilemap(Texture.load('hr-transport-belt.png'), Coord(2, 2) * px_per_unit, px_per_unit)
    structure_tilemap = Tilemap(Texture.load('hr-underground-belt-structure.png'), Coord(3, 3) * px_per_unit, px_per_unit)

    registry.register(Belt,  BeltRenderer(belt_tilemap))
    registry.register(UndergroundBelt, UndergroundBeltRenderer(belt_tilemap, structure_tilemap))
    registry.register(Splitter, SplitterRenderer(
        belt_tilemap,
        Tilemap(Texture.load('hr-splitter-east.png'), Coord(90, 84), px_per_unit),
        Tilemap(Texture.load('hr-splitter-east-top_patch.png'), Coord(90, 104), px_per_unit),
        Tilemap(Texture.load('hr-splitter-west.png'), Coord(90, 86), px_per_unit),
        Tilemap(Texture.load('hr-splitter-west-top_patch.png'), Coord(90, 96), px_per_unit),
        Tilemap(Texture.load('hr-splitter-south.png'), Coord(164, 64), px_per_unit),
        Tilemap(Texture.load('hr-splitter-north.png'), Coord(160, 70), px_per_unit)
    ))

    return registry
