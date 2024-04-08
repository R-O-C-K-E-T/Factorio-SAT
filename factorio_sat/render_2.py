import contextlib
import json
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
import threading

import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import numpy as np

from . import blueprint
from .template import BoolTemplate, CompositeTemplate, NumberTemplate, OneHotTemplate, Template
from .tile_renderer import Coord, RenderLayer, TileRenderer, make_renderer
from .tile import BaseTile, BeltConnectedTile, Splitter

DPGTag = Union[int, str]


@dataclass(frozen=True)
class TileData:
    metadata: Dict[str, Any]
    tile: BaseTile

    @property
    def animation_length(self):
        if isinstance(self.tile, Splitter):
            return 32
        elif isinstance(self.tile, BeltConnectedTile):
            return 16
        else:
            return 1


@dataclass(frozen=True)
class GridData:
    metadata: Dict[str, Any]
    template: CompositeTemplate
    tiles: np.ndarray

    def __post_init__(self):
        assert len(self.tiles.shape) == 2
        assert self.tiles.shape[0] > 0
        assert self.tiles.shape[1] > 0

    @property
    def animation_length(self):
        return max(tile.animation_length for tile in self.tiles.reshape(-1))

    def at(self, coord: Coord[int]) -> TileData:
        return self.tiles[coord.y, coord.x]

    def get_tile_metadata(self, path: Iterable[Union[int, str]]) -> np.ndarray:
        metadata = np.vectorize(lambda tile: tile.metadata, otypes=[object])(self.tiles)

        while len(path) > 0:
            key, *path = path
            metadata = np.vectorize(lambda tile: tile[key], otypes=[object])(metadata)

        return metadata


@dataclass(frozen=True)
class GridSetupData:
    window_id: DPGTag
    drawlist_id: DPGTag


@dataclass(frozen=True)
class ControlState:
    switch_frame_buttons: List[Tuple[DPGTag, Callable[[], int]]]
    switch_frame_held: Dict[DPGTag, int]
    metadata: DPGTag


class Overlay:
    def render(self, frame: int, cell_size: int):
        raise NotImplementedError


def hsv_to_rgb(hue: float, sat: float, val: float):
    c = val * sat
    hue = (hue % 1) * 6
    x = c * (1 - abs((hue % 2) - 1))
    if hue > 5:
        out = c, 0, x
    elif hue > 4:
        out = x, 0, c
    elif hue > 3:
        out = 0, x, c
    elif hue > 2:
        out = 0, c, x
    elif hue > 1:
        out = x, c, 0
    else:
        out = c, x, 0
    return np.array(out) + (val - c)


def create_palette(size: int):
    if size == 1:
        return np.array([[1.0, 1.0, 1.0]])
    elif size <= 3:
        return np.array([
            [1, 0.2, 0.2],
            [0.2, 1, 0.2],
            [0.2, 0.2, 1],
        ])[:size]
    else:
        return np.array([hsv_to_rgb(i / size, 0.8, 1.0) for i in range(size)])


@dataclass(frozen=True)
class ColourOverlay(Overlay):
    colours: np.ndarray

    def __post_init__(self):
        assert len(self.colours.shape) == 3
        assert self.colours.shape[-1] == 4

    @staticmethod
    def from_single(arr: np.ndarray, opacity: float = 0.2):
        assert len(arr.shape) == 2

        all_colours = set()
        for item in arr.reshape(-1):
            all_colours.add(item)
        all_colours.discard(None)
        palette = create_palette(len(all_colours))
        palette = dict(zip(sorted(all_colours), palette))

        result = np.empty((*arr.shape, 4), dtype=float)
        for idx, val in np.ndenumerate(arr):
            if val is None:
                result[idx] = [0, 0, 0, 0]
            else:
                result[idx] = [*palette[val], opacity]

        return ColourOverlay(result)

    def render(self, frame: int, cell_size: int):
        for idx in np.ndindex(self.colours.shape[:2]):
            y, x = idx
            colour = (self.colours[idx] * 255).astype(int)
            lower = Coord(x, y)

            dpg.draw_rectangle(
                lower * cell_size,
                (lower + 1) * cell_size,
                color=(0, 0, 0, -255),
                fill=colour.tolist()
            )


@contextlib.contextmanager
def enter_widget(tag: DPGTag):
    dpg.push_container_stack(tag)
    try:
        yield None
    finally:
        dpg.pop_container_stack()


@dataclass
class OverlaySource:
    name: str
    property_path: Tuple[str, ...]
    data_converter: Callable[[np.ndarray], Overlay]

    def generate(self, grid: GridData):
        data = grid.get_tile_metadata(self.property_path)
        return self.data_converter(data)


class App:
    def __init__(self, tile_renderer: TileRenderer[BaseTile], initial_frames: List[GridData], cell_size=32) -> None:
        assert len(initial_frames) > 0
        self.tile_renderer = tile_renderer
        self.grids: List[GridData] = list(initial_frames)
        self.current_grid_index: int = 0
        self.cell_size = cell_size
        self.overlays: List[Overlay] = []
        self.overlay_sources: List[OverlaySource] = []
        self.active_cell: Optional[Coord[int]] = None

        self.grid_setup: GridSetupData
        self.control_state: ControlState

    @property
    def current_grid(self):
        return self.grids[self.current_grid_index]

    @property
    def current_tiles(self):
        return self.current_grid.tiles

    def set_current_grid(self, index: int):
        self.current_grid_index = index
        self.set_active_tile(self.control_state.metadata, self.active_cell)
        self.overlays.clear()
        for source in self.overlay_sources:
            self.overlays.append(source.generate(self.current_grid))

    def run(self):
        width = 250 + self.cell_size * self.current_grid.tiles.shape[1]
        height = max(200, 16 + self.cell_size * self.current_grid.tiles.shape[0])

        dpg.create_viewport(title='', width=width, height=height)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        self.setup()
        # dpg.show_style_editor()
        while dpg.is_dearpygui_running():
            dpg.set_viewport_title(f'{self.current_grid_index + 1} of {len(self.grids)}')
            self.render_control(self.control_state)
            self.render_grid(self.grid_setup, self.control_state)
            dpg.render_dearpygui_frame()

    def setup(self):
        with dpg.window(tag='Primary', no_scrollbar=True) as window:
            with dpg.table(header_row=False):
                dpg.add_table_column()
                dpg.add_table_column(width_fixed=True)

                with dpg.table_row():
                    with dpg.table_cell():
                        self.control_state = self.setup_control()
                    with dpg.table_cell():
                        self.grid_setup = self.setup_grid()

        self.set_active_tile(self.control_state.metadata, self.active_cell)

        dpg.set_viewport_vsync(True)
        dpg.set_primary_window(window, True)

    def setup_control(self):
        with dpg.child_window(height=50):
            with dpg.group(horizontal=True):
                switch_frame_buttons = [
                    (dpg.add_button(label='Start'), lambda: 0),
                    (dpg.add_button(label='Prev'), lambda: max(0, self.current_grid_index - 1)),
                    (dpg.add_button(label='Next'), lambda: min(len(self.grids) - 1, self.current_grid_index + 1)),
                    (dpg.add_button(label='End'), lambda: len(self.grids) - 1),
                ]

        metadata = dpg.add_child_window(show=False)

        return ControlState(
            switch_frame_buttons=switch_frame_buttons,
            switch_frame_held={},
            metadata=metadata,
        )

    def setup_grid(self) -> GridSetupData:
        with dpg.child_window() as window_id:
            drawlist_id = dpg.add_drawlist(width=1, height=1)
        return GridSetupData(window_id, drawlist_id)

    def build_overlay_sources(self, path: Tuple[str, ...], template: Template) -> List[OverlaySource]:
        sources: List[OverlaySource] = []

        if isinstance(template, BoolTemplate):
            def hightlight(vals: np.ndarray):
                colours = np.zeros((*vals.shape, 4))
                colours[:, :, :3] = 1.0
                colours[vals.astype(bool), 3] = 0.2
                return ColourOverlay(colours)
            sources.append(OverlaySource('Highlight', path, hightlight))
        elif isinstance(template, NumberTemplate) or isinstance(template, OneHotTemplate):
            sources.append(OverlaySource('Colour', path, lambda arr: ColourOverlay.from_single(arr)))

        return sources

    def add_overlay_source(self, source: OverlaySource):
        self.overlay_sources.append(source)
        self.overlays.append(source.generate(self.current_grid))

    def set_active_tile(self, metadata_window: DPGTag, coord: Optional[Coord[int]]):
        if coord is None or coord.x < 0 or coord.y < 0 or coord.x >= self.current_tiles.shape[1] or coord.y >= self.current_tiles.shape[0]:
            coord = None

        self.active_cell = coord

        if coord is None:
            self.show_grid_metadata(metadata_window)
            return

        dpg.delete_item(metadata_window, children_only=True)
        dpg.configure_item(metadata_window, show=True)

        tile = self.current_grid.at(coord)

        template = self.current_grid.template
        with enter_widget(metadata_window):
            dpg.add_text(f'<{coord.x},{coord.y}> Metadata')

            for key, val in sorted(tile.metadata.items(), key=lambda pair: pair[0]):
                item_template = template.atomics.get(key)

                with dpg.group(horizontal=True):
                    label = dpg.add_text(key + ':', color=(255, 255, 255, 255))
                    dpg.add_text(repr(val), color=(200, 200, 200, 255))

                    options = []
                    if item_template is not None:
                        options = self.build_overlay_sources((key,), item_template)

                    if len(options) != 0:
                        with dpg.popup(label, mousebutton=dpg.mvMouseButton_Right):
                            for source in options:
                                dpg.add_selectable(label=source.name, user_data=source, callback=lambda a, e, source: self.add_overlay_source(source))

    def show_grid_metadata(self, metadata_window: DPGTag):
        dpg.delete_item(metadata_window, children_only=True)
        dpg.configure_item(metadata_window, show=self.current_grid.metadata != {})

        if self.current_grid.metadata == {}:
            return

        with enter_widget(metadata_window):
            dpg.add_text('Grid Metadata')
            for key, val in sorted(self.current_grid.metadata.items(), key=lambda pair: pair[0]):
                with dpg.group(horizontal=True):
                    dpg.add_text(key + ':', color=(255, 255, 255, 255))
                    dpg.add_text(repr(val), color=(200, 200, 200, 255))

    def render_grid(self, grid_setup: GridSetupData, control_state: ControlState):
        rows, cols = self.current_tiles.shape
        width = cols * self.cell_size
        height = rows * self.cell_size

        # TODO Why can't I query this info from the theme?
        padding = 8
        dpg.configure_item(grid_setup.window_id, width=width + 2 * padding, height=height + 2 * padding)

        dpg.delete_item(grid_setup.drawlist_id, children_only=True)
        dpg.configure_item(grid_setup.drawlist_id, width=width, height=height)
        with enter_widget(grid_setup.drawlist_id):
            for layer in RenderLayer:
                for idx, tile_data in np.ndenumerate(self.current_tiles):
                    self.tile_renderer.render(tile_data.tile, Coord(*idx[::-1]), dpg.get_frame_count(), self.cell_size, layer)

            for overlay in self.overlays:
                overlay.render(dpg.get_frame_count(), self.cell_size)

            grid_coord = (Coord(*dpg.get_drawing_mouse_pos())) / self.cell_size

            if dpg.is_item_hovered(grid_setup.drawlist_id)\
                    and grid_coord.x >= 0\
                    and grid_coord.y >= 0\
                    and grid_coord.x < self.current_tiles.shape[1]\
                    and grid_coord.y < self.current_tiles.shape[0]:
                floored = Coord(int(grid_coord.x), int(grid_coord.y))

                if dpg.is_item_clicked(grid_setup.drawlist_id):
                    if floored == self.active_cell:
                        self.set_active_tile(control_state.metadata, None)
                    else:
                        self.set_active_tile(control_state.metadata, floored)

                dpg.draw_rectangle(floored * self.cell_size, (floored + 1) * self.cell_size, color=(0, 0, 0, -255), fill=(255, 255, 255, 100))

    def render_control(self, setup: ControlState):
        for button_id, proposed_frame_callback in setup.switch_frame_buttons:
            requested_frame = proposed_frame_callback()
            if requested_frame == self.current_grid_index:
                dpg.configure_item(button_id, enabled=False)
                continue
            dpg.configure_item(button_id, enabled=True)

            if dpg.is_item_hovered(button_id) and dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
                held_for = setup.switch_frame_held.get(button_id, 0)
                if held_for == 0 or (held_for > 30 and held_for % 2 == 0):
                    self.set_current_grid(requested_frame)

                setup.switch_frame_held[button_id] = held_for + 1
            else:
                setup.switch_frame_held.pop(button_id, None)


def main():
    grid_data = parse_line(input())

    dpg.create_context()
    tile_renderer = make_renderer()
    app = App(tile_renderer, [grid_data], cell_size=32)

    def process_input():
        while True:
            try:
                line = input()
            except (ValueError, EOFError):
                break
            grid_data = parse_line(line)
            app.grids.append(grid_data)

    thread = threading.Thread(target=process_input)
    thread.start()

    try:
        app.run()
    finally:
        sys.stdin.close()
        thread.join()

    dpg.destroy_context()


flyweight = lru_cache(lambda x: x)
template_cache = lru_cache(lambda x: Template.read(json.loads(x)))
tile_cache = lru_cache(lambda x: blueprint.read_tile(json.loads(x)))


def parse_line(line: str):
    json_data = json.loads(line)

    if isinstance(json_data, dict):
        metadata = json_data.get('metadata', {})

        template = template_cache(json.dumps(json_data['template']))

        assert isinstance(template, CompositeTemplate)
        tiles = np.array(json_data['tiles'], ndmin=2, dtype=object)
    elif isinstance(json_data, list):
        tiles = np.array(json_data, ndmin=2, dtype=object)
        metadata = {}
        template = CompositeTemplate({})
    else:
        raise RuntimeError('Invalid input')

    template = flyweight(template)

    for ind, data in np.ndenumerate(tiles):
        tile = tile_cache(json.dumps(data))
        data.pop('tile', None)
        tiles[ind] = TileData(
            metadata=data,
            tile=tile,
        )

    return GridData(
        metadata=metadata,
        template=template,
        tiles=tiles
    )


def run_demo():
    dpg.create_context()
    dpg.create_viewport(title='Custom Title', width=600, height=600)

    demo.show_demo()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == '__main__':
    main()
