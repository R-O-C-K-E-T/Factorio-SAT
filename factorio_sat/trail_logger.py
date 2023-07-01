from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple, Union
from collections import defaultdict
import time
import json
import sys
import numpy as np


from .template import ArrayTemplate, BoolTemplate, CompositeTemplate, ManyHotTemplate, NumberTemplate, OneHotTemplate, Template

from . import belt_balancer
from . import interchange
from . import belt_balancer_net_free_power_of_2
from . import optimisations
from . import blueprint
from . import stringifier
from .util import ClauseList, ClauseType, LiteralType, invert_components
from .direction import Direction
from .network import deduplicate_network, open_network
from .solver import Grid, TileTemplate
from .coord import Coord
from . import tile as tile_type


from .ipasir import IPASIRLibrary


def get_all_tiles() -> Set[tile_type.BaseTile]:
    result = {
        tile_type.EmptyTile(),
    }

    for direction in Direction:
        result |= {
            tile_type.Belt(direction, direction.prev),
            tile_type.Belt(direction, direction),
            tile_type.Belt(direction, direction.next),
            tile_type.UndergroundBelt(direction, False),
            tile_type.UndergroundBelt(direction, True),
            tile_type.Splitter(direction, False),
            tile_type.Splitter(direction, True),
        }

    return result


@dataclass(frozen=True)
class LiteralLabel:
    is_negative: bool
    location: Tuple[int, int]
    path: str

    def __str__(self) -> str:
        path = f'[{self.location[0]},{self.location[1]}]{self.path}'
        if self.is_negative:
            path = '-' + path
        return path


class EliminationMap:
    def __init__(self, grid: Grid) -> None:
        self.grid = grid

        initial_len = len(grid.clauses)
        prev_len = initial_len

        self.literal_map: Dict[LiteralType, List[Tuple[Tuple[LiteralType, ...], int, int, tile_type.BaseTile]]] = defaultdict(list)

        all_tiles = get_all_tiles()
        for y in range(grid.height):
            for x in range(grid.width):
                prev_len = len(self.grid.clauses)
                for tile_instance in all_tiles:
                    # TODO Remove very cheeky mutation of grid.clauses
                    self.grid.set_tile(x, y, tile_instance)
                    new_clauses = self.grid.clauses[prev_len:]
                    del self.grid.clauses[prev_len:]

                    for clause in new_clauses:
                        clause = invert_components(clause)

                        # TODO splitter [tile.input_direction, tile.output_direction] is suspiciously strengthen to [tile.input_direction]
                        del clause[1:]
                        self.literal_map[clause[0]].append((tuple(clause[1:]), x, y, tile_instance))

        def visit(template: Template, instance: Any) -> Iterator[Tuple[LiteralType, str]]:
            if isinstance(template, BoolTemplate):
                assert isinstance(instance, int)
                yield instance, ''
            elif isinstance(template, NumberTemplate) or isinstance(template, OneHotTemplate) or isinstance(template, ManyHotTemplate):
                yield from visit(ArrayTemplate(BoolTemplate(), (template.size,)), instance)
            elif isinstance(template, ArrayTemplate):
                assert isinstance(instance, list)
                as_np = np.empty(template.shape, dtype=object)
                as_np[:] = instance
                for idx, elem in np.ndenumerate(as_np):
                    if len(idx) == 1:
                        idx, = idx
                    for lit, path in visit(template.component, elem):
                        yield lit, f'[{idx}]{path}'
            elif isinstance(template, CompositeTemplate):
                instance = instance._asdict()
                for key, sub_template in template.atomics.items():
                    for lit, path in visit(sub_template, instance[key]):
                        yield lit, f'.{key}{path}'
            else:
                assert False

        self.label_map: Dict[LiteralType, str] = {}
        for y in range(grid.height):
            for x in range(grid.width):
                instance = grid.get_tile_instance(x, y)
                for lit, path in visit(self.grid.template, instance):
                    self.label_map[lit] = LiteralLabel(False, (x, y), path)
                    self.label_map[-lit] = LiteralLabel(True, (x, y), path)

    def assume(self, lits: Set[LiteralType]):
        all_tiles = get_all_tiles()
        allowed = np.full((self.grid.height, self.grid.width), None)
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                allowed[y, x] = all_tiles.copy()

        for lit in lits:
            for remaining, x, y, tile in self.literal_map[lit]:
                if any(term not in lits for term in remaining):
                    continue
                allowed[y, x].discard(tile)

        return allowed

    def label(self, lit: LiteralType) -> Optional[LiteralLabel]:
        return self.label_map.get(lit)


def complete_16_16_problem(width: int):
    grid = belt_balancer_net_free_power_of_2.create_balancer(width, 16, 8)
    grid.block_belts_through_edges((False, True))
    grid.prevent_intersection()
    grid.enforce_maximum_underground_length()

    optimisations.expand_underground(grid, min_x=1, max_x=grid.width - 2)
    optimisations.apply_generic_optimisations(grid)
    optimisations.prevent_double_splitters(grid)

    return grid


def partial_16_16_problem(provided_columns: int):
    full_solution = stringifier.decode('''
┌───────────────────────────┐
│→ → → f G l   G → D → f L →│
│→ D l ↓ I G f I L d l ↓ L →│
│→ d l T D h k L f G l k L →│
│→ → → → d D l   ↓ ↑ G f L →│
│→ → → → D d l i ↓ W w ↓ L →│
│→ D l G d f L h ↓ I I T → →│
│→ d l ↑   T → f ↓ L D → → →│
│→ → → h i G l ↓ T → d f L →│
│→ → → f W w K ↓       T → →│
│→ D l ↓ I ↑ s S   i   L D →│
│→ d l T D h k ↓ L h G → d →│
│→ → → → d → f T l   ↑ K L →│
│→ → → → D l ↓     L h T → →│
│→ D l G d f T → f   i L D →│
│→ d l ↑ i ↓ K   ↓ L h G d →│
│→ → → h F g T l T → → h L →│
└───────────────────────────┘'''.strip().splitlines())

    grid = belt_balancer_net_free_power_of_2.create_balancer(full_solution.shape[1], full_solution.shape[0], 8)
    grid.block_belts_through_edges((False, True))
    grid.prevent_intersection()
    grid.enforce_maximum_underground_length()

    for (y, x), tile in np.ndenumerate(full_solution[:, :provided_columns]):
        grid.set_tile(x, y, tile)

    optimisations.expand_underground(grid, min_x=1, max_x=grid.width - 2)
    optimisations.apply_generic_optimisations(grid)
    optimisations.prevent_double_splitters(grid)

    return grid


def network_problem(network_path: str, width: int, height: int, underground_length: int):
    network = open_network(network_path)
    network = deduplicate_network(network)
    grid = belt_balancer.create_balancer(network, width, height, underground_length)

    grid.prevent_intersection()
    grid.block_belts_through_edges((False, True))
    grid.enforce_maximum_underground_length()

    optimisations.expand_underground(grid, min_x=1, max_x=grid.width - 2)
    optimisations.apply_generic_optimisations(grid)
    belt_balancer.prevent_double_edge_belts(grid)
    belt_balancer.setup_balancer_ends(grid, network, True, False)

    return grid


def interchange_problem(width: int, height: int, underground_length: int):
    grid = interchange.create_grid(width, height, underground_length, True)

    optimisations.apply_generic_optimisations(grid)
    interchange.prevent_passing(grid)
    interchange.prevent_awkward_underground_entry(grid)
    interchange.require_correct_transport_through_edges(grid)

    return grid


def read_number_if_defined(assignment: Set[int], number_lits: List[int]):
    number = 0
    for i, lit in enumerate(number_lits):
        if lit in assignment:
            number |= 1 << i
        elif -lit not in assignment:
            return None
    return number


def get_val(assignment: Set[int], lit: int) -> Optional[bool]:
    if lit in assignment:
        return True
    if -lit in assignment:
        return False
    return None


def get_prev_tile(tiles: np.ndarray, coord: Coord[int]) -> Optional[Coord[int]]:
    grid_tile: tile_type.BeltConnectedTile = tiles[coord.y, coord.x]

    if grid_tile.input_direction is not None:
        coord = coord - Coord(*grid_tile.input_direction.vec)
        if coord.x < 0 or coord.y < 0 or coord.x >= tiles.shape[1] or coord.y >= tiles.shape[0]:
            return None
        return coord

    if not isinstance(grid_tile, tile_type.UndergroundBelt):
        return None

    direction = Coord(*grid_tile.direction.vec)
    opposite_end = tile_type.UndergroundBelt(direction, True)

    # TODO check underground state
    while True:
        coord = coord - direction

        if coord.x < 0 or coord.y < 0 or coord.x >= tiles.shape[1] or coord.y >= tiles.shape[0]:
            break

        if tiles[coord.y, coord.x] == opposite_end:
            return coord

    return None


class PathLocation(NamedTuple):
    coord: Coord[int]
    underground_dir: Optional[Direction]

    @property
    def x(self):
        return self.coord.x

    @property
    def y(self):
        return self.coord.y


@dataclass(frozen=True)
class PathTracer:
    tiles: np.ndarray
    underground: np.ndarray

    def __post_init__(self):
        assert self.tiles.dtype == object
        assert self.underground.dtype == bool
        assert self.underground.shape == (*self.tiles.shape, 4)

    def in_bounds(self, coord: Coord[int]) -> bool:
        return coord.x >= 0 and coord.y >= 0 and coord.x < self.tiles.shape[1] and coord.y < self.tiles.shape[0]

    def iterate_coords(self) -> Iterator[Coord[int]]:
        for y in range(self.tiles.shape[0]):
            for x in range(self.tiles.shape[1]):
                yield Coord(x, y)

    def tile_at(self, coord: Coord[int]) -> Optional[tile_type.BaseTile]:
        return self.tiles[coord.y, coord.x]

    def underground_at(self, coord: Coord[int], direction: Direction) -> bool:
        return self.underground[coord.y, coord.x, direction]

    def trace_forward(self, _from: PathLocation, skip_splitters: bool = False) -> Iterator[PathLocation]:
        while _from is not None:
            yield _from
            _from = self.next(_from)

    def trace_backward(self, _from: PathLocation, skip_splitters: bool = False) -> Iterator[PathLocation]:
        while _from is not None:
            yield _from
            _from = self.prev(_from)

    def next(self, loc: PathLocation) -> Optional[PathLocation]:
        if loc.underground_dir is not None:
            present = self.underground_at(loc.coord, loc.underground_dir)
            if not present:
                return None
            next_coord = loc.coord + Coord(*loc.underground_dir.vec)

            if not self.in_bounds(next_coord):
                return None

            tile = self.tile_at(next_coord)
            if isinstance(tile, tile_type.UndergroundBelt) and not tile.is_input and tile.direction == loc.underground_dir:
                return PathLocation(next_coord, None)

            return PathLocation(next_coord, loc.underground_dir)

        tile = self.tile_at(loc.coord)
        if not isinstance(tile, tile_type.BeltConnectedTile):
            return None

        if tile.output_direction is not None:
            next_coord = loc.coord + Coord(*tile.output_direction.vec)
            if not self.in_bounds(next_coord):
                return None

            return PathLocation(next_coord, None)

        if isinstance(tile, tile_type.UndergroundBelt):
            assert tile.is_input

            next_coord = loc.coord + Coord(*tile.direction.vec)
            if not self.in_bounds(next_coord):
                return None

            # print('underground', next_coord, tile.direction, file=sys.stderr)
            return PathLocation(next_coord, tile.direction)

        return None

    def prev(self, loc: PathLocation) -> Optional[PathLocation]:
        if loc.underground_dir is not None:
            present = self.underground_at(loc.coord, loc.underground_dir)
            if not present:
                return None
            prev_coord = loc.coord - Coord(*loc.underground_dir.vec)

            if not self.in_bounds(prev_coord):
                return None

            tile = self.tile_at(prev_coord)
            if isinstance(tile, tile_type.UndergroundBelt) and tile.is_input and tile.direction == loc.underground_dir:
                return PathLocation(prev_coord, None)

            return PathLocation(prev_coord, loc.underground_dir)

        tile = self.tile_at(loc.coord)
        if not isinstance(tile, tile_type.BeltConnectedTile):
            return None

        if tile.input_direction is not None:
            prev_coord = loc.coord - Coord(*tile.input_direction.vec)
            if not self.in_bounds(prev_coord):
                return None

            return PathLocation(prev_coord, None)

        if isinstance(tile, tile_type.UndergroundBelt):
            assert not tile.is_input

            prev_coord = loc.coord - Coord(*tile.direction.vec)
            if not self.in_bounds(prev_coord):
                return None

            return PathLocation(prev_coord, tile.direction)

        return None

    def conflict_path(self, grid: Grid, path: Iterator[PathLocation]) -> ClauseType:
        clause = []
        for loc in path:
            tile_template = grid.get_tile_instance(loc.x, loc.y)
            if loc.underground_dir is None:
                tile = self.tile_at(loc.coord)

                if isinstance(tile, tile_type.UndergroundBelt):
                    if tile.is_input:
                        clause += [
                            -tile_template.is_underground_in,
                            -tile_template.input_direction[tile.direction]
                        ]
                    else:
                        clause += [
                            -tile_template.is_underground_out,
                            -tile_template.output_direction[tile.direction]
                        ]
                elif tile == tile_type.EmptyTile():
                    clause.append(-tile_template.is_empty)
                else:
                    if not isinstance(tile, tile_type.BeltConnectedTile):
                        print('wut how are we here...', loc, tile, path, file=sys.stderr)
                        print(stringifier.encode(self.tiles), file=sys.stderr)
                        exit(1)
                        return clause
                    clause += [
                        -tile_template.input_direction[tile.input_direction],
                        -tile_template.output_direction[tile.output_direction],
                    ]

                    if isinstance(tile, tile_type.Belt):
                        clause.append(-tile_template.is_belt)
                    elif isinstance(tile, tile_type.Splitter):
                        clause.append(-tile_template.is_splitter)
                        if tile.is_head:
                            clause.append(-tile_template.is_splitter_head)
                        else:
                            clause.append(tile_template.is_splitter_head)
            else:
                clause.append(-tile_template.underground[loc.underground_dir])
        return clause

    def find_loops(self) -> Iterator[list[PathLocation]]:
        visited = np.full_like(self.tiles, False)

        for start_coord in self.iterate_coords():
            if visited[start_coord.y, start_coord.x]:
                continue

            # print(len(list(tracer.trace_forward(PathLocation(coord, None)))), file=sys.stderr)
            visited[start_coord.y, start_coord.x] = True
            initial = PathLocation(start_coord, None)

            path = [initial]
            for loc in self.trace_forward(self.next(initial)):
                if loc == initial:
                    yield path
                    break

                if loc.underground_dir is None:
                    if visited[loc.y, loc.x]:
                        # Retracing segment of existing path
                        break
                    visited[loc.y, loc.x] = True

                path.append(loc)

    def get_loop(self, initial: PathLocation) -> Optional[list[PathLocation]]:
        path = [initial]
        for loc in self.trace_forward(self.next(initial)):
            if loc == initial:
                return path

            path.append(loc)

    def on_path(self, location: PathLocation) -> bool:
        if location.underground_dir is not None:
            return self.underground_at(location.coord, location.underground_dir)
        else:
            return isinstance(self.tile_at(location.coord), tile_type.BeltConnectedTile)

    def without_invalid(self) -> 'PathTracer':
        new_tracer = PathTracer(self.tiles.copy(), self.underground.copy())

        for coord in self.iterate_coords():
            loc = PathLocation(coord, None)
            next_tile = self.next(loc)
            if next_tile is None:
                continue
            if self.tile_at(next_tile) is None:
                continue
            prev_tile = self.prev(next_tile)
            if prev_tile != loc:
                new_tracer.tiles[coord.y, coord.x] = None

        return new_tracer

    def trace_all_paths(self) -> Iterator[list[PathLocation]]:
        # Loops must be eliminated before this
        visited = np.full_like(self.tiles, False)
        # print('trace', file=sys.stderr)
        for coord in self.iterate_coords():
            if self.tiles[coord.y, coord.x] is None:
                continue

            if visited[coord.y, coord.x]:
                continue

            loc = PathLocation(coord, None)
            while True:
                prev = self.prev(loc)
                if prev is None:
                    break
                if self.next(prev) != loc:
                    break

                loc = prev

            path = []
            while True:
                path.append(loc)
                if loc.underground_dir is None:
                    assert not visited[loc.y, loc.x]
                    visited[loc.y, loc.x] = True

                next_loc = self.next(loc)
                if next_loc is None:
                    break
                if self.prev(next_loc) != loc:
                    break

                loc = next_loc

            if len(path) > 1:
                yield path


def determine_tile(assignment: Set[int], tile: TileTemplate) -> Optional[tile_type.BaseTile]:
    input_direction = None
    output_direction = None
    for direction in Direction:
        if tile.input_direction[direction] in assignment:
            if input_direction is not None:
                return None
            input_direction = direction

        if tile.output_direction[direction] in assignment:
            if output_direction is not None:
                return None
            output_direction = direction

    if input_direction is None and any(-tile.input_direction[direction] not in assignment for direction in Direction):
        return None
    if output_direction is None and any(-tile.output_direction[direction] not in assignment for direction in Direction):
        return None

    if tile.is_empty in assignment:
        if input_direction is not None or output_direction is not None:
            return None
        return tile_type.EmptyTile()

    if tile.is_belt in assignment:
        if input_direction is None or output_direction is None or input_direction.reverse == output_direction:
            return None
        return tile_type.Belt(input_direction, output_direction)

    if tile.is_underground_in in assignment:
        if input_direction is None or output_direction is not None:
            return None
        return tile_type.UndergroundBelt(input_direction, True)

    if tile.is_underground_out in assignment:
        if input_direction is not None or output_direction is None:
            return None
        return tile_type.UndergroundBelt(output_direction, False)

    if tile.is_splitter in assignment:
        is_head = get_val(assignment, tile.is_splitter_head)
        if is_head is None:
            return None

        direction = input_direction
        if direction is None:
            direction = output_direction
            if direction is None:
                return None
        return tile_type.Splitter(direction, is_head)

    return None


def find_shortcuts(path_tracer: PathTracer, path: list[PathLocation]) -> Iterator[list[PathLocation]]:
    distance_map = np.full(path_tracer.tiles.shape, -1, dtype=int)
    underground_dist_map = np.full(path_tracer.underground.shape, -1, dtype=int)
    for distance, loc in enumerate(path):
        if loc.underground_dir is None:
            distance_map[loc.y, loc.x] = distance
        else:
            underground_dist_map[loc.y, loc.x, loc.underground_dir] = distance

    for distance, loc in enumerate(path):
        if loc.underground_dir is not None:
            continue
        base_require_empty: list[Coord] = []
        tile = path_tracer.tile_at(loc.coord)
        if isinstance(tile, tile_type.Belt):
            # direction = tile.input_direction
            alt_connection_directions = [tile.input_direction, tile.input_direction.next, tile.input_direction.prev]
            alt_connection_directions.remove(tile.output_direction)
            min_cut_distance = distance
        elif isinstance(tile, tile_type.UndergroundBelt):
            alt_connection_directions = [tile.direction.next, tile.direction.prev]
            if not tile.is_input:
                previous_loc = loc.coord - Coord(*tile.direction.vec)
                if not path_tracer.in_bounds(previous_loc):
                    continue

                previous_dist = distance_map[previous_loc.y, previous_loc.x]
                if previous_dist != -1 and previous_dist > distance:
                    min_cut_distance = previous_dist
                elif path_tracer.tile_at(previous_loc) == tile_type.EmptyTile():
                    min_cut_distance = distance
                    base_require_empty.append(previous_loc)
                else:
                    continue
            else:
                alt_connection_directions.append(tile.direction)
                min_cut_distance = distance
        else:
            continue

        for transverse in alt_connection_directions:
            trans_loc = loc.coord + Coord(*transverse.vec)
            if not path_tracer.in_bounds(trans_loc):
                continue

            require_empty = base_require_empty.copy()
            next_tile = path_tracer.tile_at(trans_loc)

            shortcut_len = 1
            while next_tile == tile_type.EmptyTile():

                require_empty.append(trans_loc)
                shortcut_len += 1
                trans_loc = trans_loc + Coord(*transverse.vec)
                if path_tracer.in_bounds(trans_loc):
                    next_tile = path_tracer.tile_at(trans_loc)
                else:
                    next_tile = None

            if next_tile is None:
                continue

            trans_distance = distance_map[trans_loc.y, trans_loc.x]

            if isinstance(next_tile, tile_type.UndergroundBelt) and next_tile.is_input:
                underground_next = trans_loc + Coord(*next_tile.direction.vec)
                if not path_tracer.in_bounds(underground_next):
                    continue
                if path_tracer.tile_at(underground_next) == tile_type.EmptyTile():
                    require_empty.append(underground_next)
                else:
                    underground_next_dist = distance_map[underground_next.y, underground_next.x]
                    if underground_next_dist == -1 or not (min_cut_distance < underground_next_dist < trans_distance):
                        continue

            if trans_distance != -1 and trans_distance > min_cut_distance + shortcut_len:
                assert trans_distance < len(path)
                conflict = path[distance:trans_distance + 1]

                for coord in require_empty:
                    conflict.append(PathLocation(coord, None))

                yield conflict

    for distance, loc in enumerate(path):
        tile = path_tracer.tile_at(loc.coord)

        if not isinstance(tile, tile_type.BeltConnectedTile):
            continue

        if loc.underground_dir is not None:
            if tile.output_direction != loc.underground_dir:
                continue

            tile_distance = distance_map[loc.y, loc.x]
            if tile_distance != -1 and tile_distance > distance:
                yield path[distance:tile_distance + 1]

        else:
            if tile.input_direction is None:
                continue

            underground_distance = underground_dist_map[loc.y, loc.x, tile.input_direction]
            if underground_distance != -1 and underground_distance > distance:
                yield path[distance:underground_distance + 1]


def main():
    # grid = complete_16_16_problem(13) # Goal
    # grid = partial_16_16_problem(0)
    # grid = partial_16_16_problem(4)
    # grid = partial_16_16_problem(6)
    # grid = network_problem('networks/4x4', 10, 4, 4)
    # grid = network_problem('networks/6x6', 13, 6, 8)
    # grid = interchange_problem(9, 24, 8)
    # grid = interchange_problem(16, 24, 4)
    # grid = interchange_problem(10, 24, 4)
    # grid = interchange_problem(9, 24, 4) # UNSAT
    # grid = interchange_problem(10, 32, 8) # UNSAT
    grid = interchange_problem(11, 32, 8)
    # grid = interchange_problem(14, 64, 8)
    # interchange.require_rotational_symmetry(grid)

    # solver = Solver()
    solver = IPASIRLibrary('./libcadical.so').create_solver()
    for clause in grid.clauses:
        solver.add_clause(clause)

    elim_map = EliminationMap(grid)

    all_learned = set()

    total = 0

    def print_trail(lits: List[LiteralType]):
        nonlocal total, trailing_time
        print_start = time.time()
        total += 1
        lits = set(lits)
        # allowed = elim_map.assume(lits)

        old_learned = learned.copy()
        learned.clear()
        learned_locations = defaultdict(list)

        for clause in old_learned:
            for lit in clause:
                if isinstance(lit, LiteralLabel):
                    learned_locations[lit.location].append(lit)

        tiles = np.full((grid.height, grid.width), None)

        tile_grid = np.full((grid.height, grid.width), None)
        underground_grid = np.full((grid.height, grid.width, 4), False)

        for y in range(grid.height):
            for x in range(grid.width):

                tile_instance = determine_tile(lits, grid.get_tile_instance(x, y))
                tile_grid[y, x] = tile_instance
                # valid: Set[tile_type.BaseTile] = allowed[y, x]

                # tile_instance = None
                # if len(valid) == 1:
                #     tile_instance, = valid
                #     tile_grid[y, x] = tile_instance

                underground_grid[y, x] = [grid.tiles[y, x].underground[direction] in lits for direction in Direction]

                cell = blueprint.write_tile(tile_instance or tile_type.EmptyTile())
                cell['valid'] = 1  # len(valid)
                tiles[y, x] = cell

        path_tracer = PathTracer(tile_grid, underground_grid)
        path_tracer = path_tracer.without_invalid()

        # bar = trace_paths(path_tracer)

        clauses: ClauseList = []

        loop_count = 0
        for loop in PathTracer(path_tracer.tiles.copy(), path_tracer.underground).find_loops():
            clauses.append(path_tracer.conflict_path(grid, loop))
            for loc in loop:
                if loc.underground_dir is None:
                    path_tracer.tiles[loc.y, loc.x] = None

            loop_count += 1

        conflict_areas = []

        path_ids = np.zeros((grid.height, grid.width, 2), dtype=int)
        for path_id, path in enumerate(path_tracer.trace_all_paths()):
            for distance, loc in enumerate(path):
                if loc.underground_dir is None:
                    path_ids[loc.y, loc.x] = [path_id + 1, distance]

            # for sub_path in find_shortcuts(path_tracer, path):
            sub_path = min(find_shortcuts(path_tracer, path), default=None, key=len)
            if sub_path is not None:
                print('conflict path', len(sub_path), total - 2659 - 5069 + 1, file=sys.stderr)
                clause = path_tracer.conflict_path(grid, sub_path)
                print(clause, file=sys.stderr)
                print(' '.join(str(elim_map.label(lit)) for lit in clause), file=sys.stderr)

                # if not all(-lit in lits for lit in clause):
                #     print('Undetermined clause', clause, file=sys.stderr)
                #     assert False

                clauses.append(clause)

                conflict_areas.append((
                    Coord(
                        min(node.x for node in sub_path),
                        min(node.y for node in sub_path)
                    ),
                    Coord(
                        max(node.x for node in sub_path),
                        max(node.y for node in sub_path)
                    )
                ))

        # raise RuntimeError

        # if loop_count == 0:
        #     return
        if loop_count != 0:
            print(total, file=sys.stderr)

        # print(clauses, file=sys.stderr)
        # for clause in sorted(clauses, key=len):
        for clause in clauses:
            solver.add_redundant_clause(clause)
            # break # TODO Take first

        for pos, t in np.ndenumerate(grid.tiles):
            j = tiles[pos]
            j['colour'] = read_number_if_defined(lits, t.colour)
            # j['level'] = read_number_if_defined(lits, t.level)
            # j['level_splitter'] = read_number_if_defined(lits, t.level_splitter)
            # j['level_ux'] = read_number_if_defined(lits, t.level_ux)
            # j['level_uy'] = read_number_if_defined(lits, t.level_uy)

            j['underground'] = underground_grid[pos].tolist()
            j['colour_ux'] = read_number_if_defined(lits, t.colour_ux)
            j['colour_uy'] = read_number_if_defined(lits, t.colour_uy)
            j['is_learned'] = pos[::-1] in learned_locations
            j['learned'] = ' '.join(map(str, learned_locations[pos[::-1]]))

            # j['loop_id'] = loops[pos].tolist() or None

            path_id, path_location = path_ids[pos].tolist()
            if path_id != 0:
                j['path_id'] = path_id
                j['path_location'] = path_location
            else:
                j['path_id'] = None
                j['path_location'] = None

        assert isinstance(grid.template, CompositeTemplate)

        learn_dump = []
        for clause in old_learned:
            learn_dump.append(' '.join(map(str, clause)))

        # for lower, upper in conflict_areas:
        print(json.dumps({
            'metadata': {
                'underground_length': grid.underground_length,
                'lits': len(lits),
                'learned': learn_dump,
            },
            'template': CompositeTemplate({
                'is_learned': BoolTemplate(),
                'valid': NumberTemplate(5),
                'loop_id': NumberTemplate(10),
                'path_id': NumberTemplate(10),
                'path_location': NumberTemplate(10),
                **grid.template.atomics,
            }).write(),
            'tiles': tiles.tolist(),
        }))
        trailing_time += time.time() - print_start

    def add_learned(lits):
        assert len(lits) == 1
        # print(lits)
        all_learned.add(lits[0])
        print_trail(all_learned)

    learned: List[Union[LiteralType, LiteralLabel]] = []

    def display_learned(clause):
        clause = [elim_map.label(lit) or lit for lit in clause]
        # print(len(clause), file=sys.stderr)
        # print(' '.join(clause), file=sys.stderr)
        learned.append(clause)
        # print(total, clause, file=sys.stderr)

    # solver.set_learn(display_learned, 10_000)

    # solver.set_learn(add_learned, 1)
    # solver.set_trail(25, print_trail)
    solver.set_trail(5, print_trail)
    # solver.set_trail(lambda l: print(len(l)))

    trailing_time = 0
    t0 = time.time()
    res = solver.solve()
    t1 = time.time()
    combined_time = t1 - t0
    # print(total)
    print(res, total, trailing_time, combined_time - trailing_time, combined_time, file=sys.stderr)
    # print(solver.accum_stats(), file=sys.stderr)
    # if res:
    #     solver.get_model()


if __name__ == '__main__':
    main()
