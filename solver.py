from typing import *

from pysat.formula import IDPool

from cardinality import quadratic_amo
from template import *
from template import FactorioGrid
from tile import BaseTile, Belt, EmptyTile, Splitter, UndergroundBelt
from util import *


class TileTemplate(Protocol):
    input_direction: List[LiteralType]
    output_direction: List[LiteralType]
    all_direction: List[LiteralType]
    is_splitter: LiteralType
    is_splitter_head: LiteralType
    underground: List[LiteralType]
    colour: List[LiteralType]
    colour_ux: List[LiteralType]
    colour_uy: List[LiteralType]


class Grid(FactorioGrid[TileTemplate, Dict[str, Any]]):
    def __init__(self, width: int, height: int, colours: Optional[int], underground_length: int = 4, extras: CompositeTemplateParams = {}, pool: Optional[IDPool] = None):
        assert colours is None or colours >= 1
        assert underground_length >= 0
        self.colours = colours
        self.underground_length = underground_length

        template = {
            'input_direction': OneHotTemplate(4),
            'output_direction': OneHotTemplate(4),
            'all_direction': lambda input_direction, output_direction: [*input_direction, *output_direction],
            'is_splitter': BoolTemplate(),
            'is_splitter_head': BoolTemplate(),
            'underground': ArrayTemplate(BoolTemplate(), (4,)),
        }
        if colours is not None:
            self.colour_bits = bin_length(colours)
            template.update({
                'colour': NumberTemplate(self.colour_bits),
                'colour_ux': NumberTemplate(self.colour_bits),
                'colour_uy': NumberTemplate(self.colour_bits),
            })
        else:
            self.colour_bits = None

        template.update(extras)
        template = CompositeTemplate(template)

        super().__init__(template, width, height, pool)

        for tile in self.iterate_tiles():
            self.clauses += quadratic_amo(tile.input_direction)  # Have an input direction or nothing
            self.clauses += quadratic_amo(tile.output_direction)  # Have an output direction or nothing

            self.clauses += quadratic_amo(tile.underground[0::2])  # Have a underground along -x, +x or nothing
            self.clauses += quadratic_amo(tile.underground[1::2])  # Have a underground along -y, +y or nothing

            # If a tile is a splitter, then it is not empty
            self.clauses += implies([tile.is_splitter], [tile.all_direction])
            # If a tile is a splitter head, then it is a splitter
            self.clauses.append([-tile.is_splitter_head, tile.is_splitter])

            # Splitters must output the same side as their input or have no output
            for direction in range(4):
                output = tile.output_direction.copy()
                del output[direction]

                self.clauses += implies([tile.is_splitter, tile.input_direction[direction]], set_all_false(output))

                input = tile.input_direction.copy()
                del input[direction]
                self.clauses += implies([tile.is_splitter, tile.output_direction[direction]], set_all_false(input))

            for direction in range(4):
                # Cannot input from same side as output
                self.clauses += quadratic_amo([tile.input_direction[direction], tile.output_direction[(direction + 2) % 4]])

                # Cannot have a turn and be a splitter
                self.clauses += implies([tile.is_splitter, tile.input_direction[direction]], [[-tile.output_direction[(direction + 1) % 4]]])

            # Prevent colours beyond end of range
            if self.colours is not None:
                for colour_range in (tile.colour, tile.colour_ux, tile.colour_uy):
                    self.clauses += set_maximum(self.colours - 1, colour_range)

        for direction in range(4):
            inv_direction = (direction + 2) % 4
            for tile_a, tile_b in self.iterate_tile_lines(direction_to_vec((direction + 1) % 4), 2, EdgeMode.NO_WRAP):
                if tile_b is None:  # Prevent splitter overlapping edge of grid
                    self.clauses += [
                        [-tile_a.input_direction[direction],      -tile_a.is_splitter, -tile_a.is_splitter_head],
                        [-tile_a.output_direction[direction],     -tile_a.is_splitter, -tile_a.is_splitter_head],
                        [-tile_a.input_direction[inv_direction],  -tile_a.is_splitter,  tile_a.is_splitter_head],
                        [-tile_a.output_direction[inv_direction], -tile_a.is_splitter,  tile_a.is_splitter_head],
                    ]
                    continue
                for side in (tile_a.input_direction, tile_a.output_direction):
                    # Complementary side exists
                    self.clauses += implies([side[direction], tile_a.is_splitter_head], [[tile_b.is_splitter], [-tile_b.is_splitter_head]])
                    self.clauses += implies([side[inv_direction], tile_a.is_splitter, -tile_a.is_splitter_head], [[tile_b.is_splitter_head]])

                    # Complementary side has input/output direction correct
                    self.clauses += implies([side[direction], tile_a.is_splitter_head],
                                            [[tile_b.input_direction[direction], tile_b.output_direction[direction]]])
                    self.clauses += implies([side[inv_direction], tile_a.is_splitter, -tile_a.is_splitter_head],
                                            [[tile_b.input_direction[inv_direction], tile_b.output_direction[inv_direction]]])

    def set_tile(self, x: int, y: int, tile: BaseTile):
        tile_instance = self.get_tile_instance(x, y)

        if isinstance(tile, EmptyTile):
            self.clauses += set_all_false(tile_instance.all_direction)
        elif isinstance(tile, Splitter):
            self.clauses.append([tile_instance.is_splitter])
            self.clauses.append([set_literal(tile_instance.is_splitter_head, tile.is_head)])
            self.clauses.append([tile_instance.input_direction[tile.direction], tile_instance.output_direction[tile.direction]])
        elif isinstance(tile, Belt) or isinstance(tile, UndergroundBelt):
            self.clauses += [[-tile_instance.is_splitter]]
            if tile.input_direction is None:
                self.clauses += set_all_false(tile_instance.input_direction)
            else:
                self.clauses += [[tile_instance.input_direction[tile.input_direction]]]

            if tile.output_direction is None:
                self.clauses += set_all_false(tile_instance.output_direction)
            else:
                self.clauses += [[tile_instance.output_direction[tile.output_direction]]]
        else:
            raise RuntimeError(f'Unsupported tile type {tile}')

    def read_tile(self, cell: Dict[str, Any]) -> BaseTile:
        input_direction = cell['input_direction']
        output_direction = cell['output_direction']
        if cell['is_splitter']:
            direction = input_direction
            if direction is None:
                direction = output_direction
                assert direction is not None
            return Splitter(direction, cell['is_splitter_head'])
        elif input_direction is None and output_direction is None:
            return EmptyTile()
        elif input_direction is None or output_direction is None:
            direction = input_direction
            if direction is None:
                direction = output_direction
                assert direction is not None
            return UndergroundBelt(direction, output_direction is None)
        else:
            return Belt(input_direction, output_direction)

    def prevent_colour(self, colour: int):
        for x in range(self.width):
            for y in range(self.height):
                tile = self.get_tile_instance(x, y)
                if colour == 0:
                    self.clauses += [[-lit, *set_not_number(0, tile.colour)] for lit in tile.all_direction]
                else:
                    self.clauses += [set_not_number(colour, colour_range) for colour_range in (tile.colour, tile.colour_ux, tile.colour_uy)]

    def set_colour(self, x: int, y: int, colour: int):
        assert 0 <= colour < self.colours
        tile = self.get_tile_instance(x, y)
        self.clauses += set_number(colour, tile.colour)

    def transport_quantity(self, quantity: Callable[[TileTemplate], NestedArray[LiteralType]], quantity_ux: Callable[[TileTemplate], NestedArray[LiteralType]], quantity_uy: Callable[[TileTemplate], NestedArray[LiteralType]], edge_mode: EdgeModeType):
        for direction in range(4):
            dx, dy = direction_to_vec(direction)
            for x in range(self.width):
                for y in range(self.height):
                    tile_a = self.get_tile_instance(x, y)
                    tile_b = self.get_tile_instance_offset(x, y, dx, dy, edge_mode)

                    if tile_b is None:
                        continue

                    quantity_a = flatten(quantity(tile_a))
                    quantity_b = flatten(quantity(tile_b))

                    if direction % 2 == 0:
                        quantity_ua = flatten(quantity_ux(tile_a))
                        quantity_ub = flatten(quantity_ux(tile_b))
                    else:
                        quantity_ua = flatten(quantity_uy(tile_a))
                        quantity_ub = flatten(quantity_uy(tile_b))

                    # Belt quantity consistent
                    self.clauses += implies([tile_a.output_direction[direction], -tile_a.is_splitter], set_numbers_equal(quantity_a, quantity_b))

                    # Underground quantity consistent
                    self.clauses += implies([tile_a.underground[direction]], set_numbers_equal(quantity_ua, quantity_ub))

                    # Underground transition consistent
                    self.clauses += implies([tile_a.input_direction[direction], -tile_a.is_splitter, *
                                            invert_components(tile_a.output_direction)], set_numbers_equal(quantity_a, quantity_ub))
                    self.clauses += implies([tile_b.output_direction[direction], -tile_b.is_splitter, *
                                            invert_components(tile_b.input_direction)], set_numbers_equal(quantity_ua, quantity_b))

    def prevent_bad_colouring(self, edge_mode: EdgeModeType):
        if self.colours == 1:
            return
        self.transport_quantity(lambda tile: tile.colour, lambda tile: tile.colour_ux, lambda tile: tile.colour_uy, edge_mode)

    def block_underground_through_edges(self, edges: Union[bool, Tuple[bool, bool], Tuple[bool, bool, bool, bool]] = True):
        if isinstance(edges, bool):
            edges = edges, edges

        if len(edges) == 2:
            horisontal, vertical = edges
            edges = horisontal, vertical, horisontal, vertical

        min_x_blocked, min_y_blocked, max_x_blocked, max_y_blocked = edges

        if min_x_blocked:
            for y in range(self.height):
                tile = self.get_tile_instance(0, y)
                self.clauses += set_all_false(tile.underground[0::2])
        if max_x_blocked:
            for y in range(self.height):
                tile = self.get_tile_instance(self.width - 1, y)
                self.clauses += set_all_false(tile.underground[0::2])
        if min_y_blocked:
            for x in range(self.width):
                tile = self.get_tile_instance(x, 0)
                self.clauses += set_all_false(tile.underground[1::2])
        if max_y_blocked:
            for x in range(self.width):
                tile = self.get_tile_instance(x, self.height - 1)
                self.clauses += set_all_false(tile.underground[1::2])

    def block_belts_through_edges(self, edges: Union[bool, Tuple[bool, bool], Tuple[bool, bool, bool, bool]] = True):
        if isinstance(edges, bool):
            edges = edges, edges

        if len(edges) == 2:
            horisontal, vertical = edges
            edges = horisontal, vertical, horisontal, vertical

        min_x_blocked, min_y_blocked, max_x_blocked, max_y_blocked = edges
        if min_x_blocked:
            for y in range(self.height):
                tile = self.get_tile_instance(0, y)
                self.clauses += set_all_false([tile.input_direction[0], tile.output_direction[2]])
                self.clauses.append([-tile.input_direction[2], *tile.output_direction])
                self.clauses.append([-tile.output_direction[0], *tile.input_direction])
        if max_x_blocked:
            for y in range(self.height):
                tile = self.get_tile_instance(self.width - 1, y)
                self.clauses += set_all_false([tile.input_direction[2], tile.output_direction[0]])
                self.clauses.append([-tile.input_direction[0], *tile.output_direction])
                self.clauses.append([-tile.output_direction[2], *tile.input_direction])
        if min_y_blocked:
            for x in range(self.width):
                tile = self.get_tile_instance(x, 0)
                self.clauses += set_all_false([tile.input_direction[3], tile.output_direction[1]])
                self.clauses.append([-tile.input_direction[1], *tile.output_direction])
                self.clauses.append([-tile.output_direction[3], *tile.input_direction])
        if max_y_blocked:
            for x in range(self.width):
                tile = self.get_tile_instance(x, self.height - 1)
                self.clauses += set_all_false([tile.input_direction[1], tile.output_direction[3]])
                self.clauses.append([-tile.input_direction[3], *tile.output_direction])
                self.clauses.append([-tile.output_direction[1], *tile.input_direction])

    def prevent_bad_undergrounding(self, edge_mode: EdgeModeType):
        for direction in range(4):
            reverse_dir = (direction + 2) % 4

            dx, dy = direction_to_vec(direction)
            for x in range(self.width):
                for y in range(self.height):
                    tile_a = self.get_tile_instance(x, y)

                    # Underground entrance/exit cannot be above underground segment with same direction
                    self.clauses += implies(
                        [tile_a.input_direction[direction], -tile_a.is_splitter, *invert_components(tile_a.output_direction)],
                        [[-tile_a.underground[direction]], [-tile_a.underground[reverse_dir]]]
                    )
                    self.clauses += implies(
                        [tile_a.output_direction[direction], -tile_a.is_splitter, *invert_components(tile_a.input_direction)],
                        [[-tile_a.underground[direction]], [-tile_a.underground[reverse_dir]]]
                    )

                    # Underground entrance/exit must have a underground segment after/before it
                    clause = [
                        -tile_a.input_direction[direction],
                        *tile_a.output_direction,
                        tile_a.is_splitter,
                    ]

                    tile_b = self.get_tile_instance_offset(x, y, +dx, +dy, edge_mode)
                    if tile_b is not None:
                        clause.append(tile_b.underground[direction])
                        self.clauses.append(clause)

                    clause = [
                        -tile_a.output_direction[direction],
                        *tile_a.input_direction,
                        tile_a.is_splitter,
                    ]
                    tile_b = self.get_tile_instance_offset(x, y, -dx, -dy, edge_mode)
                    if tile_b is not None:
                        clause.append(tile_b.underground[direction])
                        self.clauses.append(clause)

                    # Underground segment must propagate or have output
                    tile_b = self.get_tile_instance_offset(x, y, +dx, +dy, edge_mode)
                    if tile_b is not None:
                        self.clauses += implies(
                            [tile_a.underground[direction], -tile_b.underground[direction]],
                            [
                                [tile_b.output_direction[direction]],
                                *([-tile_b.input_direction[i]] for i in range(4)),
                                [-tile_b.is_splitter],
                            ]
                        )

                    tile_b = self.get_tile_instance_offset(x, y, -dx, -dy, edge_mode)
                    if tile_b is not None:
                        self.clauses += implies(
                            [tile_a.underground[direction], -tile_b.underground[direction]],
                            [
                                [tile_b.input_direction[direction]],
                                *([-tile_b.output_direction[i]] for i in range(4)),
                                [-tile_b.is_splitter],
                            ]
                        )

    def enforce_maximum_underground_length(self, edge_mode: EdgeModeType):
        assert self.underground_length >= 1

        if self.underground_length == float('inf'):
            return

        for direction in range(4):
            dx, dy = direction_to_vec(direction)
            for x in range(self.width):
                for y in range(self.height):
                    clause = []
                    for i in range(self.underground_length + 1):
                        tile = self.get_tile_instance_offset(x, y, dx * i, dy * i, edge_mode)

                        if tile is None:
                            break

                        clause.append(-tile.underground[direction])
                    else:
                        self.clauses.append(clause)

    def prevent_intersection(self, edge_mode: EdgeModeType):
        for direction in range(4):
            for tile_a, tile_b in self.iterate_tile_lines(direction_to_vec(direction), 2, edge_mode):
                if tile_b is None:
                    continue

                self.clauses += literals_same(tile_a.output_direction[direction], tile_b.input_direction[direction])

                # Handles special splitter output case
                self.clauses += implies([tile_a.input_direction[direction], tile_a.is_splitter, -tile_b.is_splitter], [
                    [-tile_b.input_direction[(direction + 1) % 4], -tile_b.output_direction[(direction + 1) % 4]],
                    [-tile_b.input_direction[(direction - 1) % 4], -tile_b.output_direction[(direction - 1) % 4]],

                    [-tile_b.input_direction[(direction + 1) % 4], -tile_b.output_direction[direction]],
                    [-tile_b.input_direction[(direction - 1) % 4], -tile_b.output_direction[direction]],

                    [-tile_b.input_direction[(direction + 2) % 4], -tile_b.output_direction[(direction + 1) % 4]],
                    [-tile_b.input_direction[(direction + 2) % 4], -tile_b.output_direction[(direction - 1) % 4]],
                ])

    def itersolve(self, important_variables=set(), solver='g3', ignore_colour=False):
        important_variables = set(important_variables)
        for x in range(self.width):
            for y in range(self.height):
                tile = self.get_tile_instance(x, y)

                important_variables |= set([*tile.all_direction, tile.is_splitter])

                if not ignore_colour:
                    important_variables |= set(tile.colour + tile.colour_ux + tile.colour_uy)
        return super().itersolve(important_variables, solver)
