from typing import Callable, List, Dict, Any, Optional, Protocol, Tuple, Union

from pysat.formula import IDPool

from .cardinality import quadratic_amo, quadratic_one
from .direction import Axis, Direction
from .template import (ArrayTemplate, BoolTemplate, CompositeTemplate, CompositeTemplateParams, EdgeMode,
                       EdgeModeType, FactorioGrid, NestedArray, NumberTemplate, OneHotTemplate, flatten)
from .tile import BaseTile, Belt, EmptyTile, FillerTile, Splitter, UndergroundBelt
from .util import LiteralType, implies, invert_components, literals_same, set_all_false, set_literal, set_maximum, set_not_number, set_number, set_numbers_equal


class TileTemplate(Protocol):
    type: List[LiteralType]
    is_empty: LiteralType
    is_belt: LiteralType
    is_underground_in: LiteralType
    is_underground_out: LiteralType
    is_splitter: LiteralType
    is_splitter_head: LiteralType
    is_input: LiteralType
    is_output: LiteralType
    input_direction: List[LiteralType]
    output_direction: List[LiteralType]
    all_direction: List[LiteralType]
    underground: List[LiteralType]
    colour: List[LiteralType]
    colour_ux: List[LiteralType]
    colour_uy: List[LiteralType]


class Grid(FactorioGrid[TileTemplate, Dict[str, Any]]):
    def __init__(
            self,
            width: int,
            height: int,
            colours: Optional[int],
            underground_length: int = 4,
            extras: CompositeTemplateParams = {},
            pool: Optional[IDPool] = None,
            edge_mode: EdgeModeType = EdgeMode.NO_WRAP
    ):
        assert colours is None or colours >= 1
        assert underground_length >= 0
        self.colours = colours
        self.underground_length = underground_length

        template = {
            'is_belt': BoolTemplate(),
            'is_empty': BoolTemplate(),
            'is_splitter': BoolTemplate(),
            'is_underground_in': BoolTemplate(),
            'is_underground_out': BoolTemplate(),
            'is_splitter_head': BoolTemplate(),
            'is_input': BoolTemplate(),
            'is_output': BoolTemplate(),
            'input_direction': OneHotTemplate(4),
            'output_direction': OneHotTemplate(4),
            'underground': ArrayTemplate(BoolTemplate(), (4,)),
            'type': lambda is_belt, is_empty, is_splitter, is_underground_in, is_underground_out:
            [
                is_belt,
                is_empty,
                is_splitter,
                is_underground_in,
                is_underground_out,
            ],
            'all_direction': lambda input_direction, output_direction: [*input_direction, *output_direction],
        }
        if colours is not None:
            self.colour_bits = (colours - 1).bit_length()
            template.update({
                'colour': NumberTemplate(self.colour_bits),
                'colour_ux': NumberTemplate(self.colour_bits),
                'colour_uy': NumberTemplate(self.colour_bits),
            })
        else:
            self.colour_bits = None

        template.update(extras)
        template = CompositeTemplate(template)

        super().__init__(template, width, height, pool=pool, edge_mode=edge_mode)

        for tile in self.iterate_tiles():
            # Each tile has exactly one type
            self.clauses += quadratic_one(tile.type)

            # Empty tiles must not have any inputs/outputs
            self.clauses += implies([tile.is_empty], set_all_false(tile.all_direction))
            # Belts must have inputs and outputs
            self.clauses += implies([tile.is_belt], [tile.input_direction, tile.output_direction])
            # Underground inputs must have an input, but no output
            self.clauses += implies([tile.is_underground_in], [tile.input_direction] + set_all_false(tile.output_direction))
            # Underground outputs must have an output, but no input
            self.clauses += implies([tile.is_underground_out], [tile.output_direction] + set_all_false(tile.input_direction))
            # Splitters must have at least one input/output
            self.clauses += implies([tile.is_splitter], [tile.all_direction])

            self.clauses += quadratic_amo(tile.input_direction)  # Have an input direction or nothing
            self.clauses += quadratic_amo(tile.output_direction)  # Have an output direction or nothing

            self.clauses += quadratic_amo(tile.underground[0::2])  # Have a underground along -x, +x or nothing
            self.clauses += quadratic_amo(tile.underground[1::2])  # Have a underground along -y, +y or nothing

            # If a tile is a splitter head, then it is a splitter
            self.clauses.append([-tile.is_splitter_head, tile.is_splitter])

            # Splitters must output the same side as their input or have no output
            for direction in Direction:
                output = tile.output_direction.copy()
                del output[direction]

                self.clauses += implies([tile.is_splitter, tile.input_direction[direction]], set_all_false(output))

                input = tile.input_direction.copy()
                del input[direction]
                self.clauses += implies([tile.is_splitter, tile.output_direction[direction]], set_all_false(input))

            for direction in Direction:
                # Cannot input from same side as output
                self.clauses += quadratic_amo([tile.input_direction[direction], tile.output_direction[direction.reverse]])

                # Cannot have a turn and be a splitter
                self.clauses += implies([tile.is_splitter, tile.input_direction[direction]], [[-tile.output_direction[direction.next]]])

            # Prevent colours beyond end of range
            if self.colours is not None:
                for colour_range in (tile.colour, tile.colour_ux, tile.colour_uy):
                    self.clauses += set_maximum(self.colours - 1, colour_range)

        for direction in Direction:
            inv_direction = direction.reverse
            for tile_a, tile_b in self.iterate_tile_lines(direction.next.vec, 2):
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

        # Inputs and outputs
        for x in range(self.width):
            for y in range(self.height):
                tile = self.get_tile_instance(x, y)
                if x == 0:
                    # Top left corner
                    if y == 0:
                        self.clauses += implies([tile.is_input], [[tile.input_direction[Direction.RIGHT], tile.input_direction[Direction.DOWN]]])
                        self.clauses += implies([tile.input_direction[Direction.RIGHT]], [[tile.is_input]])
                        self.clauses += implies([tile.input_direction[Direction.DOWN]], [[tile.is_input]])
                        self.clauses += implies([tile.is_output], [[tile.output_direction[Direction.LEFT], tile.output_direction[Direction.UP]]])
                        self.clauses += implies([tile.output_direction[Direction.LEFT]], [[tile.is_output]])
                        self.clauses += implies([tile.output_direction[Direction.UP]], [[tile.is_output]])
                    # Bottom left corner
                    elif y == self.height - 1:
                        self.clauses += implies([tile.is_input], [[tile.input_direction[Direction.RIGHT], tile.input_direction[Direction.UP]]])
                        self.clauses += implies([tile.input_direction[Direction.RIGHT]], [[tile.is_input]])
                        self.clauses += implies([tile.input_direction[Direction.UP]], [[tile.is_input]])
                        self.clauses += implies([tile.is_output], [[tile.output_direction[Direction.LEFT], tile.output_direction[Direction.DOWN]]])
                        self.clauses += implies([tile.output_direction[Direction.LEFT]], [[tile.is_output]])
                        self.clauses += implies([tile.output_direction[Direction.DOWN]], [[tile.is_output]])
                    # Left edge
                    else:
                        self.clauses += implies([tile.is_input], [[tile.input_direction[Direction.RIGHT]]])
                        self.clauses += implies([tile.input_direction[Direction.RIGHT]], [[tile.is_input]])
                        self.clauses += implies([tile.is_output], [[tile.output_direction[Direction.LEFT]]])
                        self.clauses += implies([tile.output_direction[Direction.LEFT]], [[tile.is_output]])
                elif x == self.width - 1:
                    # Top right corner
                    if y == 0:
                        self.clauses += implies([tile.is_input], [[tile.input_direction[Direction.LEFT], tile.input_direction[Direction.DOWN]]])
                        self.clauses += implies([tile.input_direction[Direction.LEFT]], [[tile.is_input]])
                        self.clauses += implies([tile.input_direction[Direction.DOWN]], [[tile.is_input]])
                        self.clauses += implies([tile.is_output], [[tile.output_direction[Direction.RIGHT], tile.output_direction[Direction.UP]]])
                        self.clauses += implies([tile.output_direction[Direction.RIGHT]], [[tile.is_output]])
                        self.clauses += implies([tile.output_direction[Direction.UP]], [[tile.is_output]])
                    # Bottom right corner
                    elif y == self.height - 1:
                        self.clauses += implies([tile.is_input], [[tile.input_direction[Direction.LEFT], tile.input_direction[Direction.UP]]])
                        self.clauses += implies([tile.input_direction[Direction.LEFT]], [[tile.is_input]])
                        self.clauses += implies([tile.input_direction[Direction.UP]], [[tile.is_input]])
                        self.clauses += implies([tile.is_output], [[tile.output_direction[Direction.RIGHT], tile.output_direction[Direction.DOWN]]])
                        self.clauses += implies([tile.output_direction[Direction.RIGHT]], [[tile.is_output]])
                        self.clauses += implies([tile.output_direction[Direction.DOWN]], [[tile.is_output]])
                    # Right edge
                    else:
                        self.clauses += implies([tile.is_input], [[tile.input_direction[Direction.LEFT]]])
                        self.clauses += implies([tile.input_direction[Direction.LEFT]], [[tile.is_input]])
                        self.clauses += implies([tile.is_output], [[tile.output_direction[Direction.RIGHT]]])
                        self.clauses += implies([tile.output_direction[Direction.RIGHT]], [[tile.is_output]])
                elif y == 0:
                    # Top edge (corners handled on vertical edges)
                    self.clauses += implies([tile.is_input], [[tile.input_direction[Direction.DOWN]]])
                    self.clauses += implies([tile.input_direction[Direction.DOWN]], [[tile.is_input]])
                    self.clauses += implies([tile.is_output], [[tile.output_direction[Direction.UP]]])
                    self.clauses += implies([tile.output_direction[Direction.UP]], [[tile.is_output]])
                elif y == self.height - 1:
                    # Bottom edge (corners handled on vertical edges)
                    self.clauses += implies([tile.is_input], [[tile.input_direction[Direction.UP]]])
                    self.clauses += implies([tile.input_direction[Direction.UP]], [[tile.is_input]])
                    self.clauses += implies([tile.is_output], [[tile.output_direction[Direction.DOWN]]])
                    self.clauses += implies([tile.output_direction[Direction.DOWN]], [[tile.is_output]])

                else:
                    # Not on edge
                    self.clauses.append([-tile.is_input])
                    self.clauses.append([-tile.is_output])

    def set_tile(self, x: int, y: int, tile: BaseTile):
        tile_instance = self.get_tile_instance(x, y)

        if isinstance(tile, EmptyTile):
            self.clauses.append([tile_instance.is_empty])
        elif isinstance(tile, FillerTile):
            self.clauses.append([tile_instance.is_empty])
        elif isinstance(tile, Splitter):
            self.clauses.append([tile_instance.is_splitter])
            self.clauses.append([set_literal(tile_instance.is_splitter_head, tile.is_head)])
            self.clauses.append([tile_instance.input_direction[tile.direction], tile_instance.output_direction[tile.direction]])
        elif isinstance(tile, Belt) or isinstance(tile, UndergroundBelt):
            if isinstance(tile, Belt):
                self.clauses.append([tile_instance.is_belt])
            elif isinstance(tile, UndergroundBelt):
                self.clauses.append([tile_instance.is_underground_in if tile.is_input else tile_instance.is_underground_out])

            if tile.input_direction is not None:
                self.clauses += [[tile_instance.input_direction[tile.input_direction]]]

            if tile.output_direction is not None:
                self.clauses += [[tile_instance.output_direction[tile.output_direction]]]
        else:
            raise RuntimeError(f'Unsupported tile type {tile}')

    def read_tile(self, cell: Dict[str, Any]) -> BaseTile:
        input_direction = cell['input_direction']
        output_direction = cell['output_direction']
        if input_direction is not None:
            input_direction = Direction(input_direction)
        if output_direction is not None:
            output_direction = Direction(output_direction)

        if cell['is_splitter']:
            direction = input_direction
            if direction is None:
                direction = output_direction
                assert direction is not None
            return Splitter(direction, cell['is_splitter_head'])
        elif cell['is_empty']:
            return EmptyTile()
        elif cell['is_underground_in']:
            assert input_direction is not None
            return UndergroundBelt(input_direction, True)
        elif cell['is_underground_out']:
            assert output_direction is not None
            return UndergroundBelt(output_direction, False)
        elif cell['is_belt']:
            return Belt(input_direction, output_direction)
        else:
            assert False

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

    def transport_quantity(self,
                           quantity: Callable[[TileTemplate], NestedArray[LiteralType]],
                           quantity_ux: Callable[[TileTemplate], NestedArray[LiteralType]],
                           quantity_uy: Callable[[TileTemplate], NestedArray[LiteralType]]):
        for direction in Direction:
            dx, dy = direction.vec
            for x in range(self.width):
                for y in range(self.height):
                    tile_a = self.get_tile_instance(x, y)
                    tile_b = self.get_tile_instance_offset(x, y, dx, dy)

                    if tile_b is None:
                        continue

                    quantity_a = flatten(quantity(tile_a))
                    quantity_b = flatten(quantity(tile_b))

                    if direction.axis == Axis.HORIZONTAL:
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
                    self.clauses += implies(
                        [
                            tile_a.input_direction[direction],
                            -tile_a.is_splitter,
                            *invert_components(tile_a.output_direction)
                        ],
                        set_numbers_equal(quantity_a, quantity_ub))
                    self.clauses += implies(
                        [
                            tile_b.output_direction[direction],
                            -tile_b.is_splitter,
                            *invert_components(tile_b.input_direction)
                        ],
                        set_numbers_equal(quantity_ua, quantity_b))

    def prevent_bad_colouring(self):
        if self.colours in (1, None):
            return
        self.transport_quantity(lambda tile: tile.colour, lambda tile: tile.colour_ux, lambda tile: tile.colour_uy)

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
                self.clauses += implies([tile.input_direction[Direction.LEFT]], [[-tile.is_underground_in]])
                self.clauses += implies([tile.output_direction[Direction.RIGHT]], [[-tile.is_underground_out]])
        if max_x_blocked:
            for y in range(self.height):
                tile = self.get_tile_instance(self.width - 1, y)
                self.clauses += set_all_false(tile.underground[0::2])
                self.clauses += implies([tile.input_direction[Direction.RIGHT]], [[-tile.is_underground_in]])
                self.clauses += implies([tile.output_direction[Direction.LEFT]], [[-tile.is_underground_out]])
        if min_y_blocked:
            for x in range(self.width):
                tile = self.get_tile_instance(x, 0)
                self.clauses += set_all_false(tile.underground[1::2])
                self.clauses += implies([tile.input_direction[Direction.UP]], [[-tile.is_underground_in]])
                self.clauses += implies([tile.output_direction[Direction.DOWN]], [[-tile.is_underground_out]])
        if max_y_blocked:
            for x in range(self.width):
                tile = self.get_tile_instance(x, self.height - 1)
                self.clauses += set_all_false(tile.underground[1::2])
                self.clauses += implies([tile.input_direction[Direction.DOWN]], [[-tile.is_underground_in]])
                self.clauses += implies([tile.output_direction[Direction.UP]], [[-tile.is_underground_out]])

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

    def prevent_bad_undergrounding(self):
        for direction in Direction:
            reverse_dir = direction.reverse

            dx, dy = direction.vec
            for x in range(self.width):
                for y in range(self.height):
                    tile_a = self.get_tile_instance(x, y)

                    # Underground entrance/exit cannot be above underground segment with same direction
                    self.clauses += implies(
                        [tile_a.is_underground_in, tile_a.input_direction[direction]],
                        [[-tile_a.underground[direction]], [-tile_a.underground[reverse_dir]]]
                    )
                    self.clauses += implies(
                        [tile_a.is_underground_out, tile_a.output_direction[direction]],
                        [[-tile_a.underground[direction]], [-tile_a.underground[reverse_dir]]]
                    )

                    # Underground entrance/exit must have a underground segment after/before it
                    tile_b = self.get_tile_instance_offset(x, y, +dx, +dy)
                    if tile_b is not None:
                        self.clauses += implies([tile_a.is_underground_in, tile_a.input_direction[direction]], [[tile_b.underground[direction]]])

                    tile_b = self.get_tile_instance_offset(x, y, -dx, -dy)
                    if tile_b is not None:
                        self.clauses += implies([tile_a.is_underground_out, tile_a.output_direction[direction]], [[tile_b.underground[direction]]])

                    # Underground segment must propagate or have output
                    tile_b = self.get_tile_instance_offset(x, y, +dx, +dy)
                    if tile_b is not None:
                        self.clauses += implies(
                            [tile_a.underground[direction], -tile_b.underground[direction]],
                            [
                                [tile_b.is_underground_out],
                                [tile_b.output_direction[direction]],
                            ]
                        )

                    tile_b = self.get_tile_instance_offset(x, y, -dx, -dy)
                    if tile_b is not None:
                        self.clauses += implies(
                            [tile_a.underground[direction], -tile_b.underground[direction]],
                            [
                                [tile_b.is_underground_in],
                                [tile_b.input_direction[direction]],
                            ]
                        )

    def enforce_maximum_underground_length(self):
        assert self.underground_length >= 1

        if self.underground_length == float('inf'):
            return

        for direction in Direction:
            dx, dy = direction.vec
            for x in range(self.width):
                for y in range(self.height):
                    clause = []
                    for i in range(self.underground_length + 1):
                        tile = self.get_tile_instance_offset(x, y, dx * i, dy * i)

                        if tile is None:
                            break

                        clause.append(-tile.underground[direction])
                    else:
                        self.clauses.append(clause)

    def prevent_intersection(self):
        for direction in Direction:
            for tile_a, tile_b in self.iterate_tile_lines(direction.vec, 2):
                if tile_b is None:
                    continue

                self.clauses += literals_same(tile_a.output_direction[direction], tile_b.input_direction[direction])

                # Handles special splitter output case
                self.clauses += implies([tile_a.input_direction[direction], tile_a.is_splitter, -tile_b.is_splitter], [
                    [-tile_b.input_direction[direction.next], -tile_b.output_direction[direction.next]],
                    [-tile_b.input_direction[direction.prev], -tile_b.output_direction[direction.prev]],

                    [-tile_b.input_direction[direction.next], -tile_b.output_direction[direction]],
                    [-tile_b.input_direction[direction.prev], -tile_b.output_direction[direction]],

                    [-tile_b.input_direction[direction.reverse], -tile_b.output_direction[direction.next]],
                    [-tile_b.input_direction[direction.reverse], -tile_b.output_direction[direction.prev]],
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
