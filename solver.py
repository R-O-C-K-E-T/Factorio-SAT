from cardinality import quadratic_amo
from typing import *

import numpy as np

from template import *
from util import *


class TileTemplate(Protocol):
    input_direction: List[LiteralType]
    output_direction: List[LiteralType]
    all_direction: List[LiteralType]
    is_splitter: List[LiteralType]
    underground: List[LiteralType]
    colour: List[LiteralType]
    colour_ux: List[LiteralType]
    colour_uy: List[LiteralType]

class Grid(BaseGrid[TileTemplate, Dict[str, Any]]):
    def __init__(self, width: int, height: int, colours: Optional[int], underground_length: int=4, extras: CompositeTemplateParams={}, pool: Optional[IDPool]=None):
        assert colours is None or colours >= 1
        assert underground_length >= 0
        self.colours = colours
        self.underground_length = underground_length


        template = {
            'input_direction'  : OneHotTemplate(4), 
            'output_direction' : OneHotTemplate(4), 
            'all_direction'    : lambda input_direction, output_direction: [*input_direction, *output_direction],
            'is_splitter'      : OneHotTemplate(2), 
            'underground'      : ArrayTemplate(BoolTemplate(), (4,)),
        }
        if colours is not None:
            self.colour_bits = bin_length(colours)
            template.update({
                'colour'           : NumberTemplate(self.colour_bits),
                'colour_ux'        : NumberTemplate(self.colour_bits),
                'colour_uy'        : NumberTemplate(self.colour_bits),
            })
        else:
            self.colour_bits = None
        
        template.update(extras)
        template = CompositeTemplate(template)

        super().__init__(template, width, height, pool)
        
        for y in range(height):
            for x in range(width):
                tile = self.get_tile_instance(x, y)
                self.clauses += quadratic_amo(tile.input_direction)  # Have an input direction or nothing
                self.clauses += quadratic_amo(tile.output_direction) # Have an output direction or nothing
                self.clauses += quadratic_amo(tile.is_splitter)      # Have a splitter type or nothing

                self.clauses += quadratic_amo(tile.underground[0::2]) # Have a underground along -x, +x or nothing
                self.clauses += quadratic_amo(tile.underground[1::2]) # Have a underground along -y, +y or nothing

                # If the tile is a splitter, then it is not empty
                self.clauses += implies([tile.is_splitter[0]], [tile.all_direction])
                self.clauses += implies([tile.is_splitter[1]], [tile.all_direction])


                for splitter_type in tile.is_splitter:
                    for direction in range(4):
                        output = tile.output_direction.copy()
                        del output[direction]

                        self.clauses += implies([splitter_type, tile.input_direction[direction]], set_all_false(output))

                        input = tile.input_direction.copy()
                        del input[direction]
                        self.clauses += implies([splitter_type, tile.output_direction[direction]], set_all_false(input))

                for direction in range(4):
                    # Cannot input from same side as output
                    self.clauses += quadratic_amo([tile.input_direction[direction], tile.output_direction[(direction + 2) % 4]])

                    # Cannot have a turn and be a splitter
                    for splitter in tile.is_splitter:
                        self.clauses += implies([splitter, tile.input_direction[direction]], [[-tile.output_direction[(direction + 1) % 4]]])
                    
                # Prevent colours beyond end of range
                if self.colours is not None:
                    for colour_range in (tile.colour, tile.colour_ux, tile.colour_uy):
                        self.clauses += set_maximum(self.colours - 1, colour_range)
        
        for direction in range(4):
            inv_direction = (direction + 2) % 4
            for tile_a, tile_b in self.iterate_tile_lines(direction_to_vec((direction + 1) % 4), 2, EdgeMode.BLOCK):
                if tile_b == TileResult.BLOCKED:
                    self.clauses += [
                            [-tile_a.input_direction[direction],      -tile_a.is_splitter[0]],
                            [-tile_a.output_direction[direction],     -tile_a.is_splitter[0]],
                            [-tile_a.input_direction[inv_direction],  -tile_a.is_splitter[1]],
                            [-tile_a.output_direction[inv_direction], -tile_a.is_splitter[1]],
                        ]
                    continue
                for side in (tile_a.input_direction, tile_a.output_direction):
                    # Complementary side exists
                    self.clauses.append([
                        -side[direction],
                        -tile_a.is_splitter[0],
                            tile_b.is_splitter[1],
                    ])

                    self.clauses.append([
                        -side[inv_direction],
                        -tile_a.is_splitter[1],
                            tile_b.is_splitter[0],
                    ])

                    # Complementary side has input/output direction correct
                    self.clauses.append([
                        -side[direction],
                        -tile_a.is_splitter[0],
                        tile_b.input_direction[direction],
                        tile_b.output_direction[direction],
                    ])

                    self.clauses.append([
                        -side[inv_direction],
                        -tile_a.is_splitter[1],
                        tile_b.input_direction[inv_direction],
                        tile_b.output_direction[inv_direction],
                    ])
    
    def set_tile(self, x: int, y: int, tile: Optional[BaseTile]):
        tile_instance = self.get_tile_instance(x, y)

        if tile is None:
            self.clauses += [[-lit] for lit in tile_instance.all_direction + tile_instance.is_splitter]
        elif isinstance(tile, Splitter):
            self.clauses.append([tile_instance.is_splitter[tile.side]])
            self.clauses.append([tile_instance.input_direction[tile.direction], tile_instance.output_direction[tile.direction]])
        elif isinstance(tile, Belt) or isinstance(tile, UndergroundBelt):
            self.clauses += [[-tile_instance.is_splitter[0]], [-tile_instance.is_splitter[1]]]
            if tile.input_direction is None:
                self.clauses += [[-lit] for lit in tile_instance.input_direction]
            else:
                self.clauses += [[tile_instance.input_direction[tile.input_direction]]]

            if tile.output_direction is None:
                self.clauses += [[-lit] for lit in tile_instance.output_direction]
            else:
                self.clauses += [[tile_instance.output_direction[tile.output_direction]]]
    
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

                    if isinstance(tile_b, TileResult):
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
                    self.clauses += implies([tile_a.output_direction[direction],  *invert_components(tile_a.is_splitter)], set_numbers_equal(quantity_a, quantity_b))
                    
                    # Underground quantity consistent
                    self.clauses += implies([tile_a.underground[direction]], set_numbers_equal(quantity_ua, quantity_ub))

                    # Underground transition consistent
                    self.clauses += implies([tile_a.input_direction[direction], *invert_components(tile_a.output_direction + tile_a.is_splitter)], set_numbers_equal(quantity_a, quantity_ub))
                    self.clauses += implies([tile_b.output_direction[direction], *invert_components(tile_b.input_direction + tile_b.is_splitter)], set_numbers_equal(quantity_ua, quantity_b))
    

    def prevent_bad_colouring(self, edge_mode: EdgeModeType):
        if self.colours == 1:
            return
        self.transport_quantity(lambda tile: tile.colour, lambda tile: tile.colour_ux, lambda tile: tile.colour_uy, edge_mode)
        
    def prevent_bad_undergrounding(self, edge_mode: EdgeModeType):
        for direction in range(4):
            reverse_dir = (direction + 2) % 4
            
            dx, dy = direction_to_vec(direction)
            for x in range(self.width):
                for y in range(self.height):
                    tile_a = self.get_tile_instance(x, y)

                    # Underground entrance/exit cannot be above underground segment with same direction
                    self.clauses += implies(
                        [tile_a.input_direction[direction], *invert_components(tile_a.output_direction + tile_a.is_splitter)],
                        [[-tile_a.underground[direction]], [-tile_a.underground[reverse_dir]]]
                    )
                    self.clauses += implies(
                        [tile_a.output_direction[direction], *invert_components(tile_a.input_direction + tile_a.is_splitter)],
                        [[-tile_a.underground[direction]], [-tile_a.underground[reverse_dir]]]
                    )

                    # Underground entrance/exit must have a underground segment after/before it
                    clause = [
                        -tile_a.input_direction[direction],
                        *tile_a.output_direction,
                        *tile_a.is_splitter,
                    ]

                    tile_b = self.get_tile_instance_offset(x, y, +dx, +dy, edge_mode)
                    if tile_b == TileResult.BLOCKED:
                        self.clauses.append(clause)
                    elif tile_b != TileResult.IGNORED:
                        clause.append(tile_b.underground[direction])
                        self.clauses.append(clause)

                    clause = [
                        -tile_a.output_direction[direction],
                        *tile_a.input_direction,
                        *tile_a.is_splitter,
                    ]
                    tile_b = self.get_tile_instance_offset(x, y, -dx, -dy, edge_mode)
                    if tile_b == TileResult.BLOCKED:
                        self.clauses.append(clause)
                    elif tile_b != TileResult.IGNORED:
                        clause.append(tile_b.underground[direction])
                        self.clauses.append(clause)

                    # Underground segment must propagate or have output
                    tile_b = self.get_tile_instance_offset(x, y, +dx, +dy, edge_mode)
                    if tile_b == TileResult.BLOCKED:
                        self.clauses.append([-tile_a.underground[direction]])
                    elif tile_b != TileResult.IGNORED:
                        self.clauses += implies(
                            [tile_a.underground[direction], -tile_b.underground[direction]], 
                            [
                                [tile_b.output_direction[direction]], 
                                *([-tile_b.input_direction[i]] for i in range(4)),
                                [-tile_b.is_splitter[0]], 
                                [-tile_b.is_splitter[1]],
                            ]
                        )
                    
                    tile_b = self.get_tile_instance_offset(x, y, -dx, -dy, edge_mode)
                    if tile_b == TileResult.BLOCKED:
                        self.clauses.append([-tile_a.underground[direction]])
                    elif tile_b != TileResult.IGNORED:
                        self.clauses += implies(
                            [tile_a.underground[direction], -tile_b.underground[direction]], 
                            [
                                [tile_b.input_direction[direction]], 
                                *([-tile_b.output_direction[i]] for i in range(4)),
                                [-tile_b.is_splitter[0]], 
                                [-tile_b.is_splitter[1]],
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

                        if isinstance(tile, TileResult):
                            break
                        
                        clause.append(-tile.underground[direction])
                    else:
                        self.clauses.append(clause)

    def prevent_intersection(self, edge_mode: EdgeModeType):
        for x in range(self.width):
            for y in range(self.height):
                tile_a = self.get_tile_instance(x, y)
                for direction in range(4):
                    inv_direction = (direction + 2) % 4
                    dx, dy = direction_to_vec(direction)

                    tile_b = self.get_tile_instance_offset(x, y, dx, dy, edge_mode)
                    if tile_b == TileResult.BLOCKED:
                        self.clauses += [
                            [-tile_a.output_direction[direction]],
                            [-tile_a.input_direction[inv_direction]],
                            *implies([tile_a.input_direction[direction]],      [[-lit] for lit in tile_a.is_splitter]),
                            *implies([tile_a.output_direction[inv_direction]], [[-lit] for lit in tile_a.is_splitter]),
                        ]
                    elif tile_b != TileResult.IGNORED:
                        self.clauses += literals_same(tile_a.output_direction[direction], tile_b.input_direction[direction])

                        # Handles special splitter output case
                        for splitter in tile_a.is_splitter:
                            self.clauses += implies([tile_a.input_direction[direction], splitter, *invert_components(tile_b.is_splitter)], [   
                                [-tile_b.input_direction[(direction + 1) % 4], -tile_b.output_direction[(direction + 1) % 4]],
                                [-tile_b.input_direction[(direction - 1) % 4], -tile_b.output_direction[(direction - 1) % 4]],

                                [-tile_b.input_direction[(direction + 1) % 4], -tile_b.output_direction[direction]],
                                [-tile_b.input_direction[(direction - 1) % 4], -tile_b.output_direction[direction]],

                                [-tile_b.input_direction[(direction + 2) % 4], -tile_b.output_direction[(direction + 1) % 4]],
                                [-tile_b.input_direction[(direction + 2) % 4], -tile_b.output_direction[(direction - 1) % 4]],
                            ])

    def itersolve(self, ignore_colour=False, solver='g3'):
        important_variables = set()
        for x in range(self.width):
            for y in range(self.height):
                tile = self.get_tile_instance(x, y)

                important_variables |= set(tile.all_direction + tile.is_splitter)

                if not ignore_colour:
                    important_variables |= set(tile.colour + tile.colour_ux + tile.colour_uy)
        return super().itersolve(important_variables, solver)