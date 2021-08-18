from cardinality import quadratic_amo, quadratic_one
from typing import *

import numpy as np

from util import *

class Grid(BaseGrid):
    def __init__(self, width: int, height: int, colours: int, extras: Optional[TileTemplate]=None):
        assert colours >= 1
        self.colours = colours

        self.colour_bits = bin_length(colours + 1)

        template = TileTemplate({
            'is_empty'           : 'bool',
            'is_belt'            : 'bool',
            'is_underground_in'  : 'bool',
            'is_underground_out' : 'bool',
            'is_splitter'        : 'bool',
            'type'               : 'alias is_empty is_belt is_underground_in is_underground_out is_splitter',
            
            'input_direction'    : 'one_hot 4', 
            'output_direction'   : 'one_hot 4',
            'all_direction'      : 'alias input_direction output_direction',

            'splitter_side'      : 'bool',
            'splitter_direction' : 'one_hot 4',
            
            'underground'        : 'arr 4', 
            'colour'             : 'num ' + str(self.colour_bits), # colour_in
            'colour_out'         : 'num ' + str(self.colour_bits),
            'colour_ux'          : 'num ' + str(self.colour_bits),
            'colour_uy'          : 'num ' + str(self.colour_bits),
        })

        if extras is not None:
            template = template.merge(extras)
        super().__init__(template, width, height)
        
        for y in range(height):
            for x in range(width):
                tile = self.get_tile_instance(x, y)
                self.clauses += quadratic_one(tile.type)
                self.clauses += quadratic_amo(tile.input_direction)  # Have an input direction or nothing
                self.clauses += quadratic_amo(tile.output_direction) # Have an output direction or nothing
                self.clauses += quadratic_amo(tile.underground[0::2]) # Have a underground along -x, +x or nothing
                self.clauses += quadratic_amo(tile.underground[1::2]) # Have a underground along -y, +y or nothing
                self.clauses += quadratic_amo(tile.splitter_direction)

                self.clauses += implies([tile.is_empty], set_number(0, tile.all_direction))

                self.clauses += implies([tile.is_splitter], [tile.splitter_direction, tile.all_direction])
                self.clauses += implies([-tile.is_splitter], set_number(0, tile.splitter_direction + [tile.splitter_side]))
                for direction in range(4):
                    self.clauses += implies([tile.is_splitter, tile.output_direction[direction]], [[tile.splitter_direction[direction]]])
                    self.clauses += implies([tile.is_splitter, tile.input_direction[direction]], [[tile.splitter_direction[direction]]])

                self.clauses += implies([tile.is_belt], [tile.input_direction, tile.output_direction])
                self.clauses += implies([tile.is_underground_in],  [tile.input_direction] + set_number(0, tile.output_direction))
                self.clauses += implies([tile.is_underground_out], set_number(0, tile.input_direction) + [tile.output_direction])

                # Cannot output opposite of input
                for direction in range(4):
                    self.clauses += quadratic_amo([tile.input_direction[direction], tile.output_direction[(direction + 2) % 4]])

                # Colour is 0 iff there is no input/output
                self.clauses += implies(invert_components(tile.input_direction),  set_number(0, tile.colour))
                self.clauses += implies(invert_components(tile.output_direction), set_number(0, tile.colour_out))
                self.clauses += implies(invert_components(tile.colour),           set_number(0, tile.input_direction))
                self.clauses += implies(invert_components(tile.colour_out),       set_number(0, tile.output_direction))
                self.clauses += implies(invert_components(tile.underground[0::2]), set_number(0, tile.colour_ux))
                self.clauses += implies(invert_components(tile.underground[1::2]), set_number(0, tile.colour_uy))

                for colour_range in (tile.colour, tile.colour_out, tile.colour_ux, tile.colour_uy):
                    for colour in range(self.colours + 1, 1<<self.colour_bits):
                        self.clauses.append(set_not_number(colour, colour_range))
    
    def setup_multitile_entities(self, edge_mode: EdgeModeType):
        for x in range(self.width):
            for y in range(self.height):
                tile_a = self.get_tile_instance(x, y)
                for direction in range(4):
                    inv_direction = (direction + 2) % 4
                    tile_b = self.get_tile_instance_offset(x, y, *direction_to_vec((direction + 1) % 4), edge_mode)

                    if tile_b == BLOCKED_TILE:
                        # If no room for splitter's complementary side, then splitter cannot be placed here with then given direction
                        self.clauses.append([-tile_a.is_splitter, tile_a.splitter_side, -tile_a.splitter_direction[direction]])
                        self.clauses.append([-tile_a.is_splitter, -tile_a.splitter_side, -tile_a.splitter_direction[inv_direction]])
                        continue
                    if tile_b == IGNORED_TILE:
                        continue
                    
                    self.clauses += implies([tile_a.is_splitter, -tile_a.splitter_side, tile_a.splitter_direction[direction]], [[tile_b.is_splitter], [tile_b.splitter_side], [tile_b.splitter_direction[direction]]])
                    self.clauses += implies([tile_a.is_splitter,  tile_a.splitter_side, tile_a.splitter_direction[inv_direction]], [[tile_b.is_splitter], [-tile_b.splitter_side], [tile_b.splitter_direction[inv_direction]]])
    
    def set_tile(self, x: int, y: int, tile: Optional[BaseTile]):
        tile_instance = self.get_tile_instance(x, y)

        if tile is None:
            self.clauses += [[tile_instance.is_empty]]
        elif isinstance(tile, Splitter):
            self.clauses += [
                [tile_instance.is_splitter], 
                [set_variable(tile_instance.splitter_side, tile.side)], 
                [tile_instance.splitter_direction[tile.direction]],
            ]
        elif isinstance(tile, Belt):
            self.clauses += [
                [tile_instance.is_belt],
                [tile_instance.input_direction[tile.input_direction]],
                [tile_instance.output_direction[tile.output_direction]],
            ]
        elif isinstance(tile, UndergroundBelt):
            if tile.is_input:
                self.clauses += [
                    [tile_instance.is_underground_in],
                    [tile_instance.input_direction[tile.direction]],
                ]
            else:
                self.clauses += [
                    [tile_instance.is_underground_out],
                    [tile_instance.output_direction[tile.direction]],
                ]
        else:
            assert False
    
    def prevent_colour(self, colour: int):
        raise NotImplementedError
        for x in range(self.width):
            for y in range(self.height):
                tile = self.get_tile_instance(x, y)
                if colour == 0:
                    self.clauses += [[-var, *set_not_number(0, tile.colour)] for var in tile.all_direction]
                else:
                    self.clauses += [set_not_number(colour, colour_range) for colour_range in (tile.colour, tile.colour_ux, tile.colour_uy)]

    def set_colour(self, x: int, y: int, colour: int):
        # TODO fix for underground stuff
        assert 0 <= colour < self.colours
        tile = self.get_tile_instance(x, y)
        self.clauses += set_number(colour + 1, tile.colour)
        self.clauses += set_number(colour + 1, tile.colour_out)

    def prevent_bad_colouring(self, edge_mode: EdgeModeType):
        if self.colours == 1:
            return

        for direction in range(4):
            dx, dy = direction_to_vec(direction)
            for x in range(self.width):
                for y in range(self.height):
                    tile_a = self.get_tile_instance(x, y)
                    tile_b = self.get_tile_instance_offset(x, y, dx, dy, edge_mode)

                    if tile_b in (BLOCKED_TILE, IGNORED_TILE):
                        continue

                    if direction % 2 == 0:
                        colour_a = tile_a.colour_ux
                        colour_b = tile_b.colour_ux
                    else:
                        colour_a = tile_a.colour_uy
                        colour_b = tile_b.colour_uy

                    # Belt colours consistent
                    self.clauses += implies([tile_a.output_direction[direction]], set_numbers_equal(tile_a.colour_out, tile_b.colour))
                    
                    # Underground colours consistent
                    self.clauses += implies([tile_a.underground[direction], tile_b.underground[direction]], set_numbers_equal(colour_a, colour_b))

                    # Underground transition consistent
                    self.clauses += implies([tile_a.is_underground_in, tile_a.input_direction[direction]], set_numbers_equal(tile_a.colour, colour_b))
                    self.clauses += implies([tile_b.is_underground_out, tile_b.output_direction[direction]], set_numbers_equal(colour_a, tile_b.colour_out))

        for tile in self.iterate_tiles():
            self.clauses += implies([tile.is_belt], set_numbers_equal(tile.colour, tile.colour_out))
    
    def prevent_bad_undergrounding(self, edge_mode: EdgeModeType):
        for x in range(self.width):
            for y in range(self.height):
                tile_a = self.get_tile_instance(x, y)
                for direction in range(4):
                    reverse_dir = (direction + 2) % 4
                    dx, dy = direction_to_vec(direction)

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
                    clause = [
                        -tile_a.is_underground_in,
                        -tile_a.input_direction[direction],
                    ]

                    tile_b = self.get_tile_instance_offset(x, y, +dx, +dy, edge_mode)
                    if tile_b == BLOCKED_TILE:
                        self.clauses.append(clause)
                    elif tile_b != IGNORED_TILE:
                        clause.append(tile_b.underground[direction])
                        self.clauses.append(clause)

                    clause = [
                        -tile_a.is_underground_out,
                        -tile_a.output_direction[direction],
                    ]
                    tile_b = self.get_tile_instance_offset(x, y, -dx, -dy, edge_mode)
                    if tile_b == BLOCKED_TILE:
                        self.clauses.append(clause)
                    elif tile_b != IGNORED_TILE:
                        clause.append(tile_b.underground[direction])
                        self.clauses.append(clause)

                    # Underground segment must propagate or have output
                    tile_b = self.get_tile_instance_offset(x, y, +dx, +dy, edge_mode)
                    if tile_b == BLOCKED_TILE:
                        self.clauses.append([-tile_a.underground[direction]])
                    elif tile_b != IGNORED_TILE:
                        self.clauses += implies(
                            [tile_a.underground[direction], -tile_b.underground[direction]], 
                            [
                                [tile_b.is_underground_out],
                                [tile_b.output_direction[direction]], 
                            ]
                        )
                    
                    tile_b = self.get_tile_instance_offset(x, y, -dx, -dy, edge_mode)
                    if tile_b == BLOCKED_TILE:
                        self.clauses.append([-tile_a.underground[direction]])
                    elif tile_b != IGNORED_TILE:
                        self.clauses += implies(
                            [tile_a.underground[direction], -tile_b.underground[direction]], 
                            [
                                [tile_b.is_underground_in],
                                [tile_b.input_direction[direction]], 
                            ]
                        )
    
    def set_maximum_underground_length(self, length, edge_mode: EdgeModeType):
        assert length >= 1

        for direction in range(4):
            dx, dy = direction_to_vec(direction)
            for x in range(self.width):
                for y in range(self.height):
                    clause = []
                    for i in range(length + 1):
                        tile = self.get_tile_instance_offset(x, y, dx * i, dy * i, edge_mode)

                        if tile in (BLOCKED_TILE, IGNORED_TILE):
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
                    if tile_b == BLOCKED_TILE:
                        self.clauses += [
                            [-tile_a.output_direction[direction]],
                            [-tile_a.input_direction[inv_direction]],
                            [-tile_a.splitter_direction[direction]],
                        ]
                    elif tile_b != IGNORED_TILE:
                        self.clauses += variables_same(tile_a.output_direction[direction], tile_b.input_direction[direction])

                        # Handles special splitter output case
                        self.clauses += implies([tile_a.splitter_direction[direction], tile_a.is_splitter, -tile_b.is_splitter], [
                            [-tile_b.input_direction[(direction + 1) % 4]],
                            [-tile_b.input_direction[(direction - 1) % 4]],
                            [tile_b.input_direction[direction], -tile_b.output_direction[(direction + 1) % 4]],
                            [tile_b.input_direction[direction], -tile_b.output_direction[(direction - 1) % 4]],
                        ])

    def prevent_empty_along_underground(self, length, edge_mode: EdgeModeType):
        assert length >= 1

        for direction in range(4):
            dx, dy = direction_to_vec(direction)
            for x in range(self.width):
                for y in range(self.height):
                    tiles = [self.get_tile_instance(x, y)]
                    for i in range(1, length + 2):
                        new_tile = self.get_tile_instance_offset(x, y, dx * i, dy * i, edge_mode)
                        if new_tile in (BLOCKED_TILE, IGNORED_TILE):
                            break

                        tiles.append(new_tile)
                        
                        start, *middle, end = tiles
                        if len(middle) > 0:
                            clause = [
                                -start.is_underground_in,
                                -start.input_direction[direction],
                            ]

                            for tile in middle:
                                clause.append(-tile.is_empty)
                            
                            clause += [
                                -end.is_underground_out,
                                -end.output_direction[direction],
                            ]

                            self.clauses.append(clause)

    def prevent_small_loops(self):
        for x in range(self.width-1):
            for y in range(self.height-1):
                tile00 = self.get_tile_instance(x+0, y+0)
                tile01 = self.get_tile_instance(x+0, y+1)
                tile10 = self.get_tile_instance(x+1, y+0)
                tile11 = self.get_tile_instance(x+1, y+1)

                self.clauses.append([
                    -tile00.input_direction[2],
                    -tile00.output_direction[3],

                    -tile01.input_direction[3],
                    -tile01.output_direction[0],

                    -tile11.input_direction[0],
                    -tile11.output_direction[1],

                    -tile10.input_direction[1],
                    -tile10.output_direction[2],
                ])

                self.clauses.append([
                    -tile00.input_direction[1],
                    -tile00.output_direction[0],

                    -tile10.input_direction[0],
                    -tile10.output_direction[3],

                    -tile11.input_direction[3],
                    -tile11.output_direction[2],

                    -tile01.input_direction[2],
                    -tile01.output_direction[1],
                ])

    def itersolve(self, ignore_colour=False, solver='g3'):
        important_variables = set()
        for x in range(self.width):
            for y in range(self.height):
                tile = self.get_tile_instance(x, y)

                important_variables |= set(tile.all_direction + tile.type)

                if not ignore_colour:
                    important_variables |= set(tile.colour + tile.colour_out + tile.colour_ux + tile.colour_uy)
        return super().itersolve(important_variables, solver)

BELT_TILES = [Belt(direction, (direction + curve) % 4) for direction in range(4) for curve in range(-1, 2)]
UNDERGROUND_TILES = [UndergroundBelt(direction, type) for direction in range(4) for type in range(2)]
SPLITTER_TILES = [Splitter(direction, i) for direction in range(4) for i in range(2)]
ALL_TILES = [None] + BELT_TILES + UNDERGROUND_TILES + SPLITTER_TILES
