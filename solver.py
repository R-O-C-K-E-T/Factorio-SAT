from cardinality import quadratic_amo
from typing import *

import numpy as np

from util import *

class Grid(BaseGrid):
    def __init__(self, width: int, height: int, colours: int, extras: Optional[TileTemplate]=None):
        assert colours >= 1
        self.colours = colours

        self.colour_bits = bin_length(colours)

        template = TileTemplate({
            'input_direction'  : 'one_hot 4', 
            'output_direction' : 'one_hot 4',
            'all_direction'    : 'alias input_direction output_direction',
            'is_splitter'      : 'one_hot 2',
            'underground'      : 'arr 4', 
            'colour'           : 'num ' + str(self.colour_bits),
            'colour_ux'        : 'num ' + str(self.colour_bits),
            'colour_uy'        : 'num ' + str(self.colour_bits),
        })

        if extras is not None:
            template = template.merge(extras)
        super().__init__(template, width, height)
        
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

                        self.clauses += implies([splitter_type, tile.input_direction[direction]], set_number(0, output))

                        input = tile.input_direction.copy()
                        del input[direction]
                        self.clauses += implies([splitter_type, tile.output_direction[direction]], set_number(0, input))

                for direction in range(4):
                    # Cannot input from same side as output
                    self.clauses += quadratic_amo([tile.input_direction[direction], tile.output_direction[(direction + 2) % 4]])

                    # Cannot have a turn and be a splitter
                    for splitter in tile.is_splitter:
                        self.clauses += implies([splitter, tile.input_direction[direction]], [[-tile.output_direction[(direction + 1) % 4]]])
                    
                # Prevent colours beyond end of range
                for colour_range in (tile.colour, tile.colour_ux, tile.colour_uy):
                    self.clauses += set_maximum(self.colours - 1, colour_range)
        
        for y0 in range(height):
            for x0 in range(width):
                tile_a = self.get_tile_instance(x0, y0)

                for direction in range(4):
                    inv_direction = (direction + 2) % 4
                    dx, dy = direction_to_vec((direction + 1) % 4)
                    
                    x1, y1 = x0 + dx, y0 + dy
                    if x1 < 0 or x1 >= width or y1 < 0 or y1 >= height:
                        # If no room for splitter's complementary side, then splitter cannot be placed here with then given direction
                        self.clauses += [
                            [-tile_a.input_direction[direction],      -tile_a.is_splitter[0]],
                            [-tile_a.output_direction[direction],     -tile_a.is_splitter[0]],
                            [-tile_a.input_direction[inv_direction],  -tile_a.is_splitter[1]],
                            [-tile_a.output_direction[inv_direction], -tile_a.is_splitter[1]],
                        ]
                        continue
                    
                    tile_b = self.get_tile_instance(x1, y1)
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
            self.clauses += [[-variable] for variable in tile_instance.all_direction + tile_instance.is_splitter]
        elif isinstance(tile, Splitter):
            self.clauses.append([tile_instance.is_splitter[tile.side]])
            self.clauses.append([tile_instance.input_direction[tile.direction], tile_instance.output_direction[tile.direction]])
        elif isinstance(tile, Belt) or isinstance(tile, UndergroundBelt):
            self.clauses += [[-tile_instance.is_splitter[0]], [-tile_instance.is_splitter[1]]]
            if tile.input_direction is None:
                self.clauses += [[-variable] for variable in tile_instance.input_direction]
            else:
                self.clauses += [[tile_instance.input_direction[tile.input_direction]]]

            if tile.output_direction is None:
                self.clauses += [[-variable] for variable in tile_instance.output_direction]
            else:
                self.clauses += [[tile_instance.output_direction[tile.output_direction]]]
    
    def prevent_colour(self, colour: int):
        for x in range(self.width):
            for y in range(self.height):
                tile = self.get_tile_instance(x, y)
                if colour == 0:
                    self.clauses += [[-var, *set_not_number(0, tile.colour)] for var in tile.all_direction]
                else:
                    self.clauses += [set_not_number(colour, colour_range) for colour_range in (tile.colour, tile.colour_ux, tile.colour_uy)]

    def set_colour(self, x: int, y: int, colour: int):
        assert 0 <= colour < self.colours
        tile = self.get_tile_instance(x, y)
        self.clauses += set_number(colour, tile.colour)

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
                    self.clauses += implies([tile_a.output_direction[direction],  *invert_components(tile_a.is_splitter)], set_numbers_equal(tile_a.colour, tile_b.colour))
                    
                    # Underground colours consistent
                    self.clauses += implies([tile_a.underground[direction]], set_numbers_equal(colour_a, colour_b))

                    # Underground transition consistent
                    self.clauses += implies([tile_a.input_direction[direction], *invert_components(tile_a.output_direction + tile_a.is_splitter)], set_numbers_equal(tile_a.colour, colour_b))
                    self.clauses += implies([tile_b.output_direction[direction], *invert_components(tile_b.input_direction + tile_b.is_splitter)], set_numbers_equal(colour_a, tile_b.colour))
    
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
                    if tile_b == BLOCKED_TILE:
                        self.clauses.append(clause)
                    elif tile_b != IGNORED_TILE:
                        clause.append(tile_b.underground[direction])
                        self.clauses.append(clause)

                    clause = [
                        -tile_a.output_direction[direction],
                        *tile_a.input_direction,
                        *tile_a.is_splitter,
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
                                [tile_b.output_direction[direction]], 
                                *([-tile_b.input_direction[i]] for i in range(4)),
                                [-tile_b.is_splitter[0]], 
                                [-tile_b.is_splitter[1]],
                            ]
                        )
                    
                    tile_b = self.get_tile_instance_offset(x, y, -dx, -dy, edge_mode)
                    if tile_b == BLOCKED_TILE:
                        self.clauses.append([-tile_a.underground[direction]])
                    elif tile_b != IGNORED_TILE:
                        self.clauses += implies(
                            [tile_a.underground[direction], -tile_b.underground[direction]], 
                            [
                                [tile_b.input_direction[direction]], 
                                *([-tile_b.output_direction[i]] for i in range(4)),
                                [-tile_b.is_splitter[0]], 
                                [-tile_b.is_splitter[1]],
                            ]
                        )
    
    def set_maximum_underground_length(self, length: int, edge_mode: EdgeModeType):
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
                            *implies([tile_a.input_direction[direction]],      [[-variable] for variable in tile_a.is_splitter]),
                            *implies([tile_a.output_direction[inv_direction]], [[-variable] for variable in tile_a.is_splitter]),
                        ]
                    elif tile_b != IGNORED_TILE:
                        self.clauses += variables_same(tile_a.output_direction[direction], tile_b.input_direction[direction])

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

    def prevent_empty_along_underground(self, underground_length: int, edge_mode: EdgeModeType):
        for direction in range(4):
            dx, dy = direction_to_vec(direction)
            for x in range(self.width):
                for y in range(self.height):
                    tiles = [self.get_tile_instance(x, y)]
                    for i in range(1, underground_length+2):
                        new_tile = self.get_tile_instance_offset(x, y, dx * i, dy * i, edge_mode)
                        if new_tile in (BLOCKED_TILE, IGNORED_TILE):
                            break

                        tiles.append(new_tile)
                        
                        start, *middle, end = tiles
                        if len(middle) > 0:
                            clause = [
                                -start.input_direction[direction],
                                *start.output_direction
                            ]

                            for tile in middle:
                                clause += tile.all_direction
                            
                            clause += [
                                *end.input_direction,
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

                important_variables |= set(tile.all_direction + tile.is_splitter)

                if not ignore_colour:
                    important_variables |= set(tile.colour + tile.colour_ux + tile.colour_uy)
        return super().itersolve(important_variables, solver)

BELT_TILES = [Belt(direction, (direction + curve) % 4) for direction in range(4) for curve in range(-1, 2)]
UNDERGROUND_TILES = [UndergroundBelt(direction, type) for direction in range(4) for type in range(2)]
SPLITTER_TILES = [Splitter(direction, i) for direction in range(4) for i in range(2)]
ALL_TILES = [None] + BELT_TILES + UNDERGROUND_TILES + SPLITTER_TILES
