from cardinality import quadratic_amo, quadratic_one
from typing import *

import numpy as np

from util import *
from template import *

# TODO Remove
import sys

class Grid(BaseGrid[NamedTuple, Dict[str, Any]]):
    def __init__(self, width: int, height: int, colour_bits: int, flow_bits: int, extras: CompositeTemplateParams={}):
        assert width > 0 and height > 0 and colour_bits >= 0 and flow_bits > 0
        template = CompositeTemplate({
            'is_empty'              : BoolTemplate(),
            'is_belt'               : BoolTemplate(),
            'is_underground_in'     : BoolTemplate(),
            'is_underground_out'    : BoolTemplate(),
            'is_underground'        : lambda is_underground_in, is_underground_out: [is_underground_in, is_underground_out], 
            'is_splitter'           : OneHotTemplate(2),
            'is_inserter'           : OneHotTemplate(2),
            'is_assembling_machine' : BoolTemplate(),
            'type'                  : lambda is_empty, is_belt, is_underground, is_splitter, is_inserter, is_assembling_machine: [is_empty, is_belt, *is_underground, *is_splitter, *is_inserter, is_assembling_machine],

            'assembling_x'          : OneHotTemplate(3),
            'assembling_y'          : OneHotTemplate(3),

            'input_direction'       : OneHotTemplate(4),
            'output_direction'      : OneHotTemplate(4),
            'all_direction'         : lambda input_direction, output_direction: [*input_direction, *output_direction],
            'colour_direction'      : OneHotTemplate(4),
            'inserter_direction'    : OneHotTemplate(4),

            'underground'           : ArrayTemplate(BoolTemplate(), (4,)),

            'colour'                : ArrayTemplate(NumberTemplate(colour_bits), (2,)),
            'colour_ux'             : ArrayTemplate(NumberTemplate(colour_bits), (2,)),
            'colour_uy'             : ArrayTemplate(NumberTemplate(colour_bits), (2,)),
            
            'flow_in'               : ArrayTemplate(NumberTemplate(flow_bits), (2,)),
            'flow_placed'           : ArrayTemplate(NumberTemplate(flow_bits + 1, is_signed=True), (2,2,4)), # (side,inserter_type,direction)
            'flow_out'              : ArrayTemplate(NumberTemplate(flow_bits), (2,)),

            'flow_splitter'         : ArrayTemplate(NumberTemplate(flow_bits + 1), (2,)), # Prevents overflow

            'flow_ux'               : ArrayTemplate(NumberTemplate(flow_bits), (2,)),
            'flow_uy'               : ArrayTemplate(NumberTemplate(flow_bits), (2,)),
            **extras,
        })
        super().__init__(template, width, height)

        for x0 in range(self.width):
            for y0 in range(self.height):
                tile = self.get_tile_instance(x0, y0)

                self.clauses += quadratic_one(tile.type)

                self.clauses += quadratic_amo(tile.input_direction)
                self.clauses += quadratic_amo(tile.output_direction)
                self.clauses += quadratic_amo(tile.underground[0::2])
                self.clauses += quadratic_amo(tile.underground[1::2])
                self.clauses += quadratic_amo(tile.assembling_x)
                self.clauses += quadratic_amo(tile.assembling_y)
                self.clauses += quadratic_amo(tile.colour_direction)
                self.clauses += quadratic_amo(tile.inserter_direction)

                self.clauses += implies([tile.is_empty], set_all_false(tile.all_direction + tile.flow_out[0] + tile.flow_out[1] + tile.colour[0] + tile.colour[1]))
                self.clauses += implies([tile.is_belt], [tile.all_direction, tile.colour_direction])
                self.clauses += implies([tile.is_underground_in],  [tile.input_direction]  + set_all_false(tile.output_direction))
                self.clauses += implies([tile.is_underground_out], [tile.output_direction, tile.colour_direction] + set_all_false(tile.input_direction))
                self.clauses += implies([tile.is_assembling_machine], set_all_false(tile.all_direction))

                self.clauses += implies(invert_components(tile.is_inserter), set_all_false(tile.inserter_direction))
                self.clauses += implies(invert_components([*tile.is_splitter, tile.is_belt, tile.is_underground_out]), set_all_false(tile.colour_direction))

                self.clauses += implies([-tile.is_splitter[0]], set_all_false(tile.flow_splitter[0] + tile.flow_splitter[1]))

                for i in range(2):
                    self.clauses += implies([tile.is_inserter[i]], [tile.inserter_direction] + set_all_false(tile.all_direction + tile.flow_out[0] + tile.flow_out[1] + tile.colour[0] + tile.colour[1]))

                for i in range(2):
                    self.clauses += implies([tile.is_splitter[i]], [tile.colour_direction])
                    for direction in range(4):
                        self.clauses += implies([tile.is_splitter[i], tile.input_direction[direction]],  [[tile.colour_direction[direction]]])
                        self.clauses += implies([tile.is_splitter[i], tile.output_direction[direction]], [[tile.colour_direction[direction]]])

                # There is an input/output iff there is flow
                self.clauses += implies([*invert_components(tile.input_direction),  -tile.is_underground_out], set_all_false(tile.flow_in[0]  + tile.flow_in[1]))
                self.clauses += implies([*invert_components(tile.output_direction), -tile.is_underground_in],  set_all_false(tile.flow_out[0] + tile.flow_out[1]))

                self.clauses += implies(invert_components(tile.flow_in[0]  + tile.flow_in[1]),  set_all_false(tile.input_direction  + [tile.is_underground_out]))
                self.clauses += implies(invert_components(tile.flow_out[0] + tile.flow_out[1]), set_all_false(tile.output_direction + [tile.is_underground_in]))

                # Assembling machine has coordinates
                self.clauses += implies([-tile.is_assembling_machine], set_all_false(tile.assembling_x + tile.assembling_y))
                self.clauses += implies([tile.is_assembling_machine], [tile.assembling_x, tile.assembling_y])

                # Underground flow/colour exists iff there is underground occuring
                self.clauses += implies(invert_components(tile.underground[0::2]), set_all_false(tile.flow_ux[0] + tile.flow_ux[1] + tile.colour_ux[0] + tile.colour_ux[1]))
                self.clauses += implies(invert_components(tile.underground[1::2]), set_all_false(tile.flow_uy[0] + tile.flow_uy[1] + tile.colour_uy[0] + tile.colour_uy[1]))
                self.clauses += [
                    [-tile.underground[0], *tile.flow_ux[0], *tile.flow_ux[1]],
                    [-tile.underground[1], *tile.flow_uy[0], *tile.flow_uy[1]],
                    [-tile.underground[2], *tile.flow_ux[0], *tile.flow_ux[1]],
                    [-tile.underground[3], *tile.flow_uy[0], *tile.flow_uy[1]],
                ]

                # Cannot output opposite of input
                for direction in range(4):
                    self.clauses += quadratic_amo([tile.input_direction[direction], tile.output_direction[(direction + 2) % 4]])

                for direction in range(4):
                    self.clauses += implies([tile.output_direction[direction]], [[tile.colour_direction[direction]]])
                    self.clauses += implies([tile.is_belt, *invert_components(tile.output_direction), tile.input_direction[direction]], [[tile.colour_direction[direction]]])

    def prevent_colour(self, colour: int):
        for x in range(self.width):
            for y in range(self.height):
                tile = self.get_tile_instance(x, y)
                for side in range(2):
                    self.clauses.append(set_not_number(colour, tile.colour[side]))
                    self.clauses.append(set_not_number(colour, tile.colour_ux[side]))
                    self.clauses.append(set_not_number(colour, tile.colour_uy[side]))

    def set_maximum_flow(self, colour: int, max_flow: int):
        for flow in range(max_flow + 1, 1 << len(self.get_tile_instance(0,0).flow_in[0])):
            for x in range(self.width):
                for y in range(self.height):
                    tile = self.get_tile_instance(x, y)
                    for side in range(2):
                        precondition = set_not_number(colour, tile.colour[side])
                        self.clauses.append(precondition + set_not_number(flow, tile.flow_in[side]))
                        self.clauses.append(precondition + set_not_number(flow, tile.flow_out[side]))
                        self.clauses.append(precondition + set_not_number(flow, tile.flow_ux[side]))
                        self.clauses.append(precondition + set_not_number(flow, tile.flow_uy[side]))

    def setup_multitile_entities(self, edge_mode: EdgeModeType):
        for x in range(self.width):
            for y in range(self.height):
                tile_a = self.get_tile_instance(x, y)
                for direction in range(4):
                    inv_direction = (direction + 2) % 4

                    # Splitter must have complementary side
                    tile_b = self.get_tile_instance_offset(x, y, *direction_to_vec((direction + 1) % 4), edge_mode)
                    if tile_b is not None:
                        self.clauses += implies(
                            [tile_a.colour_direction[direction], tile_a.is_splitter[0]], 
                            [[tile_b.is_splitter[1]], [tile_b.colour_direction[direction]]]
                        )

                        self.clauses += implies(
                            [tile_a.colour_direction[inv_direction], tile_a.is_splitter[1]], 
                            [[tile_b.is_splitter[0]], [tile_b.colour_direction[inv_direction]]]
                        )
                
                # Assembling machine must be 3x3
                tile_b = self.get_tile_instance_offset(x, y, +1,  0, edge_mode)
                if tile_b is not None:
                    self.clauses += literals_same(tile_a.assembling_x[1], tile_b.assembling_x[2])
                    for i in range(3):
                        self.clauses += implies([tile_a.is_assembling_machine, tile_b.is_assembling_machine], literals_same(tile_a.assembling_y[i], tile_b.assembling_y[i]))

                tile_b = self.get_tile_instance_offset(x, y, -1,  0, edge_mode)
                if tile_b is not None:
                    self.clauses += literals_same(tile_a.assembling_x[1], tile_b.assembling_x[0])
                    for i in range(3):
                        self.clauses += implies([tile_a.is_assembling_machine, tile_b.is_assembling_machine], literals_same(tile_a.assembling_y[i], tile_b.assembling_y[i]))
                
                tile_b = self.get_tile_instance_offset(x, y,  0, +1, edge_mode)
                if tile_b is not None:
                    self.clauses += literals_same(tile_a.assembling_y[1], tile_b.assembling_y[2])

                tile_b = self.get_tile_instance_offset(x, y,  0, -1, edge_mode)
                if tile_b is not None:
                    self.clauses += literals_same(tile_a.assembling_y[1], tile_b.assembling_y[0])

    def prevent_intersection(self, edge_mode: EdgeModeType):
        for x in range(self.width):
            for y in range(self.height):
                tile_a = self.get_tile_instance(x, y)
                for direction in range(4):
                    inv_direction = (direction + 2) % 4
                    dx, dy = direction_to_vec(direction)

                    tile_b = self.get_tile_instance_offset(x, y, dx, dy, edge_mode)
                    if tile_b is not None:
                        self.clauses += literals_same(tile_a.output_direction[direction], tile_b.input_direction[direction])

                        # Handles special output case where there is no output flow, but the belt/splitter is pointing into an invalid tile
                        self.clauses += implies([tile_a.colour_direction[direction]], [   
                            [-tile_b.input_direction[(direction + 1) % 4]],
                            [-tile_b.input_direction[(direction - 1) % 4]],
                            [tile_b.input_direction[direction], -tile_b.output_direction[(direction + 1) % 4]],
                            [tile_b.input_direction[direction], -tile_b.output_direction[(direction - 1) % 4]],

                            #[-tile_b.is_splitter[0], -tile_b.colour_direction[direction], tile_a.output_direction[direction]],
                            #[-tile_b.is_splitter[1], -tile_b.colour_direction[direction], tile_a.output_direction[direction]],
                        ])

    def prevent_bad_undergrounding(self, edge_mode: EdgeModeType):
        for direction in range(4):
            reverse_dir = (direction + 2) % 4

            dx, dy = direction_to_vec(direction)
            for x in range(self.width):
                for y in range(self.height):
                    tile_a = self.get_tile_instance(x, y)

                    self.clauses += implies(
                        [tile_a.is_underground_in, tile_a.input_direction[direction]],
                        [[-tile_a.underground[direction]], [-tile_a.underground[reverse_dir]]]
                    )
                    self.clauses += implies(
                        [tile_a.is_underground_out, tile_a.output_direction[direction]],
                        [[-tile_a.underground[direction]], [-tile_a.underground[reverse_dir]]]
                    )

                    clause = [
                        -tile_a.input_direction[direction],
                        -tile_a.is_underground_in,
                    ]
                    
                    tile_b = self.get_tile_instance_offset(x, y, +dx, +dy, edge_mode)
                    if tile_b is not None:
                        clause.append(tile_b.underground[direction])
                        self.clauses.append(clause)


                    clause = [
                        -tile_a.output_direction[direction],
                        -tile_a.is_underground_out,
                    ]
                    
                    tile_b = self.get_tile_instance_offset(x, y, -dx, -dy, edge_mode)
                    if tile_b is not None:
                        clause.append(tile_b.underground[direction])
                        self.clauses.append(clause)

                    

                    tile_b = self.get_tile_instance_offset(x, y, +dx, +dy, edge_mode)
                    if tile_b is not None:
                        self.clauses += implies(
                            [tile_a.underground[direction], -tile_b.underground[direction]], 
                            [
                                [tile_b.output_direction[direction]], 
                                [tile_b.is_underground_out],
                            ]
                        )
                    
                    tile_b = self.get_tile_instance_offset(x, y, -dx, -dy, edge_mode)
                    if tile_b is not None:
                        self.clauses += implies(
                            [tile_a.underground[direction], -tile_b.underground[direction]], 
                            [
                                [tile_b.input_direction[direction]], 
                                [tile_b.is_underground_in],
                            ]
                        )
    def set_maximum_underground_length(self, length: int, edge_mode: EdgeModeType):
        assert length > 0

        for direction in range(4):
            dx, dy = direction_to_vec(direction)

            for x in range(self.width):
                for y in range(self.height):
                    clause = []
                    for i in range(length + 1):
                        tile = self.get_tile_instance_offset(x, y, dx * i, dy * i, edge_mode)

                        if tile is None:
                            break

                        clause.append(-tile.underground[direction])
                    else:
                        self.clauses.append(clause)

    
    def prevent_empty_along_underground(self, maximum_underground_length, edge_mode: EdgeModeType):
        assert maximum_underground_length > 0

        for direction in range(4):
            dx, dy = direction_to_vec(direction)
            for x in range(self.width):
                for y in range(self.height):
                    tiles = [self.get_tile_instance(x, y)]
                    for i in range(1, maximum_underground_length + 2):
                        new_tile = self.get_tile_instance_offset(x, y, dx * i, dy * i, edge_mode)
                        if new_tile is None:
                            break

                        tiles.append(new_tile)

                        start, *middle, end = tiles
                        if len(middle) > 0:
                            clause = [-start.input_direction[direction], -start.is_underground_in]

                            for tile in middle:
                                clause.append(-tile.is_empty)
                            
                            clause += [-end.output_direction[direction], -end.is_underground_out]
                            self.clauses.append(clause)

    def prevent_bad_colouring(self, edge_mode: EdgeModeType):
        for direction in range(4):
            dx, dy = direction_to_vec(direction)

            for x in range(self.width):
                for y in range(self.height):
                    tile_a = self.get_tile_instance(x, y)
                    tile_b = self.get_tile_instance_offset(x, y, dx, dy, edge_mode)

                    if tile_b is not None:
                        if direction % 2 == 0:
                            colour_a = tile_a.colour_ux
                            colour_b = tile_b.colour_ux
                        else:
                            colour_a = tile_a.colour_uy
                            colour_b = tile_b.colour_uy

                        for side in range(2):
                            # Belt colours consistent
                            self.clauses += implies([tile_a.colour_direction[direction]], set_numbers_equal(tile_a.colour[side], tile_b.colour[side]))

                            # Underground colours consistent
                            self.clauses += implies([tile_a.underground[direction], tile_b.underground[direction]], set_numbers_equal(colour_a[side], colour_b[side]))

                            # Underground transition consistent
                            self.clauses += implies([tile_a.input_direction[direction],  tile_a.is_underground_in],  set_numbers_equal(tile_a.colour[side], colour_b[side]))
                            self.clauses += implies([tile_b.output_direction[direction], tile_b.is_underground_out], set_numbers_equal(colour_a[side], tile_b.colour[side]))
                    
                    tile_b = self.get_tile_instance_offset(x, y, *(direction_to_vec((direction + 1) % 4)), edge_mode)
                    if tile_b is not None:
                        for side in range(2):
                            # Complementary side has same colours
                            self.clauses += implies([tile_a.is_splitter[0], tile_a.colour_direction[direction]], set_numbers_equal(tile_a.colour[side], tile_b.colour[side]))
    
    def prevent_bad_insertion(self, edge_mode: EdgeModeType):
        for x in range(self.width):
            for y in range(self.height):
                tile_b = self.get_tile_instance(x, y)

                for inserter_type, inserter_offset in enumerate((1, 2)):
                    for direction in range(4):
                        inv_direction = (direction + 2) % 4

                        dx, dy = direction_to_vec(direction)
                        tile_a = self.get_tile_instance_offset(x, y, dx * -inserter_offset, dy * -inserter_offset, edge_mode)
                        tile_c = self.get_tile_instance_offset(x, y, dx * +inserter_offset, dy * +inserter_offset, edge_mode)

                        if tile_a is None or tile_c is None:
                            continue

                        # TODO remove restriction
                        self.clauses += implies([tile_b.is_inserter[inserter_type], tile_b.inserter_direction[direction]], [[tile_a.is_assembling_machine, tile_b.is_assembling_machine]])

                        consequences = []
                        for side in range(2):
                            # Amount taken from source is equal to amount dropped
                            consequences += invert_number(tile_a.flow_placed[side][inserter_type][direction], tile_c.flow_placed[side][inserter_type][inv_direction], self.allocate_variable)
                            
                            # Amount dropped is not negative
                            consequences.append([-tile_c.flow_placed[side][inserter_type][inv_direction][-1]])

                        # Some amount must be dropped
                        consequences.append([tile_a.flow_placed[0][inserter_type][direction][-1], tile_a.flow_placed[1][inserter_type][direction][-1]])

                        self.clauses += implies([tile_b.is_inserter[inserter_type], tile_b.inserter_direction[direction]], consequences)
                            
        
        for x in range(self.width):
            for y in range(self.height):
                tile_a = self.get_tile_instance(x, y)
                for inserter_type, inserter_offset in enumerate((1, 2)):
                    for direction in range(4):
                        dx, dy = direction_to_vec(direction)
                        tile_b = self.get_tile_instance_offset(x, y, dx * inserter_offset, dy * inserter_offset, edge_mode)

                        if tile_b is None:
                            continue

                        blocking_clauses = set_all_false(tile_a.flow_placed[0][inserter_type][direction] + tile_a.flow_placed[1][inserter_type][direction])

                        self.clauses += implies([-tile_b.is_inserter[inserter_type]], blocking_clauses)
                        self.clauses += implies([-tile_b.inserter_direction[direction], -tile_b.inserter_direction[(direction + 2) % 4]], blocking_clauses)

    def enforce_flow_summation(self, edge_mode: EdgeModeType):
        always_true = self.allocate_variable()
        self.clauses.append([always_true])

        for x in range(self.width):
            for y in range(self.height):
                tile = self.get_tile_instance(x, y)
                consequences = []
                for side in range(2):
                    source_numbers = [
                        [*tile.flow_in[side], -always_true]
                    ]
                    for inserter_type in range(2):
                        for direction in range(4):
                            source_numbers.append(tile.flow_placed[side][inserter_type][direction])
                    consequences += sum_numbers(source_numbers, [*tile.flow_out[side], -always_true], self.allocate_variable, True)
                self.clauses += implies([-tile.is_assembling_machine, -tile.is_splitter[0], -tile.is_splitter[1]], consequences)


        for x in range(self.width):
            for y in range(self.height):
                tile_a = self.get_tile_instance(x, y)
                for direction in range(4):
                    tile_b = self.get_tile_instance_offset(x, y, *direction_to_vec((direction + 1) % 4), edge_mode)

                    if tile_b is None:
                        continue

                    for side in range(2):
                        self.clauses += implies([tile_a.is_splitter[0]], add_numbers(tile_a.flow_in[side],  tile_b.flow_in[side],  tile_a.flow_splitter[side], self.allocate_variable))
                        self.clauses += implies([tile_a.is_splitter[0]], add_numbers(tile_a.flow_out[side], tile_b.flow_out[side], tile_a.flow_splitter[side], self.allocate_variable))

    def enforce_insertion_side(self):
        for x in range(self.width):
            for y in range(self.height):
                tile = self.get_tile_instance(x, y)

                for belt_direction in range(4):
                    consequences = []

                    for inserter_type in range(2):
                        for inserter_direction in range(4):
                            # Side to prevent insertion into
                            if belt_direction == (inserter_direction + 1) % 4:
                                side = 1
                            else:
                                side = 0

                            # If not negative then zero
                            consequences += implies([-tile.flow_placed[side][inserter_type][inserter_direction][-1]], set_all_false(tile.flow_placed[side][inserter_type][inserter_direction][:-1]))

                    self.clauses += implies([tile.is_belt, tile.output_direction[belt_direction]], consequences)
                    self.clauses += implies([tile.is_underground_in, tile.output_direction[belt_direction]], consequences)
                    self.clauses += implies([tile.is_underground_out, tile.output_direction[belt_direction]], consequences)

    def prevent_bad_flow(self, edge_mode: EdgeModeType):
        for x in range(self.width):
            for y in range(self.height):
                tile_a = self.get_tile_instance(x, y)

                no_flow_in = set_all_false(np.array(tile_a.flow_placed).reshape(-1).tolist())

                # TODO remove restriction
                self.clauses += implies([tile_a.is_splitter[0]], no_flow_in)
                self.clauses += implies([tile_a.is_splitter[1]], no_flow_in)
                self.clauses += implies([tile_a.is_empty],       no_flow_in)
                self.clauses += implies([tile_a.is_inserter[0]], no_flow_in)
                self.clauses += implies([tile_a.is_inserter[1]], no_flow_in)
                

                for direction in range(4):
                    dx, dy = direction_to_vec(direction)
                    tile_b = self.get_tile_instance_offset(x, y, dx, dy, edge_mode)

                    if tile_b is None:
                        continue
                    for side in range(2):
                        self.clauses += implies([tile_a.output_direction[direction]], set_numbers_equal(tile_a.flow_out[side], tile_b.flow_in[side]))

                        if direction % 2 == 0:
                            flow_ua = tile_a.flow_ux
                            flow_ub = tile_b.flow_ux
                        else:
                            flow_ua = tile_a.flow_uy
                            flow_ub = tile_b.flow_uy

                        self.clauses += implies([tile_a.underground[direction], tile_b.underground[direction]], set_numbers_equal(flow_ua[side], flow_ub[side]))

                        self.clauses += implies([tile_a.input_direction[direction], tile_a.is_underground_in], set_numbers_equal(tile_a.flow_out[side], flow_ub[side]))
                        self.clauses += implies([tile_b.output_direction[direction], tile_b.is_underground_out], set_numbers_equal(flow_ua[side], tile_b.flow_in[side]))
                        