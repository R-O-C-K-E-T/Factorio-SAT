from collections import defaultdict

from pysat.card import EncType
from cardinality import library_atleast, library_equals, quadratic_one, library_equals
import time, argparse, json, sys
import numpy as np

from solver import Grid, Belt
from util import *
from network import get_input_output_colours, open_network


def set_numbers(value_a: int, value_b: int, variables_a: List[VariableType], variables_b: List[VariableType]) -> ClauseList:
    # One set of variables is set to value_a, the other is set to value_b
    assert len(variables_a) == len(variables_b)
    total_bits = len(variables_a)
    assert value_a < (1 << total_bits)
    assert value_b < (1 << total_bits)

    clauses = []
    differences = []
    for var_a, var_b, bit_a, bit_b in zip(variables_a, variables_b, get_bits(value_a, total_bits), get_bits(value_b, total_bits)):
        if bit_a == bit_b:
            clauses.append([set_variable(var_a, bit_a)])
            clauses.append([set_variable(var_b, bit_a)])
        else:
            clauses += variables_different(var_a, var_b)
            differences.append((var_a, var_b, bit_a))

    if len(differences) != 0:
        var_a0, var_b0, bit_a0 = differences[0]
        #clauses += variables_different(var_a0, var_b0)
        for var_a1, var_b1, bit_a1 in differences[1:]:
            if bit_a0 == bit_a1: # Bits are correlated
                clauses += variables_same(var_a0, var_a1)
                #clauses += variables_different(var_a0, var_b1)
            else: # Anti-correlated
                clauses += variables_different(var_a0, var_a1)
                #clauses += variables_same(var_a0, var_b1)

    return clauses


def setup_balancer_ends_with_offsets(grid, network, start_offset: int, end_offset: int):
    (input_colour, input_count), (output_colour, output_count) = get_input_output_colours(network)

    assert start_offset + input_count <= grid.height and end_offset + output_count <= grid.height

    for y in range(start_offset):
        grid.set_tile(0, y, None)

    for y in range(start_offset, start_offset + input_count):
        grid.set_tile(0, y, Belt(0, 0))
        grid.set_colour(grid.width-1, y, input_colour)
    
    for y in range(start_offset+input_count, grid.height):
        grid.set_tile(0, y, None)
    

    for y in range(end_offset):
        grid.set_tile(grid.width-1, y, None)

    for y in range(end_offset, end_offset + output_count):
        grid.set_tile(grid.width-1, y, Belt(0, 0))
        grid.set_colour(grid.width-1, y, output_colour)

    for y in range(end_offset+output_count, grid.height):
        grid.set_tile(grid.width-1, y, None)

def setup_balancer_ends(grid: Grid, network, aligned: bool):
    (input_colour, input_count), (output_colour, output_count) = get_input_output_colours(network.elements())

    start_offsets = [grid.allocate_variable() for _ in range(grid.height - input_count)]
    end_offsets   = [grid.allocate_variable() for _ in range(grid.height - output_count)]

    for x, offsets, colour, count in zip((0, grid.width - 1), (start_offsets, end_offsets), (input_colour, output_colour), (input_count, output_count)):
        if len(offsets) == 0:
            for y in range(grid.height):
                tile = grid.get_tile_instance(x, y)
                grid.clauses += [[tile.input_direction[0]], [tile.output_direction[0]], [-tile.is_splitter[0]], [-tile.is_splitter[1]]]
                grid.clauses += set_number(colour, tile.colour)
        else:
            grid.clauses += quadratic_one(offsets)
            for dy, variable in enumerate(offsets):
                consequences = []
                for y in range(grid.height):
                    tile = grid.get_tile_instance(x, y)
                    if y in range(dy, dy + count):
                        consequences += [[tile.input_direction[0]], [tile.output_direction[0]], [-tile.is_splitter[0]], [-tile.is_splitter[1]]]
                        consequences += set_number(colour, tile.colour)
                    else:
                        consequences += set_number(0, tile.all_direction)
                grid.clauses += implies([variable], consequences)

    if aligned:
        if input_count >= output_count:
            for i, start_offset in enumerate(start_offsets):
                grid.clauses += implies([start_offset], [end_offsets[i:(i + 1 + input_count - output_count)]])
        else:
            for i, end_offset in enumerate(end_offsets):
                grid.clauses += implies([end_offset], [start_offsets[i:(i + 1 + output_count - input_count)]])

def deduplicate_network(network):
    key = lambda colour: -math.inf if colour is None else colour
    network = [(tuple(sorted(inputs, key=key)), tuple(sorted(outputs, key=key))) for inputs, outputs in network]
    return Counter(network)

def create_balancer(network, width: int, height: int) -> Grid:
    assert width > 0 and height > 0

    all_colours = set()
    for input_colours, output_colours in network:
        for colour in input_colours + output_colours:
            all_colours.add(colour)
    all_colours.discard(None)

    grid = Grid(width, height, max(all_colours) + 1, TileTemplate({'node': 'one_hot ' + str(len(network))}))
    for colour in range(max(all_colours) + 1):
        if colour in all_colours:
            continue
        grid.prevent_colour(colour)
    
    grid.prevent_bad_undergrounding(EDGE_MODE_BLOCK)
    grid.prevent_bad_colouring(EDGE_MODE_BLOCK)

    # There is exactly one splitter of each type
    for i, count in enumerate(network.values()):
        variables = [grid.get_tile_instance(x, y).node[i] for x in range(grid.width) for y in range(grid.height)]
        grid.clauses += library_equals(variables, count, grid.pool, EncType.kmtotalizer)


    # Each splitter has one type
    for x in range(grid.width):
        for y in range(grid.height):
            tile = grid.get_tile_instance(x, y)
            #grid.clauses += heule_one([-tile.is_splitter[0], *tile.node], grid.allocate_variable)
            grid.clauses += quadratic_one([-tile.is_splitter[0], *tile.node])
            #grid.clauses += library_equals([-tile.is_splitter[0], *tile.node], 1, grid.pool)

    for i, (input_colours, output_colours) in enumerate(network):
        assert sum(colour is None for colour in input_colours + output_colours) <= 1

        for x in range(grid.width):
            for y in range(grid.height):
                tile00 = grid.get_tile_instance(x, y)

                #grid.clauses.append([tile00.is_splitter[0], -tile00.node[i]])

                if any(colour is None for colour in input_colours):
                    assert not any(colour is None for colour in output_colours)
                    grid.clauses.append([-tile00.node[i], *tile00.output_direction])
                else:
                    grid.clauses.append([-tile00.node[i], *tile00.input_direction])

                for direction in range(4):
                    dx0, dy0 = direction_to_vec(direction)
                    dx1, dy1 = direction_to_vec((direction + 1) % 4)

                    precondition = [
                        tile00.node[i],       
                    ]
                    if any(colour is None for colour in input_colours):
                        assert not any(colour is None for colour in output_colours)
                        precondition.append(tile00.output_direction[direction])
                    else:
                        precondition.append(tile00.input_direction[direction])
                    
                    tile10 = grid.get_tile_instance_offset(x, y, dx0, dy0, EDGE_MODE_BLOCK)
                    tile01 = grid.get_tile_instance_offset(x, y, dx1, dy1, EDGE_MODE_BLOCK)
                    tile11 = grid.get_tile_instance_offset(x, y, dx0 + dx1, dy0 + dy1, EDGE_MODE_BLOCK)

                    if any(tile == BLOCKED_TILE for tile in (tile00, tile10, tile01, tile11)):
                        grid.clauses.append(invert_components(precondition))
                        continue

                    colour_a, colour_b = input_colours
                    if colour_a is None or colour_b is None:
                        colour = colour_a
                        if colour is None:
                            colour = colour_b
                            assert colour is not None
                        grid.clauses += implies(precondition, variables_different(tile00.input_direction[direction], tile01.input_direction[direction]))
                        grid.clauses += implies(precondition + [tile00.input_direction[direction]], set_number(colour, tile00.colour))
                        grid.clauses += implies(precondition + [tile01.input_direction[direction]], set_number(colour, tile01.colour))
                    else:
                        grid.clauses += implies(precondition, [[tile00.input_direction[direction]], [tile01.input_direction[direction]]])
                        grid.clauses += implies(precondition, set_numbers(*input_colours, tile00.colour, tile01.colour))
                    
                    colour_a, colour_b = output_colours
                    if colour_a is None or colour_b is None:
                        colour = colour_a
                        if colour is None:
                            colour = colour_b
                            assert colour is not None

                        grid.clauses += implies(precondition, variables_different(tile00.output_direction[direction], tile01.output_direction[direction]))
                        grid.clauses += implies(precondition + [tile00.output_direction[direction]], set_number(colour, tile10.colour))
                        grid.clauses += implies(precondition + [tile01.output_direction[direction]], set_number(colour, tile11.colour))
                    else:
                        grid.clauses += implies(precondition, [[tile00.output_direction[direction]], [tile01.output_direction[direction]]])
                        grid.clauses += implies(precondition, set_numbers(*output_colours, tile10.colour, tile11.colour))
    return grid

def enforce_edge_splitters(grid: Grid, network):
    (network_input_colour, _), (network_output_colour, _) = get_input_output_colours(network.elements())

    recirculate_input = 0
    recirculate_output = 0
    for inputs, outputs in network.elements():
        for colour in inputs:
            if colour == network_output_colour:
                recirculate_output += 1
        for colour in outputs:
            if colour == network_input_colour:
                recirculate_input += 1

    input_splitters = [(i, count) for i, ((input_colours, _), count) in enumerate(network.items()) if all(colour == network_input_colour for colour in input_colours)]
    if recirculate_input == 0:
        for i, count in input_splitters:
            variables = [grid.get_tile_instance(1, y).node[i] for y in range(grid.height)]
            grid.clauses += library_equals(variables, count, grid.pool, EncType.kmtotalizer)
            for y in range(grid.height):
                tile = grid.get_tile_instance(1, y)
                grid.clauses += implies([tile.node[i]], [[tile.input_direction[0], tile.output_direction[0]]])
    else:
        edge_splitter_min = sum(count for _, count in input_splitters) - recirculate_input
        if edge_splitter_min > 0:
            variables = [grid.get_tile_instance(1, y).node[i] for y in range(grid.height) for i, _ in input_splitters]
            grid.clauses += library_atleast(variables, edge_splitter_min, grid.pool)

    output_splitters = [(i, count) for i, ((_, output_colours), count) in enumerate(network.items()) if all(colour == network_output_colour for colour in output_colours)]
    if recirculate_output == 0:
        for i, count in output_splitters:
            variables = [grid.get_tile_instance(grid.width - 2, y).node[i] for y in range(grid.height)]
            grid.clauses += library_equals(variables, count, grid.pool, EncType.kmtotalizer)

            # grid.clauses.append([grid.get_tile_instance(grid.width - 2, y).node[i] for y in range(grid.height)])
            for y in range(grid.height):
                tile = grid.get_tile_instance(grid.width - 2, y)
                grid.clauses += implies([tile.node[i]], [[tile.input_direction[0], tile.output_direction[0]]])
    else:
        edge_splitter_min = sum(count for _, count in output_splitters) - recirculate_output
        if edge_splitter_min > 0:
            variables = [grid.get_tile_instance(grid.width - 2, y).node[i] for y in range(grid.height) for i, _ in output_splitters]
            grid.clauses += library_atleast(variables, edge_splitter_min, grid.pool)

def prevent_double_edge_belts(grid: Grid):
    for x in (1, max(grid.width - 2, 1)):
        for y in range(grid.height - 1):
            tile_a = grid.get_tile_instance(x, y)
            tile_b = grid.get_tile_instance(x, y + 1)

            grid.clauses.append([
                -tile_a.input_direction[0], -tile_a.output_direction[0], *tile_a.is_splitter,
                -tile_b.input_direction[0], -tile_b.output_direction[0], *tile_b.is_splitter,
            ])

def glue_splitters(grid: Grid):
    for x in range(grid.width):
        for y in range(grid.height):
            tile = grid.get_tile_instance(x, y)
            for direction in range(4):
                if direction == 0 and (x == 1 or x == grid.width - 2): # Ignore edge splitters
                    continue
                
                dx0, dy0 = direction_to_vec(direction)
                dx1, dy1 = direction_to_vec((direction + 1) % 4)

                tile_in0 = grid.get_tile_instance_offset(x, y, -dx0, -dy0, EDGE_MODE_BLOCK)
                tile_in1 = grid.get_tile_instance_offset(x, y, -dx0 + dx1, -dy0 + dy1, EDGE_MODE_BLOCK)

                if tile_in0 == BLOCKED_TILE or tile_in1 == BLOCKED_TILE:
                    continue

                grid.clauses.append([
                    -tile.is_splitter[0],
                    -tile.input_direction[direction],

                    -tile_in0.input_direction[direction],
                    -tile_in0.output_direction[direction],
                    *tile_in0.is_splitter,

                    -tile_in1.input_direction[direction],
                    -tile_in1.output_direction[direction],
                    *tile_in1.is_splitter,
                ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a belt balancer from a splitter graph')
    parser.add_argument('network', type=argparse.FileType('r'), help='Splitter network')
    parser.add_argument('width', type=int, help='Belt balancer maximum width')
    parser.add_argument('height', type=int, help='Belt balancer maximum height')
    parser.add_argument('--edge-splitters', action='store_true', help='Enforce that any splitter that has both connections to the input/output of the balancer must be placed on the edge')
    parser.add_argument('--edge-belts', action='store_true', help='Prevents two side by side belts at the inputs/outputs of a balancer (weaker version of --edge-splitters)')
    parser.add_argument('--glue-splitters', action='store_true', help='Prevents a configuration where a splitter has two straight input belts. Effectively pushing all splitters as far back as possible')
    parser.add_argument('--aligned', action='store_true', help='Enforces balancer input aligns with output')
    parser.add_argument('--underground-length', type=int, default=4, help='Sets the maximum length of underground section (excludes ends)')
    parser.add_argument('--all', action='store_true', help='Generate all belt balancers')
    parser.add_argument('--solver', type=str, default='Glucose3', help='Backend SAT solver to use')
    args = parser.parse_args()

    if args.edge_splitters and args.edge_belts:
        raise RuntimeError('--edge-splitters and --edge-belts are mutually exclusive')

    network = open_network(args.network)
    args.network.close()

    network = deduplicate_network(network)

    grid = create_balancer(network, args.width, args.height)
    grid.prevent_intersection((EDGE_MODE_IGNORE, EDGE_MODE_BLOCK))

    if args.edge_splitters:
        enforce_edge_splitters(grid, network)
    if args.edge_belts:
        prevent_double_edge_belts(grid)
    if args.glue_splitters:
        glue_splitters(grid)

    #setup_balancer_ends_with_offsets(grid, network, 1, 0)#args.start_offset, args.end_offset)
    grid.set_maximum_underground_length(args.underground_length, EDGE_MODE_BLOCK)
    grid.prevent_empty_along_underground(args.underground_length, EDGE_MODE_BLOCK)

    setup_balancer_ends(grid, network, args.aligned)

    # for clause in grid.clauses:
    #     print(clause, sep=' ')

    for solution in grid.itersolve(True, args.solver):
        print(json.dumps(solution.tolist()))
        if not args.all:
            break