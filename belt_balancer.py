import time, argparse, json
import numpy as np

from solver import Grid, Belt
from util import *
from network import get_exterior_colours, open_network


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
    input_colours, output_colours = get_exterior_colours(network)

    assert start_offset + len(input_colours) <= grid.height and end_offset + len(output_colours) <= grid.height

    for y in range(start_offset):
        grid.set_tile(0, y, None)

    for y in range(start_offset, start_offset + len(input_colours)):
        grid.set_tile(0, y, Belt(0, 0))
    
    for y in range(start_offset+len(input_colours), grid.height):
        grid.set_tile(0, y, None)
    

    for y in range(end_offset):
        grid.set_tile(grid.width-1, y, None)

    for y in range(end_offset, end_offset + len(output_colours)):
        grid.set_tile(grid.width-1, y, Belt(0, 0))

    for y in range(end_offset+len(output_colours), grid.height):
        grid.set_tile(grid.width-1, y, None)

    for colour in input_colours:
        variables = []
        for tile in [grid.get_tile_instance(0, y) for y in range(start_offset, start_offset + len(input_colours))]:
            variable = grid.allocate_variable()
            grid.clauses += implies([variable], set_number(colour, tile.colour))
            variables.append(variable)
        grid.clauses.append(variables)

    for colour in output_colours:
        variables = []
        for tile in [grid.get_tile_instance(grid.width - 1, y) for y in range(end_offset, end_offset + len(output_colours))]:
            variable = grid.allocate_variable()
            grid.clauses += implies([variable], set_number(colour, tile.colour))
            variables.append(variable)
        grid.clauses.append(variables)

def setup_balancer_ends(grid: Grid, network, aligned: bool):
    input_colours, output_colours = get_exterior_colours(network)
    assert len(input_colours) <= grid.height and len(output_colours) <= grid.height

    start_offsets = [grid.allocate_variable() for _ in range(grid.height - len(input_colours))]
    end_offsets   = [grid.allocate_variable() for _ in range(grid.height - len(output_colours))]

    for x, offsets, colour_set in zip((0, grid.width - 1), (start_offsets, end_offsets), (input_colours, output_colours)):
        if len(offsets) == 0:
            for y in range(grid.height):
                tile = grid.get_tile_instance(x, y)
                grid.clauses += [[tile.input_direction[0]], [tile.output_direction[0]], [-tile.is_splitter[0]], [-tile.is_splitter[1]]]
        else:
            grid.clauses += exactly_one_set(offsets)
            for dy, variable in enumerate(offsets):
                consequences = []
                for y in range(grid.height):
                    tile = grid.get_tile_instance(x, y)
                    if y in range(dy, dy + len(colour_set)):
                        consequences += [[tile.input_direction[0]], [tile.output_direction[0]], [-tile.is_splitter[0]], [-tile.is_splitter[1]]]
                    else:
                        consequences += set_number(0, tile.all_direction)
                grid.clauses += implies([variable], consequences)

        for colour in colour_set:
            variables = []
            for tile in [grid.get_tile_instance(x, y) for y in range(grid.height)]:
                variable = grid.allocate_variable()
                grid.clauses += implies([variable], set_number(colour, tile.colour) + [[tile.input_direction[0]], [tile.output_direction[0]]])
                variables.append(variable)
            grid.clauses.append(variables)

    if aligned:
        if len(input_colours) >= len(output_colours):
            for i, start_offset in enumerate(start_offsets):
                grid.clauses += implies([start_offset], [end_offsets[i:(i + 1 + len(input_colours) - len(output_colours))]])
        else:
            for i, end_offset in enumerate(end_offsets):
                grid.clauses += implies([end_offset], [start_offsets[i:(i + 1 + len(output_colours) - len(input_colours))]])

def create_balancer(network, width: int, height: int) -> Grid:
    assert width > 0 and height > 0

    network_input_colours = set()
    network_output_colours = set()

    for input_colours, output_colours in network:
        network_input_colours.update(input_colours)
        network_output_colours.update(output_colours)
    
    network_input_colours.discard(None)
    network_output_colours.discard(None)

    all_colours = network_input_colours | network_output_colours
    network_input_colours, network_output_colours = network_input_colours - network_output_colours, network_output_colours - network_input_colours

    grid = Grid(width, height, max(all_colours) + 1, TileTemplate({'node': 'one_hot ' + str(len(network))}))


    for colour in range(max(all_colours) + 1):
        if colour in all_colours:
            continue
        grid.prevent_colour(colour)
    
    grid.prevent_intersection((EDGE_MODE_IGNORE, EDGE_MODE_BLOCK))

    grid.prevent_bad_undergrounding(EDGE_MODE_BLOCK)
    grid.prevent_bad_colouring(EDGE_MODE_BLOCK)

    # There is exactly one splitter of each type
    for i in range(len(network)):
        variables = [grid.get_tile_instance(x, y).node[i] for x in range(grid.width) for y in range(grid.height)]
        grid.clauses += exactly_one_set_using_location(variables, grid.allocate_variable)
        #grid.clauses += exactly_one_set_using_tree(variables, grid.allocate_variable)
        #grid.clauses += exactly_one_set(variables)

    # Each splitter has one type
    for x in range(grid.width):
        for y in range(grid.height):
            tile = grid.get_tile_instance(x, y)
            #grid.clauses += no_two_set(tile.node)
            #grid.clauses.append([*tile.node, -tile.is_splitter[0]])

            grid.clauses += exactly_one_set([-tile.is_splitter[0], *tile.node])
            #grid.clauses += exactly_one_set_using_location([-tile.is_splitter[0], *tile.node], grid.allocate_variable)
            #grid.clauses += exactly_one_set_using_tree([-tile.is_splitter[0], *tile.node], grid.allocate_variable)

    network_input_colours, network_output_colours = get_exterior_colours(network)

    for i, (input_colours, output_colours) in enumerate(network):
        assert sum(colour is None for colour in input_colours + output_colours) <= 1

        for x00 in range(grid.width):
            for y00 in range(grid.height):
                tile00 = grid.get_tile_instance(x00, y00)

                #grid.clauses.append([tile00.is_splitter[0], -tile00.node[i]])

                if any(colour is None for colour in input_colours):
                    grid.clauses.append([-tile00.node[i], *tile00.output_direction])
                else:
                    grid.clauses.append([-tile00.node[i], *tile00.input_direction])

                for direction in range(4):
                    dx0, dy0 = direction_to_vec(direction)
                    dx1, dy1 = direction_to_vec((direction + 1) % 4)

                    x10, y10 = x00 + dx0, y00 + dy0
                    x01, y01 = x00 + dx1, y00 + dy1
                    x11, y11 = x00 + dx0 + dx1, y00 + dy0 + dy1
                    

                    precondition = [
                        tile00.node[i],       
                    ]
                    if any(colour is None for colour in input_colours):
                        precondition.append(tile00.output_direction[direction])
                    else:
                        precondition.append(tile00.input_direction[direction])
                    

                    if x11 < 0 or x11 >= grid.width or y11 < 0 or y11 >= grid.height:
                        grid.clauses.append(invert_components(precondition))
                        continue

                    tile01 = grid.get_tile_instance(x01, y01)
                    tile10 = grid.get_tile_instance(x10, y10)
                    tile11 = grid.get_tile_instance(x11, y11)


                    colour_a, colour_b = input_colours
                    if colour_a is None or colour_b is None:
                        colour = colour_a
                        if colour is None:
                            colour = colour_b
                            assert colour is not None
                        
                        grid.clauses += implies(precondition + [tile00.input_direction[direction]], set_number(colour, tile00.colour))
                        grid.clauses += implies(precondition + [tile01.input_direction[direction]], set_number(colour, tile01.colour))
                    else:
                        grid.clauses += implies(precondition, [[tile00.input_direction[direction]], [tile01.input_direction[direction]]])

                        if all(colour in network_input_colours for colour in input_colours):
                            grid.clauses += implies(precondition, set_number(colour_a, tile00.colour))
                            grid.clauses += implies(precondition, set_number(colour_b, tile01.colour))
                        else:
                            grid.clauses += implies(precondition, set_numbers(*input_colours, tile00.colour, tile01.colour))
                    
                    colour_a, colour_b = output_colours
                    if colour_a is None or colour_b is None:
                        colour = colour_a
                        if colour is None:
                            colour = colour_b
                            assert colour is not None

                        grid.clauses += implies(precondition + [tile00.output_direction[direction]], set_number(colour, tile10.colour))
                        grid.clauses += implies(precondition + [tile01.output_direction[direction]], set_number(colour, tile11.colour))
                    else:
                        grid.clauses += implies(precondition, [[tile00.output_direction[direction]], [tile01.output_direction[direction]]])

                        if all(colour in network_output_colours for colour in output_colours):
                            grid.clauses += implies(precondition, set_number(colour_a, tile10.colour))
                            grid.clauses += implies(precondition, set_number(colour_b, tile11.colour))
                        else:
                            grid.clauses += implies(precondition, set_numbers(*output_colours, tile10.colour, tile11.colour))

    return grid

def enforce_edge_splitters(grid: Grid, network):
    network_input_colours, network_output_colours = get_exterior_colours(network)

    # Any optimal splitter must have a splitter the same size that has both inputs/outputs connected directly to the overall input/outputs (I think)
    for i, (node_input_colours, node_output_colours) in enumerate(network):
        if all(colour in network_input_colours for colour in node_input_colours):
            grid.clauses.append([grid.get_tile_instance(1, y).node[i] for y in range(grid.height)])
            for y in range(grid.height):
                tile = grid.get_tile_instance(1, y)
                grid.clauses.append([-tile.node[i], tile.input_direction[0], tile.output_direction[0]])
        
        elif all(colour in network_output_colours for colour in node_output_colours):
            grid.clauses.append([grid.get_tile_instance(grid.width - 2, y).node[i] for y in range(grid.height)])
            for y in range(grid.height):
                tile = grid.get_tile_instance(grid.width - 2, y)
                grid.clauses.append([-tile.node[i], tile.input_direction[0], tile.output_direction[0]]) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a belt balancer from a splitter graph')
    parser.add_argument('network', type=argparse.FileType('r'), help='Splitter network')
    parser.add_argument('width', type=int, help='Belt balancer maximum width')
    parser.add_argument('height', type=int, help='Belt balancer maximum height')
    parser.add_argument('--edge-splitters', action='store_true', help='Enforce that any splitter that has both connections to the input/output of the balancer must be placed on the edge')
    parser.add_argument('--aligned', action='store_true', help='Enforces balancer input aligns with output')
    parser.add_argument('--underground-length', type=int, default=4, help='Sets the maximum length of underground section (excludes ends)')
    parser.add_argument('--all', action='store_true', help='Generate all belt balancers')
    parser.add_argument('--solver', type=str, default='Glucose3', help='Backend SAT solver to use')
    args = parser.parse_args()

    network = open_network(args.network)
    args.network.close()

    grid = create_balancer(network, args.width, args.height)

    if args.edge_splitters:
        enforce_edge_splitters(grid, network)

    #setup_balancer_ends_with_offsets(grid, network, 1, 0)#args.start_offset, args.end_offset)
    grid.set_maximum_underground_length(args.underground_length, EDGE_MODE_BLOCK)
    grid.prevent_empty_along_underground(args.underground_length, EDGE_MODE_BLOCK)

    setup_balancer_ends(grid, network, args.aligned)

    for solution in grid.itersolve(True, args.solver):
        print(json.dumps(solution.tolist()))
        if not args.all:
            break