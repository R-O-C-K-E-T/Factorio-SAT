from pysat.card import EncType
from cardinality import library_atleast, library_equals, quadratic_one, library_equals
import argparse, json

import optimisations
from solver import Grid, Belt
from template import OneHotTemplate, EdgeMode, TileResult
from util import *
from network import get_input_output_colours, open_network, deduplicate_network


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
            for dy, literal in enumerate(offsets):
                consequences = []
                for y in range(grid.height):
                    tile = grid.get_tile_instance(x, y)
                    if y in range(dy, dy + count):
                        consequences += [[tile.input_direction[0]], [tile.output_direction[0]], [-tile.is_splitter[0]], [-tile.is_splitter[1]]]
                        consequences += set_number(colour, tile.colour)
                    else:
                        consequences += set_all_false(tile.all_direction)
                grid.clauses += implies([literal], consequences)

    if aligned:
        if input_count >= output_count:
            for i, start_offset in enumerate(start_offsets):
                grid.clauses += implies([start_offset], [end_offsets[i:(i + 1 + input_count - output_count)]])
        else:
            for i, end_offset in enumerate(end_offsets):
                grid.clauses += implies([end_offset], [start_offsets[i:(i + 1 + output_count - input_count)]])

def create_balancer(network, width: int, height: int) -> Grid:
    assert width > 0 and height > 0

    all_colours = set()
    for input_colours, output_colours in network:
        for colour in input_colours + output_colours:
            all_colours.add(colour)
    all_colours.discard(None)

    grid = Grid(width, height, max(all_colours) + 1, {'node': OneHotTemplate(len(network))})
    for colour in range(max(all_colours) + 1):
        if colour in all_colours:
            continue
        grid.prevent_colour(colour)
    
    grid.prevent_bad_undergrounding(EdgeMode.BLOCK)
    grid.prevent_bad_colouring(EdgeMode.BLOCK)

    # There is exactly one splitter of each type
    for i, count in enumerate(network.values()):
        literals = [grid.get_tile_instance(x, y).node[i] for x in range(grid.width) for y in range(grid.height)]
        grid.clauses += library_equals(literals, count, grid.pool, EncType.kmtotalizer)


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
                    
                    tile10 = grid.get_tile_instance_offset(x, y, dx0, dy0, EdgeMode.BLOCK)
                    tile01 = grid.get_tile_instance_offset(x, y, dx1, dy1, EdgeMode.BLOCK)
                    tile11 = grid.get_tile_instance_offset(x, y, dx0 + dx1, dy0 + dy1, EdgeMode.BLOCK)

                    if any(tile == TileResult.BLOCKED for tile in (tile00, tile10, tile01, tile11)):
                        grid.clauses.append(invert_components(precondition))
                        continue

                    colour_a, colour_b = input_colours
                    if colour_a is None or colour_b is None:
                        colour = colour_a
                        if colour is None:
                            colour = colour_b
                            assert colour is not None
                        grid.clauses += implies(precondition, literals_different(tile00.input_direction[direction], tile01.input_direction[direction]))
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

                        grid.clauses += implies(precondition, literals_different(tile00.output_direction[direction], tile01.output_direction[direction]))
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
            literals = [grid.get_tile_instance(1, y).node[i] for y in range(grid.height)]
            grid.clauses += library_equals(literals, count, grid.pool, EncType.kmtotalizer)
            for y in range(grid.height):
                tile = grid.get_tile_instance(1, y)
                grid.clauses += implies([tile.node[i]], [[tile.input_direction[0], tile.output_direction[0]]])
    else:
        edge_splitter_min = sum(count for _, count in input_splitters) - recirculate_input
        if edge_splitter_min > 0:
            literals = [grid.get_tile_instance(1, y).node[i] for y in range(grid.height) for i, _ in input_splitters]
            grid.clauses += library_atleast(literals, edge_splitter_min, grid.pool)

    output_splitters = [(i, count) for i, ((_, output_colours), count) in enumerate(network.items()) if all(colour == network_output_colour for colour in output_colours)]
    if recirculate_output == 0:
        for i, count in output_splitters:
            literals = [grid.get_tile_instance(grid.width - 2, y).node[i] for y in range(grid.height)]
            grid.clauses += library_equals(literals, count, grid.pool, EncType.kmtotalizer)

            # grid.clauses.append([grid.get_tile_instance(grid.width - 2, y).node[i] for y in range(grid.height)])
            for y in range(grid.height):
                tile = grid.get_tile_instance(grid.width - 2, y)
                grid.clauses += implies([tile.node[i]], [[tile.input_direction[0], tile.output_direction[0]]])
    else:
        edge_splitter_min = sum(count for _, count in output_splitters) - recirculate_output
        if edge_splitter_min > 0:
            literals = [grid.get_tile_instance(grid.width - 2, y).node[i] for y in range(grid.height) for i, _ in output_splitters]
            grid.clauses += library_atleast(literals, edge_splitter_min, grid.pool)

def prevent_double_edge_belts(grid: Grid):
    for x in (1, max(grid.width - 2, 1)):
        for y in range(grid.height - 1):
            tile_a = grid.get_tile_instance(x, y)
            tile_b = grid.get_tile_instance(x, y + 1)

            grid.clauses.append([
                -tile_a.input_direction[0], -tile_a.output_direction[0], *tile_a.is_splitter,
                -tile_b.input_direction[0], -tile_b.output_direction[0], *tile_b.is_splitter,
            ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a belt balancer from a splitter graph')
    parser.add_argument('network', type=argparse.FileType('r'), help='Splitter network')
    parser.add_argument('width', type=int, help='Belt balancer maximum width')
    parser.add_argument('height', type=int, help='Belt balancer maximum height')
    parser.add_argument('--edge-splitters', action='store_true', help='Enforce that any splitter that has both connections to the input/output of the balancer must be placed on the edge')
    parser.add_argument('--edge-belts', action='store_true', help='Prevents two side by side belts at the inputs/outputs of a balancer (weaker version of --edge-splitters)')
    parser.add_argument('--glue-splitters', action='store_true', help='Prevents a configuration where a splitter has two straight input belts. Effectively pushing all splitters as far back as possible')
    parser.add_argument('--expand-underground', action='store_true', help='Ensures that underground belts are expanded if possible')
    parser.add_argument('--prevent-mergeable-underground', action='store_true', help='Prevent underground segments that can be merged')
    parser.add_argument('--prevent-bad-patterns', action='store_true', help='Prevents patterns of belts that are not helpful, i.e. always substitutable for a simpler configuration.')
    parser.add_argument('--break-symmetry', action='store_true', help='Prevents vertical reflections being considered')
    parser.add_argument('--fast', action='store_true', help='Enables all speed improving options')
    parser.add_argument('--aligned', action='store_true', help='Enforces balancer input aligns with output')
    parser.add_argument('--underground-length', type=int, default=4, help='Sets the maximum length of underground section (excludes ends)')
    parser.add_argument('--all', action='store_true', help='Generate all belt balancers')
    parser.add_argument('--solver', type=str, default='Glucose3', help='Backend SAT solver to use')
    args = parser.parse_args()

    if args.underground_length == -1:
        args.underground_length = float('inf')

    if args.edge_splitters and args.edge_belts:
        raise RuntimeError('--edge-splitters and --edge-belts are mutually exclusive')

    network = open_network(args.network)
    args.network.close()

    network = deduplicate_network(network)

    grid = create_balancer(network, args.width, args.height)
    grid.prevent_intersection((EdgeMode.IGNORE, EdgeMode.BLOCK))

    if args.edge_splitters or args.fast:
        enforce_edge_splitters(grid, network)
    if args.edge_belts:
        prevent_double_edge_belts(grid)
    if args.glue_splitters or args.fast:
        optimisations.glue_splitters(grid)
    if args.expand_underground or args.fast:
        optimisations.expand_underground(grid, args.underground_length, min_x=1, max_x=grid.width-2)
    if args.prevent_mergeable_underground or args.fast:
        optimisations.prevent_mergeable_underground(grid, args.underground_length, EdgeMode.BLOCK)
    if args.break_symmetry:
        optimisations.break_vertical_symmetry(grid)
    if args.prevent_bad_patterns or args.fast:
        optimisations.prevent_belt_hooks(grid, EdgeMode.BLOCK)
        optimisations.prevent_semicircles(grid, EdgeMode.BLOCK)
        optimisations.prevent_small_loops(grid)

    #setup_balancer_ends_with_offsets(grid, network, 1, 0)#args.start_offset, args.end_offset)
    grid.set_maximum_underground_length(args.underground_length, EdgeMode.BLOCK)
    optimisations.prevent_empty_along_underground(grid, args.underground_length, EdgeMode.BLOCK)

    setup_balancer_ends(grid, network, args.aligned)

    # for clause in grid.clauses:
    #     print(clause, sep=' ')

    for solution in grid.itersolve(True, args.solver):
        print(json.dumps(solution.tolist()))
        if not args.all:
            break