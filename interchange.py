import argparse
import json

import belt_balancer
from direction import Direction
import optimisations
from solver import Grid
from template import EdgeMode
from util import invert_components, set_literal, set_number, set_numbers


def prevent_passing(grid: Grid):
    assert len(grid.get_tile_instance(0, 0).colour) == 1

    for direction in Direction:
        inv_direction = direction.reverse
        for block in grid.iterate_tile_blocks(direction.vec, 2, direction.next.vec, 2, EdgeMode.NO_WRAP):
            if (block == None).any():
                continue

            for colour_sign in (False, True):
                grid.clauses.append(invert_components([
                    set_literal(block[0, 0].colour[0], colour_sign),
                    set_literal(block[0, 1].colour[0], colour_sign),
                    set_literal(block[1, 0].colour[0], colour_sign),
                    set_literal(block[1, 1].colour[0], colour_sign),

                    block[0, 0].input_direction[direction],
                    block[0, 0].output_direction[direction],
                    block[0, 1].input_direction[direction],
                    block[0, 1].output_direction[direction],

                    block[1, 0].input_direction[inv_direction],
                    block[1, 0].output_direction[inv_direction],
                    block[1, 1].input_direction[inv_direction],
                    block[1, 1].output_direction[inv_direction],
                ]))


def prevent_awkward_underground_entry(grid: Grid):
    for direction in Direction:
        inv_direction = direction.reverse
        across_direction = direction.next
        for block in grid.iterate_tile_blocks(across_direction.vec, 3, direction.vec, 3, EdgeMode.NO_WRAP):
            block[0, 1:3] = None  # Unimportant tiles

            if (block == None).any():
                continue

            grid.clauses.append(invert_components([
                *invert_components(block[0, 0].all_direction),
                block[0, 0].underground[direction],

                -block[1, 0].underground[direction],

                *invert_components(block[1, 1].all_direction),

                # block[2,0].input_direction[direction],
                block[2, 0].output_direction[across_direction],

                # block[2,1].input_direction[across_direction],
                block[2, 1].output_direction[across_direction],

                block[2, 2].output_direction[inv_direction],

                block[1, 2].output_direction[inv_direction],
            ]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finds an interchange for building composite balancers')
    parser.add_argument('width', type=int, help='Interchange width')
    parser.add_argument('height', type=int, help='Combined balancer size')
    parser.add_argument('--underground-length', type=int, default=4, help='Sets the maximum length of underground section (excludes ends)')
    parser.add_argument('--alternating', action='store_true', help='Restrict output colours to an alternating pattern')
    parser.add_argument('--all', action='store_true', help='Generate all belt balancers')
    parser.add_argument('--solver', type=str, default='Glucose3', help='Backend SAT solver to use')
    parser.add_argument('--partial', type=argparse.FileType('r'), help='Partial interchange to base solution from')
    args = parser.parse_args()

    if args.height < 1:
        raise RuntimeError('Height not positive')

    if args.height % 2 == 1:
        raise RuntimeError('Height not multiple of 2')

    grid = Grid(args.width, args.height, 2, args.underground_length)

    # No splitters
    for tile in grid.iterate_tiles():
        grid.clauses.append([-tile.is_splitter])

    for y in range(0, grid.height):
        grid.clauses.append([grid.get_tile_instance(0, y).input_direction[0]])
        grid.clauses.append([grid.get_tile_instance(grid.width - 1, y).output_direction[0]])

    for y in range(0, grid.height // 2):
        grid.set_colour(0, y, 0)
    for y in range(grid.height // 2, grid.height):
        grid.set_colour(0, y, 1)

    if args.alternating:
        for y in range(grid.height):
            tile = grid.get_tile_instance(grid.width - 1, y)
            grid.clauses += set_number(y % 2, tile.colour)
    else:
        for y in range(0, grid.height, 2):
            tile0 = grid.get_tile_instance(grid.width - 1, y)
            tile1 = grid.get_tile_instance(grid.width - 1, y + 1)
            grid.clauses += set_numbers(0, 1, tile0.colour, tile1.colour)

    grid.block_underground_through_edges()
    grid.block_belts_through_edges((False, True))

    grid.prevent_bad_undergrounding(EdgeMode.NO_WRAP)
    grid.prevent_bad_colouring(EdgeMode.NO_WRAP)

    grid.prevent_intersection(EdgeMode.NO_WRAP)
    grid.enforce_maximum_underground_length(EdgeMode.NO_WRAP)

    optimisations.expand_underground(grid)
    optimisations.apply_generic_optimisations(grid)
    optimisations.break_vertical_symmetry(grid)
    optimisations.break_horisontal_symmetry(grid)
    optimisations.prevent_spirals(grid)

    prevent_passing(grid)

    prevent_awkward_underground_entry(grid)
    # for tile in grid.iterate_tiles():
    #     grid.clauses += implies(invert_components(tile.all_direction), set_all_false(tile.underground))

    if args.partial is not None:
        with args.partial:
            belt_balancer.set_nonempty_tiles(grid, args.partial.read())

    for solution in grid.itersolve(solver=args.solver, ignore_colour=True):
        print(json.dumps(solution.tolist()))
        if not args.all:
            break
