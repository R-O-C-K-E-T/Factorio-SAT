import argparse, json, sys

from solver import Grid, Belt
from util import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finds an interchange for building composite balancers')
    parser.add_argument('width', type=int, help='Interchange width')
    parser.add_argument('height', type=int, help='Combined balancer size')
    parser.add_argument('--underground-length', type=int, default=4, help='Sets the maximum length of underground section (excludes ends)')
    parser.add_argument('--alternating', action='store_true', help='Restrict output colours to an alternating pattern')
    parser.add_argument('--all', action='store_true', help='Generate all belt balancers')
    parser.add_argument('--solver', type=str, default='Glucose3', help='Backend SAT solver to use')
    args = parser.parse_args()

    if args.height < 1:
        raise RuntimeError('Height not positive')

    if args.height % 2 == 1:
        raise RuntimeError('Height not multiple of 2')

    grid = Grid(args.width, args.height, 2)


    # No splitters
    for tile in grid.iterate_tiles():
        grid.clauses += set_number(0, tile.is_splitter)


    for y in range(0, grid.height):
        grid.clauses.append([grid.get_tile_instance(0, y).input_direction[0]])
        grid.clauses.append([grid.get_tile_instance(grid.width - 1, y).output_direction[0]])

    for y in range(0, grid.height // 2):
        grid.set_colour(0, y, 0)
    for y in range(grid.height // 2, grid.height):
        grid.set_colour(0, y, 1)

    if args.alternating:
        for y in range(grid.height):
            tile = grid.get_tile_instance(grid.width-1, y)
            grid.clauses += set_number(y % 2, tile.colour)
    else:
        for y in range(0, grid.height, 2):
            tile0 = grid.get_tile_instance(grid.width-1, y)
            tile1 = grid.get_tile_instance(grid.width-1, y+1)
            grid.clauses += set_numbers(0, 1, tile0.colour, tile1.colour)

    grid.prevent_bad_undergrounding(EDGE_MODE_BLOCK)
    grid.prevent_bad_colouring(EDGE_MODE_BLOCK)

    grid.prevent_intersection((EDGE_MODE_IGNORE, EDGE_MODE_BLOCK))
    grid.set_maximum_underground_length(args.underground_length, EDGE_MODE_BLOCK)
    grid.prevent_empty_along_underground(args.underground_length, EDGE_MODE_BLOCK)
    # for tile in grid.iterate_tiles():
    #     grid.clauses += implies(invert_components(tile.all_direction), set_number(0, tile.underground))
    

    print(len(grid.clauses), file=sys.stderr)

    for solution in grid.itersolve(True, args.solver):
        print(json.dumps(solution.tolist()))
        if not args.all:
            break