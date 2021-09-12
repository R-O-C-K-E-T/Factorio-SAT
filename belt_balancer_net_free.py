from collections import defaultdict

from pysat.card import EncType
from cardinality import library_equals, quadratic_one
import argparse, json

from solver import Grid, Belt
from util import *
import belt_balancer


def create_balancer(width: int, height: int) -> Grid:
    assert width > 0
    assert height > 0
    assert is_power_of_two(height)

    levels = int(math.log2(height)) - 1


    grid = Grid(width, height, 2**(height // 2), TileTemplate({'level': f'one_hot {levels}', 'level_primary': f'one_hot {levels}'}))
    
    grid.prevent_bad_undergrounding(EDGE_MODE_BLOCK)
    grid.prevent_bad_colouring(EDGE_MODE_BLOCK)

    for tile in grid.iterate_tiles():
        for is_splitter in tile.is_splitter:
            grid.clauses += implies([is_splitter], [tile.input_direction, tile.output_direction])
        grid.clauses += implies([tile.is_splitter[0]], quadratic_one(tile.level))
        grid.clauses += implies([tile.is_splitter[1]], quadratic_one(tile.level))
        grid.clauses += implies(invert_components(tile.is_splitter), set_number(0, tile.level))

        grid.clauses += implies([tile.is_splitter[0]], set_numbers_equal(tile.level, tile.level_primary))
        grid.clauses += implies([-tile.is_splitter[0]], set_number(0, tile.level_primary))

    for x in range(grid.width):
        for y in range(grid.height):
            tile00 = grid.get_tile_instance(x, y)

            for i in range(levels):
                grid.clauses += implies([tile00.level[i]], library_equals(tile00.colour, 2**i, grid.pool))

            for direction in range(4):
                dx0, dy0 = direction_to_vec(direction)
                dx1, dy1 = direction_to_vec((direction + 1) % 4)

                precondition = [
                    tile00.is_splitter[0],    
                    tile00.input_direction[direction],   
                ]
                
                tile10 = grid.get_tile_instance_offset(x, y, dx0, dy0, EDGE_MODE_BLOCK)
                tile01 = grid.get_tile_instance_offset(x, y, dx1, dy1, EDGE_MODE_BLOCK)
                tile11 = grid.get_tile_instance_offset(x, y, dx0 + dx1, dy0 + dy1, EDGE_MODE_BLOCK)

                if any(tile == BLOCKED_TILE for tile in (tile00, tile10, tile01, tile11)):
                    grid.clauses.append(invert_components(precondition))
                    continue

                for in_bit0, in_bit1, out_bit0, out_bit1 in zip(tile00.colour, tile01.colour, tile10.colour, tile11.colour):
                    grid.clauses += implies(precondition, [
                        [-in_bit0, -in_bit1],

                        [-in_bit0, out_bit0],
                        [-in_bit0, out_bit1],

                        [-in_bit1, out_bit0],
                        [-in_bit1, out_bit1],

                        [in_bit0, in_bit1, -out_bit0],
                        [in_bit0, in_bit1, -out_bit1],
                    ])

                grid.clauses += implies(precondition, set_numbers_equal(tile00.level, tile01.level))
                    

    for y in range(height):
        grid.set_colour(0, y, 2**(y // 2))
        grid.set_tile(0, y, Belt(0, 0))

    for y in range(height):
        grid.set_colour(grid.width-1, y, grid.colours - 1)
        grid.set_tile(grid.width-1, y, Belt(0, 0))

    for y in range(height):
        tile = grid.get_tile_instance(grid.width - 2, y)
        grid.clauses.append(tile.is_splitter)
        grid.clauses.append([tile.level[-1]])

    return grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates n to n belt balancers where n is a power of two. Note that the outputs of this program do not include the first row of splitters.')
    parser.add_argument('width', type=int, help='Belt balancer maximum width')
    parser.add_argument('size', type=int, help='Belt balancer size')
    parser.add_argument('--underground-length', type=int, default=4, help='Sets the maximum length of underground section (excludes ends)')
    parser.add_argument('--all', action='store_true', help='Generate all belt balancers')
    parser.add_argument('--solver', type=str, default='Glucose3', help='Backend SAT solver to use')
    args = parser.parse_args()

    grid = create_balancer(args.width, args.size)
    grid.prevent_intersection((EDGE_MODE_IGNORE, EDGE_MODE_BLOCK))

    grid.set_maximum_underground_length(args.underground_length, EDGE_MODE_BLOCK)
    grid.prevent_empty_along_underground(args.underground_length, EDGE_MODE_BLOCK)

    grid.prevent_small_loops()
    belt_balancer.glue_splitters(grid)
    belt_balancer.prevent_belt_hooks(grid, EDGE_MODE_BLOCK)
    belt_balancer.expand_underground(grid, args.underground_length, min_x=1, max_x=grid.width-2)
    belt_balancer.prevent_mergeable_underground(grid, args.underground_length, EDGE_MODE_BLOCK)

    for i in range(int(math.log2(args.size)) - 1):
        grid.clauses += library_equals([tile.level_primary[i] for tile in grid.iterate_tiles()], args.size // 2, grid.pool)

    for solution in grid.itersolve(True, args.solver):
        print(json.dumps(solution.tolist()))
        if not args.all:
            break