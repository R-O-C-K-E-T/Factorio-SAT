import argparse
import json
import math

from . import belt_balancer
from . import optimisations
from .direction import Direction
from .cardinality import library_equals, quadratic_one
from .solver import Belt, Grid
from .template import OneHotTemplate
from .util import implies, invert_components, is_power_of_two, literals_different, set_all_false, set_numbers_equal


def create_balancer(width: int, height: int, underground_length: int) -> Grid:
    assert width > 0
    assert height > 0
    assert is_power_of_two(height)

    levels = int(math.log2(height)) - 2

    grid = Grid(width, height, 2**(height // 2), underground_length, {'level': OneHotTemplate(levels), 'level_primary': OneHotTemplate(levels)})

    grid.block_underground_through_edges()
    grid.prevent_bad_undergrounding()
    grid.prevent_bad_colouring()

    for tile in grid.iterate_tiles():
        grid.clauses += implies([tile.is_splitter], [tile.input_direction, tile.output_direction])
        grid.clauses += implies([tile.is_splitter_head], quadratic_one(tile.level))
        grid.clauses += implies([tile.is_splitter, -tile.is_splitter_head], quadratic_one(tile.level))
        grid.clauses += implies([-tile.is_splitter], set_all_false(tile.level))

        grid.clauses += implies([tile.is_splitter], set_numbers_equal(tile.level, tile.level_primary))
        grid.clauses += implies([-tile.is_splitter], set_all_false(tile.level_primary))

    for x in range(grid.width):
        for y in range(grid.height):
            tile00 = grid.get_tile_instance(x, y)

            for i in range(levels):
                grid.clauses += implies([tile00.level[i]], library_equals(tile00.colour, 2**i, grid.pool))

            for direction in Direction:
                dx0, dy0 = direction.vec
                dx1, dy1 = direction.next.vec

                precondition = [
                    tile00.is_splitter_head,
                    tile00.input_direction[direction],
                ]

                tile10 = grid.get_tile_instance_offset(x, y, dx0, dy0)
                tile01 = grid.get_tile_instance_offset(x, y, dx1, dy1)
                tile11 = grid.get_tile_instance_offset(x, y, dx0 + dx1, dy0 + dy1)

                if any(tile is None for tile in (tile00, tile10, tile01, tile11)):
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
        grid.set_tile(0, y, Belt(Direction.RIGHT, Direction.RIGHT))
        grid.set_colour(0, y, 2**(y // 2))

    for y in range(height):
        grid.set_tile(grid.width - 1, y, Belt(Direction.RIGHT, Direction.RIGHT))

    for y in range(0, height, 2):
        tile_a = grid.get_tile_instance(grid.width - 1, y)
        tile_b = grid.get_tile_instance(grid.width - 1, y + 1)
        for bit_a, bit_b in zip(tile_a.colour, tile_b.colour):
            grid.clauses += literals_different(bit_a, bit_b)

    checkerboard_tiles = [grid.get_tile_instance(x, y) for y in range(grid.height) for x in range(grid.width) if (x + y) % 2]
    for i in range(int(math.log2(height)) - 2):
        grid.clauses += library_equals([tile.level_primary[i] for tile in checkerboard_tiles], height // 2, grid.pool)

    return grid


def main():
    parser = argparse.ArgumentParser(
        description='Creates n to n belt balancers where n is a power of two. '
                    'Note that the outputs of this program do not include the first and last rows of splitters.')
    parser.add_argument('width', type=int, help='Belt balancer maximum width')
    parser.add_argument('size', type=int, help='Belt balancer size')
    parser.add_argument('--underground-length', type=int, default=4, help='Sets the maximum length of underground section (excludes ends)')
    parser.add_argument('--all', action='store_true', help='Generate all belt balancers')
    parser.add_argument('--solver', type=str, default='Glucose3', help='Backend SAT solver to use')
    parser.add_argument('--partial', type=argparse.FileType('r'), help='Partial balancer to base solution from')
    args = parser.parse_args()

    if args.underground_length == -1:
        args.underground_length = float('inf')

    grid = create_balancer(args.width, args.size, args.underground_length)

    grid.block_belts_through_edges((False, True))
    grid.prevent_intersection()

    grid.enforce_maximum_underground_length()

    optimisations.expand_underground(grid, min_x=1, max_x=grid.width - 2)
    optimisations.apply_generic_optimisations(grid)

    if args.partial is not None:
        with args.partial:
            belt_balancer.set_nonempty_tiles(grid, args.partial.read())

    for solution in grid.itersolve(solver=args.solver, ignore_colour=True):
        print(json.dumps(solution.tolist()))
        if not args.all:
            break


if __name__ == '__main__':
    main()
