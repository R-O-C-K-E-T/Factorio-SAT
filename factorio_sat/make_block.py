import argparse
import json
import sys

from . import optimisations
from . import solver
from .direction import Axis, Direction
from .template import EdgeMode, EdgeModeType
from .util import implies, increment_number, invert_components, set_all_false, set_number, set_numbers_equal


def ensure_loop_length(grid: solver.Grid, edge_mode: EdgeModeType):
    for y in range(grid.height):
        for x in range(grid.width):
            tile_a = grid.get_tile_instance(x, y)

            if x == 0 and y == 0:
                grid.clauses += [[-lit] for lit in tile_a.colour]
            else:
                grid.clauses.append(tile_a.colour)

            for direction in Direction:
                dx, dy = direction.vec
                tile_b = grid.get_tile_instance_offset(x, y, dx, dy, edge_mode)
                x1, y1 = x + dx, y + dy

                if tile_b is None:
                    continue

                if direction.axis == Axis.HORIZONTAL:
                    colour_a = tile_a.colour_ux
                    colour_b = tile_b.colour_ux
                else:
                    colour_a = tile_a.colour_uy
                    colour_b = tile_b.colour_uy

                if x1 == 0 and y1 == 0:
                    grid.clauses += implies([tile_a.output_direction[direction]], set_number(grid.colours - 1, tile_a.colour))
                else:
                    grid.clauses += implies([tile_a.output_direction[direction]], increment_number(tile_a.colour, tile_b.colour))

                grid.clauses += implies([tile_a.input_direction[direction], *invert_components(tile_a.output_direction)],
                                        increment_number(tile_a.colour, colour_b))

                grid.clauses += implies([*invert_components(tile_b.input_direction), tile_b.output_direction[direction]],
                                        set_numbers_equal(colour_a, tile_b.colour))
                grid.clauses += implies([tile_a.underground[direction], tile_b.underground[direction]], set_numbers_equal(colour_a, colour_b))


def prevent_parallel(grid: solver.Grid, edge_mode: EdgeModeType):
    for x in range(grid.width):
        for y in range(grid.height):
            tile_a = grid.get_tile_instance(x, y)
            for direction in (Direction.RIGHT, Direction.UP):
                tile_b = grid.get_tile_instance_offset(x, y, *direction.next.vec, edge_mode)
                if tile_b is None:
                    continue

                grid.clauses.append([-tile_a.underground[direction + 0], -tile_b.underground[direction + 0]])
                grid.clauses.append([-tile_a.underground[direction + 2], -tile_b.underground[direction + 0]])
                grid.clauses.append([-tile_a.underground[direction + 0], -tile_b.underground[direction + 2]])
                grid.clauses.append([-tile_a.underground[direction + 2], -tile_b.underground[direction + 2]])


def main():
    parser = argparse.ArgumentParser(description='Creates a stream of blocks of random belts')
    parser.add_argument('width', type=int, help='Block width')
    parser.add_argument('height', type=int, help='Block height')
    parser.add_argument('--tile', action='store_true', help='Makes output blocks tilable')
    parser.add_argument('--allow-empty', action='store_true', help='Allow empty tiles')
    parser.add_argument('--underground-length', type=int, default=4, help='Maximum length of underground section (excludes ends)')
    parser.add_argument('--no-parallel', action='store_true', help='Prevent parallel underground segments')
    parser.add_argument('--all', action='store_true', help='Produce all blocks')
    parser.add_argument('--label', type=str, help='Output blueprint label')
    parser.add_argument('--solver', type=str, default='Glucose3', help='Backend SAT solver to use')
    parser.add_argument('--single-loop', action='store_true', help='Prevent multiple loops')
    parser.add_argument('--output', type=argparse.FileType('w'), nargs='?', help='Output file, if no file provided then results are sent to standard out')
    args = parser.parse_args()

    if args.allow_empty and args.single_loop:
        raise RuntimeError('Incompatible options: allow-empty + single-loop')

    if args.underground_length < 0:
        raise RuntimeError('Underground length cannot be negative')
    if args.single_loop:
        grid = solver.Grid(args.width, args.height, args.width * args.height, args.underground_length)
    else:
        grid = solver.Grid(args.width, args.height, 1)

    edge_mode = EdgeMode.WRAP if args.tile else EdgeMode.NO_WRAP

    grid.prevent_intersection(edge_mode)
    grid.prevent_bad_undergrounding(edge_mode)
    if not args.tile:
        grid.block_belts_through_edges()
        grid.block_underground_through_edges()

    optimisations.prevent_small_loops(grid)

    if grid.underground_length > 0:
        grid.enforce_maximum_underground_length(edge_mode)
        optimisations.prevent_empty_along_underground(grid, edge_mode)

    if args.no_parallel:
        prevent_parallel(grid, edge_mode)

    if args.single_loop:
        ensure_loop_length(grid, edge_mode)

    for tile in grid.iterate_tiles():
        if not args.allow_empty:
            grid.clauses.append(tile.all_direction)  # Ban Empty

        if args.underground_length == 0:  # Ban underground
            grid.clauses += set_all_false(tile.underground)

        grid.clauses.append([-tile.is_splitter])  # Ban splitters

    if args.output is not None:
        with args.output:
            for solution in grid.itersolve(solver=args.solver, ignore_colour=True):
                json.dump(solution.tolist(), args.output)
                args.output.write('\n')
                if not args.all:
                    break
    else:
        for i, solution in enumerate(grid.itersolve(solver=args.solver, ignore_colour=True)):
            print(json.dumps(solution.tolist()))

            if i == 0:
                sys.stdout.flush()  # Push the first one out as fast a possible

            if not args.all:
                break


if __name__ == '__main__':
    main()
