from typing import List, Optional

import numpy as np

from .direction import Direction
from .solver import Grid
from .template import EdgeMode, EdgeModeType
from .util import LiteralType, break_symmetry, implies, invert_components, set_literal, set_not_number


def prevent_empty_along_underground(grid: Grid, edge_mode: EdgeModeType):
    underground_length = min(grid.underground_length, max(grid.width, grid.height) - 2)

    for direction in Direction:
        dx, dy = direction.vec
        for x in range(grid.width):
            for y in range(grid.height):
                tiles = [grid.get_tile_instance(x, y)]
                for i in range(1, underground_length + 2):
                    new_tile = grid.get_tile_instance_offset(x, y, dx * i, dy * i, edge_mode)
                    if new_tile is None:
                        break

                    tiles.append(new_tile)

                    start, *middle, end = tiles
                    if len(middle) > 0:
                        clause = [
                            -start.is_underground_in,
                            -start.input_direction[direction],
                        ]

                        for tile in middle:
                            clause.append(-tile.is_empty)

                        clause += [
                            -end.is_underground_out,
                            -end.output_direction[direction],
                        ]

                        grid.clauses.append(clause)


def prevent_small_loops(grid: Grid):
    for x in range(grid.width - 1):
        for y in range(grid.height - 1):
            tile00 = grid.get_tile_instance(x + 0, y + 0)
            tile01 = grid.get_tile_instance(x + 0, y + 1)
            tile10 = grid.get_tile_instance(x + 1, y + 0)
            tile11 = grid.get_tile_instance(x + 1, y + 1)

            grid.clauses.append([
                -tile00.input_direction[2],
                -tile00.output_direction[3],

                -tile01.input_direction[3],
                -tile01.output_direction[0],

                -tile11.input_direction[0],
                -tile11.output_direction[1],

                -tile10.input_direction[1],
                -tile10.output_direction[2],
            ])

            grid.clauses.append([
                -tile00.input_direction[1],
                -tile00.output_direction[0],

                -tile10.input_direction[0],
                -tile10.output_direction[3],

                -tile11.input_direction[3],
                -tile11.output_direction[2],

                -tile01.input_direction[2],
                -tile01.output_direction[1],
            ])


def glue_splitters(grid: Grid):
    for x in range(grid.width):
        for y in range(grid.height):
            tile = grid.get_tile_instance(x, y)
            for direction in Direction:
                if direction == Direction.RIGHT and (x == 1 or x == grid.width - 2):  # Ignore edge splitters, TODO make this more generic
                    continue

                dx0, dy0 = direction.vec
                dx1, dy1 = direction.next.vec

                tile_in0 = grid.get_tile_instance_offset(x, y, -dx0, -dy0, EdgeMode.NO_WRAP)
                tile_in1 = grid.get_tile_instance_offset(x, y, -dx0 + dx1, -dy0 + dy1, EdgeMode.NO_WRAP)

                if tile_in0 is None or tile_in1 is None:
                    continue

                grid.clauses.append([
                    -tile.is_splitter_head,
                    -tile.input_direction[direction],

                    -tile_in0.input_direction[direction],
                    -tile_in0.output_direction[direction],
                    -tile_in0.is_belt,

                    -tile_in1.input_direction[direction],
                    -tile_in1.output_direction[direction],
                    -tile_in1.is_belt,
                ])


def shrink_underground(grid: Grid, edge_mode: EdgeModeType):
    # Has some correctness problems
    for x in range(grid.width):
        for y in range(grid.height):
            tile_a = grid.get_tile_instance(x, y)
            for direction in Direction:
                dx0, dy0 = direction.vec
                tile_b = grid.get_tile_instance_offset(x, y, dx0, dy0, edge_mode)
                if tile_b is None:
                    continue

                grid.clauses += implies([
                    -tile_a.underground[direction],
                    tile_b.underground[direction],
                ], [[-tile_b.is_empty]])

                grid.clauses += implies([
                    tile_a.underground[direction],
                    -tile_b.underground[direction],
                ], [[-tile_a.is_empty]])


def expand_underground_infinite(grid: Grid, min_x: int = 0, min_y: int = 0, max_x: Optional[int] = None, max_y: Optional[int] = None):
    if max_x is None:
        max_x = grid.width - 1
    if max_y is None:
        max_y = grid.height - 1

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            for direction in Direction:
                dx, dy = direction.vec

                far_x = x + dx * 1
                far_y = y + dy * 1
                if far_x > max_x or far_y > max_y or far_x < min_x or far_y < min_y:
                    continue

                tiles = [grid.get_tile_instance_offset(x, y, dx * i, dy * i, EdgeMode.NO_WRAP) for i in range(2)]
                assert all(tile is not None for tile in tiles)

                # Prevent: Belt, Input Underground
                grid.clauses.append([
                    -tiles[0].input_direction[direction],
                    -tiles[0].output_direction[direction],
                    -tiles[0].is_belt,

                    -tiles[1].is_underground_in,
                    -tiles[1].input_direction[direction],
                ])

                # Prevent: Output Underground, Belt
                grid.clauses.append([
                    -tiles[0].is_underground_out,
                    -tiles[1].output_direction[direction],

                    -tiles[1].input_direction[direction],
                    -tiles[1].output_direction[direction],
                    -tiles[1].is_belt,
                ])


def expand_underground(grid: Grid, min_x: int = 0, min_y: int = 0, max_x: Optional[int] = None, max_y: Optional[int] = None):
    if grid.underground_length == float('inf'):
        expand_underground_infinite(grid, min_x, min_y, max_x, max_y)
        return

    if max_x is None:
        max_x = grid.width - 1
    if max_y is None:
        max_y = grid.height - 1

    for underground_length in range(2, grid.underground_length + 1):
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for direction in Direction:
                    dx, dy = direction.vec

                    far_x = x + dx * (underground_length + 1)
                    far_y = y + dy * (underground_length + 1)
                    if far_x > max_x or far_y > max_y or far_x < min_x or far_y < min_y:
                        continue

                    tiles = [grid.get_tile_instance_offset(x, y, dx * i, dy * i, EdgeMode.NO_WRAP) for i in range(underground_length + 2)]
                    assert all(tile is not None for tile in tiles)

                    # BI--O
                    grid.clauses.append([
                        -tiles[0].input_direction[direction],
                        -tiles[0].output_direction[direction],
                        -tiles[0].is_belt,

                        tiles[1].underground[direction],

                        *(-tile.underground[direction] for tile in tiles[2:-1]),

                        tiles[-1].underground[direction],
                    ])

                    # I--OB
                    grid.clauses.append([
                        tiles[0].underground[direction],

                        *(-tile.underground[direction] for tile in tiles[1:-2]),

                        tiles[-2].underground[direction],

                        -tiles[-1].input_direction[direction],
                        -tiles[-1].output_direction[direction],
                        -tiles[-1].is_belt,
                    ])


def prevent_belt_hooks(grid: Grid, edge_mode: EdgeModeType):
    for x in range(grid.width):
        for y in range(grid.height):
            tile00 = grid.get_tile_instance(x, y)
            for direction in Direction:
                for tangent in (direction.next, direction.prev):
                    inverse = direction.reverse

                    dx0, dy0 = direction.vec
                    dx1, dy1 = tangent.vec
                    tile01 = grid.get_tile_instance_offset(x, y, dx0, dy0, edge_mode)
                    tile10 = grid.get_tile_instance_offset(x, y, dx1, dy1, edge_mode)
                    tile11 = grid.get_tile_instance_offset(x, y, dx0 + dx1, dy0 + dy1, edge_mode)

                    if any(tile is None for tile in (tile01, tile10, tile11)):
                        continue

                    for in_direction in (direction, tangent):
                        for out_direction in (tangent, inverse):
                            grid.clauses.append([
                                -tile00.is_belt,
                                -tile00.input_direction[in_direction],
                                -tile00.output_direction[direction],

                                -tile01.output_direction[tangent],

                                -tile11.output_direction[inverse],

                                -tile10.is_belt,
                                -tile10.output_direction[out_direction],
                            ])


def get_mergeable_underground_variations(underground_length: int):
    if underground_length < 4:
        return

    for offset in range(1, underground_length - 2):
        yield [
            False,  # Start 1
            *([True] * offset),
            False,  # End 1
            False,  # Start 2
            *([True] * (underground_length - offset - 2)),
            False,  # End 2
        ]


def prevent_mergeable_underground(grid: Grid, edge_mode: EdgeModeType):
    max_underground_length = min(grid.underground_length, max(grid.width, grid.height) - 2)

    for underground_length in range(4, max_underground_length + 1):
        for x in range(grid.width):
            for y in range(grid.height):
                for direction in Direction:
                    dx, dy = direction.vec
                    tiles = [grid.get_tile_instance_offset(x, y, dx * i, dy * i, edge_mode) for i in range(underground_length + 2)]
                    if any(tile is None for tile in tiles):
                        continue

                    for variation in get_mergeable_underground_variations(underground_length):
                        grid.clauses.append([-set_literal(tile.underground[direction], is_underground) for tile, is_underground in zip(tiles, variation)])


def prevent_semicircles(grid: Grid, edge_mode: EdgeModeType):
    for x in range(grid.width):
        for y in range(grid.height):
            for direction in Direction:
                for tangent in (direction.next, direction.prev):
                    inverse_tangent = tangent.reverse
                    dx0, dy0 = direction.vec
                    dx1, dy1 = tangent.vec
                    func = np.frompyfunc(lambda j, i: grid.get_tile_instance_offset(x, y, dx0 * i + dx1 * j, dy0 * i + dy1 * j, edge_mode), 2, 1)
                    tiles = func(*np.ogrid[0:2, 0:3])

                    if (tiles == None).any():
                        continue

                    for in_direction in (direction, tangent):
                        for out_direction in (inverse_tangent, direction):
                            grid.clauses.append([
                                -tiles[0, 0].input_direction[in_direction],
                                *([tiles[0, 0].is_splitter] if tangent == in_direction else []),

                                *tiles[0, 1].all_direction,

                                -tiles[0, 2].input_direction[inverse_tangent],
                                -tiles[0, 2].output_direction[out_direction],
                                *([tiles[0, 2].is_splitter] if inverse_tangent == out_direction else []),

                                -tiles[1, 0].input_direction[tangent],

                                -tiles[1, 1].input_direction[direction],
                                tiles[1, 1].is_splitter,

                                -tiles[1, 2].input_direction[direction],
                            ])


def prevent_underground_hook(grid: Grid, edge_mode: EdgeModeType):
    # TODO Make sound
    for x in range(grid.width):
        for y in range(grid.height):
            tile_underground_transition = grid.get_tile_instance(x, y)
            for direction in Direction:
                dx0, dy0 = direction.vec

                tile_empty = grid.get_tile_instance_offset(x, y, dx0, dy0, edge_mode)
                if tile_empty is None:
                    continue

                tile_a = grid.get_tile_instance_offset(x, y, -dx0, -dy0, edge_mode)
                if tile_a is None:
                    continue

                inv_direction = direction.reverse

                for tangent in (direction.next, direction.prev):
                    inv_tangent = tangent.reverse
                    dx1, dy1 = tangent.vec

                    tile_b = grid.get_tile_instance_offset(x, y, -dx1 - dx0, -dy1 - dy0, edge_mode)
                    if tile_b is None:
                        continue
                    tile_c = grid.get_tile_instance_offset(x, y, -dx1, -dy1, edge_mode)
                    if tile_c is None:
                        continue

                    for in_direction in (inv_direction, tangent):
                        grid.clauses.append([
                            *tile_empty.all_direction,
                            -tile_empty.underground[direction],

                            tile_underground_transition.underground[direction],

                            # -tile_a.output_direction[direction],
                            -tile_a.input_direction[tangent],

                            # -tile_b.output_direction[tangent],
                            -tile_b.input_direction[inv_direction],

                            # -tile_c.output_direction[inv_direction],
                            -tile_c.input_direction[in_direction],
                            tile_c.is_splitter,
                        ])

                    for out_direction in (direction, inv_tangent):
                        grid.clauses.append([
                            *tile_empty.all_direction,
                            -tile_empty.underground[inv_direction],

                            tile_underground_transition.underground[inv_direction],

                            # -tile_a.input_direction[inv_direction],
                            -tile_a.output_direction[inv_tangent],

                            # -tile_b.input_direction[inv_tangent],
                            -tile_b.output_direction[direction],

                            # -tile_c.input_direction[direction],
                            -tile_c.output_direction[out_direction],
                            tile_c.is_splitter,
                        ])


def prevent_zigzags(grid: Grid, edge_mode: EdgeModeType):
    # TODO Make sound
    for direction in Direction:
        for tangent in (direction.next, direction.prev):
            for tiles in grid.iterate_tile_blocks(direction.vec, 2, tangent.vec, 2, edge_mode):
                if (tiles == None).any():
                    continue

                grid.clauses.append([
                    -tiles[0, 0].input_direction[direction],
                    -tiles[0, 0].output_direction[tangent],

                    *tiles[0, 1].all_direction,

                    # -tiles[1,0].input_direction[tangent],
                    -tiles[1, 0].output_direction[direction],

                    # tiles[1,1].input_direction[direction],
                    -tiles[1, 1].output_direction[tangent],
                ])


def break_vertical_symmetry(grid: Grid):
    top: List[LiteralType] = []
    bot: List[LiteralType] = []

    for dy in range(max(2, grid.height // 2)):
        for x in range(grid.width):
            top_tile = grid.get_tile_instance(x, dy)
            bot_tile = grid.get_tile_instance(x, grid.height - 1 - dy)

            top += top_tile.all_direction[::2]
            bot += bot_tile.all_direction[::2]

            top += top_tile.type
            bot += bot_tile.type

            top += top_tile.all_direction[1::4] + top_tile.all_direction[3::4]
            bot += bot_tile.all_direction[3::4] + bot_tile.all_direction[1::4]

        grid.clauses += break_symmetry(top, bot, grid.allocate_variable)


def break_horisontal_symmetry(grid: Grid, min_x: int = 0, max_x: Optional[int] = None):
    start: List[LiteralType] = []
    end: List[LiteralType] = []

    if max_x is None:
        max_x = grid.width - 1

    for dx in range(min(2, (max_x - min_x + 1) // 2)):
        for y in range(grid.height):
            start_tile = grid.get_tile_instance(min_x + dx, y)
            end_tile = grid.get_tile_instance(max_x - dx, y)

            start += start_tile.input_direction + start_tile.output_direction
            end += end_tile.output_direction + end_tile.input_direction

            start += start_tile.type
            end += end_tile.type

    grid.clauses += break_symmetry(start, end, grid.allocate_variable)


def prevent_spirals(grid: Grid):
    for direction in Direction:
        inv_direction = direction.reverse
        for across_direction in (direction.next, direction.prev):
            inv_across_direction = across_direction.reverse
            for block in grid.iterate_tile_blocks(across_direction.vec, 3, direction.vec, 3, EdgeMode.NO_WRAP):
                block[0, 0] = None  # Unimportant tile

                if (block == None).any():
                    continue

                for spiral_input_direction in (across_direction, direction):
                    grid.clauses.append(invert_components([
                        block[1, 0].input_direction[spiral_input_direction],
                        block[1, 0].output_direction[direction],

                        # block[2, 0].input_direction[direction],
                        block[2, 0].output_direction[across_direction],

                        # block[2, 1].input_direction[across_direction],
                        block[2, 1].output_direction[across_direction],
                        block[2, 1].is_belt,

                        # block[2, 2].input_direction[across_direction],
                        block[2, 2].output_direction[inv_direction],

                        # block[1, 2].input_direction[inv_direction],
                        block[1, 2].output_direction[inv_direction],

                        # block[0, 2].input_direction[inv_direction],
                        block[0, 2].output_direction[inv_across_direction],

                        # block[0, 1].input_direction[inv_across_direction],
                        block[0, 1].output_direction[direction],

                        # block[1, 1].input_direction[direction],
                        # block[1, 1].output_direction[direction],
                    ]))

                    grid.clauses.append(invert_components([
                        block[1, 0].output_direction[spiral_input_direction.reverse],
                        block[1, 0].input_direction[inv_direction],

                        # block[2, 0].output_direction[inv_direction],
                        block[2, 0].input_direction[inv_across_direction],

                        # block[2, 1].output_direction[inv_across_direction],
                        block[2, 1].input_direction[inv_across_direction],
                        block[2, 1].is_belt,

                        # block[2, 2].output_direction[inv_across_direction],
                        block[2, 2].input_direction[direction],

                        # block[1, 2].output_direction[direction],
                        block[1, 2].input_direction[direction],

                        # block[0, 2].output_direction[direction],
                        block[0, 2].input_direction[across_direction],

                        # block[0, 1].output_direction[across_direction],
                        block[0, 1].input_direction[inv_direction],

                        # block[1, 1].output_direction[inv_direction],
                        # block[1, 1].input_direction[inv_direction],
                    ]))


def prevent_belt_parallel_splitter(grid: Grid, edge_mode: EdgeModeType):
    for direction in Direction:
        for across_direction, is_head in ((direction.next, True), (direction.prev, False)):
            for block in grid.iterate_tile_blocks(direction.vec, 2, across_direction.vec, 2, edge_mode):
                if (block == None).any():
                    continue

                for in_direction in (direction, across_direction):
                    grid.clauses.append(invert_components([
                        block[0, 0].input_direction[in_direction],
                        block[0, 0].output_direction[direction],

                        *([-block[0, 0].is_splitter] if in_direction == direction else []),

                        block[1, 0].output_direction[across_direction],

                        block[1, 1].is_splitter,
                        set_literal(block[1, 1].is_splitter_head, is_head),
                    ]))


def glue_partial_splitters(grid: Grid, edge_mode: EdgeModeType):
    for direction in Direction:
        across_direction = direction.next
        for block in grid.iterate_tile_blocks(direction.vec, 2, across_direction.vec, 2, edge_mode, max_x=grid.width - 2):
            if (block == None).any():
                continue

            grid.clauses.append(invert_components([
                block[0, 0].input_direction[direction],
                block[0, 0].is_splitter_head,

                -block[1, 0].input_direction[direction],

                block[0, 1].input_direction[direction],
                block[0, 1].output_direction[direction],
                block[0, 1].is_belt,

                block[1, 1].input_direction[direction],
                block[1, 1].output_direction[direction],
                block[1, 1].is_belt,
            ]))

            grid.clauses.append(invert_components([
                -block[0, 0].input_direction[direction],

                block[1, 0].input_direction[direction],
                block[1, 0].is_splitter,
                -block[1, 0].is_splitter_head,

                block[0, 1].input_direction[direction],
                block[0, 1].output_direction[direction],
                block[0, 1].is_belt,

                block[1, 1].input_direction[direction],
                block[1, 1].output_direction[direction],
                block[1, 1].is_belt,
            ]))


def apply_generic_optimisations(grid: Grid):
    prevent_small_loops(grid)
    prevent_empty_along_underground(grid, EdgeMode.NO_WRAP)
    glue_splitters(grid)
    prevent_belt_hooks(grid, EdgeMode.NO_WRAP)
    prevent_mergeable_underground(grid, EdgeMode.NO_WRAP)
    prevent_semicircles(grid, EdgeMode.NO_WRAP)
    prevent_underground_hook(grid, EdgeMode.NO_WRAP)
    prevent_zigzags(grid, EdgeMode.NO_WRAP)
    prevent_belt_parallel_splitter(grid, EdgeMode.NO_WRAP)
    glue_partial_splitters(grid, EdgeMode.NO_WRAP)
