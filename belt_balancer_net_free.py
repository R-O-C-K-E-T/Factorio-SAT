from cardinality import library_equals, quadratic_amo, quadratic_one
import argparse, json, sys, math, warnings

from solver import Grid
from template import ArrayTemplate, BoolTemplate, NumberTemplate, EdgeMode, flatten
from util import *
import belt_balancer, optimisations

def lcm(*args):
    if len(args) == 1:
        return args[0]
    first, *rest = args

    rest_lcm = lcm(*rest)
    return (first * rest_lcm) // math.gcd(first, rest_lcm)

def next_power_of_two(x: int) -> int:
    return 1 << max(x - 1, 0).bit_length()

def create_n_to_n_balancer(width: int, height: int, underground_length: int, size: int) -> Grid:
    assert width > 0
    assert height > 0
    assert size > 0

    # denominator = 4 # 2x2 -> 1, 3x3 -> 2, 4x4 -> 1, 5x5 -> 8, 6x6 -> 2

    denominator = next_power_of_two(size)
    denominator //= math.gcd(denominator, size)

    full_flow = size * denominator # 2x2 -> 2, 3x3 -> 6, 4x4 -> 4, 5x5 -> 40, 6x6 -> 12
    flow_bits = full_flow.bit_length()
    
    grid = Grid(width, height, None, underground_length, {
        'flow'       : ArrayTemplate(NumberTemplate(flow_bits), (size - 1,)),
        'flow_diff'  : ArrayTemplate(BoolTemplate(), (size - 1, flow_bits)),
        'flow_carry' : ArrayTemplate(BoolTemplate(), (size - 1, flow_bits - 1)),
        'flow_ux'    : ArrayTemplate(NumberTemplate(flow_bits), (size - 1,)),
        'flow_uy'    : ArrayTemplate(NumberTemplate(flow_bits), (size - 1,)),
    })

    grid.block_underground_through_edges()
    grid.prevent_bad_undergrounding(EdgeMode.NO_WRAP)

    for tile in grid.iterate_tiles():
        grid.clauses += implies([tile.is_splitter], [tile.input_direction, tile.output_direction])

        for flow_component in tile.flow:
            grid.clauses += set_maximum(full_flow, flow_component)

    grid.transport_quantity(lambda tile: tile.flow, lambda tile: tile.flow_ux, lambda tile: tile.flow_uy, EdgeMode.NO_WRAP)

    for x in range(grid.width):
        for y in range(grid.height):
            tile00 = grid.get_tile_instance(x, y)
            for direction in range(4):
                dx0, dy0 = direction_to_vec(direction)
                dx1, dy1 = direction_to_vec((direction + 1) % 4)

                precondition = [
                    tile00.is_splitter_head,    
                    tile00.input_direction[direction],   
                ]
                
                tile10 = grid.get_tile_instance_offset(x, y, dx0, dy0, EdgeMode.NO_WRAP)
                tile01 = grid.get_tile_instance_offset(x, y, dx1, dy1, EdgeMode.NO_WRAP)
                tile11 = grid.get_tile_instance_offset(x, y, dx0 + dx1, dy0 + dy1, EdgeMode.NO_WRAP)

                if any(tile is None for tile in (tile00, tile10, tile01, tile11)):
                    continue

                for in_flow0, in_flow1, flow_carry, out_flow0, out_flow1 in zip(tile00.flow, tile01.flow, tile00.flow_carry, tile10.flow, tile11.flow):
                    grid.clauses += implies(precondition, [
                        *add_numbers(in_flow0[1:], in_flow1[1:], out_flow0, make_fixed_allocator(flow_carry), in_flow0[0]),
                        *literals_same(in_flow0[0], in_flow1[0]),
                        *set_numbers_equal(out_flow0, out_flow1),
                    ])

                for flow_bit0, flow_bit1, diff_bit in zip(flatten(tile00.flow), flatten(tile01.flow), flatten(tile00.flow_diff)):
                    grid.clauses += implies(precondition, [
                        [ flow_bit0,  flow_bit1, -diff_bit],
                        [-flow_bit0, -flow_bit1, -diff_bit],
                    ])

                grid.clauses += implies(precondition, [flatten(tile00.flow_diff)])
    
    for y in range(grid.height):
        tile = grid.get_tile_instance(0, y)
        grid.clauses.append([-tile.is_splitter])
        for i, flow_component in enumerate(tile.flow):
            grid.clauses += set_number(full_flow if y % size == i else 0, flow_component)

    for y in range(grid.height):
        tile = grid.get_tile_instance(grid.width - 1, y)
        grid.clauses.append([-tile.is_splitter])
        for flow_component in tile.flow:
            grid.clauses += set_number(denominator, flow_component)


    return grid

def create_n_to_m_balancer(width: int, height: int, underground_length: int, input_count: int, output_count: int) -> Grid:
    assert width > 0
    assert height > 0
    assert input_count > 0
    assert output_count > 0

    if input_count == output_count:
        return create_n_to_n_balancer(width, height, underground_length, input_count)

    # 1x2 -> 2, 1x3 -> 3, 1x4 -> 4, 1x5 -> 5, 1x6 -> 6, 1x7 -> 7, 2x2 -> 2, 2x3 -> 6, 2x4 -> 4, 2x5 -> 10, 
    # full_flow = 40
    max_belt_flow = lcm(next_power_of_two(min(input_count, output_count)), 2, input_count, output_count)
    print(max_belt_flow, file=sys.stderr)

    total_flow = max_belt_flow * min(input_count, output_count)

    forward_input_flow = total_flow // input_count
    forward_output_flow = forward_input_flow // output_count

    backward_input_flow = total_flow // output_count
    backward_output_flow = backward_input_flow // input_count

    print(forward_input_flow, forward_output_flow, backward_input_flow, backward_output_flow, file=sys.stderr)

    flow_bits = max_belt_flow.bit_length()
    grid = Grid(width, height, None, underground_length, {
        'forward': {
            'flow'  : ArrayTemplate(NumberTemplate(flow_bits), (input_count,)),
            'diff'  : ArrayTemplate(BoolTemplate(), (input_count, flow_bits)),
            'carry' : ArrayTemplate(BoolTemplate(), (input_count, flow_bits)),
            'ux'    : ArrayTemplate(NumberTemplate(flow_bits), (input_count,)),
            'uy'    : ArrayTemplate(NumberTemplate(flow_bits), (input_count,)),
        },
        'backward': {
            'flow'  : ArrayTemplate(NumberTemplate(flow_bits), (output_count,)),
            'diff'  : ArrayTemplate(BoolTemplate(), (output_count, flow_bits)),
            'carry' : ArrayTemplate(BoolTemplate(), (output_count, flow_bits)),
            'ux'    : ArrayTemplate(NumberTemplate(flow_bits), (output_count,)),
            'uy'    : ArrayTemplate(NumberTemplate(flow_bits), (output_count,)),
        },
    })

    grid.block_underground_through_edges()
    grid.prevent_bad_undergrounding(EdgeMode.NO_WRAP)

    for tile in grid.iterate_tiles():
        for flow_component in tile.forward.flow + tile.backward.flow:
            grid.clauses += set_maximum(max_belt_flow, flow_component)

        for flow_direction in (tile.forward.flow, tile.backward.flow):
            top_bits = [flow_component[-1] for flow_component in flow_direction]
            grid.clauses += quadratic_amo(top_bits)

    grid.transport_quantity(lambda tile: tile.forward.flow,  lambda tile: tile.forward.ux,  lambda tile: tile.forward.uy, EdgeMode.NO_WRAP)
    grid.transport_quantity(lambda tile: tile.backward.flow, lambda tile: tile.backward.ux, lambda tile: tile.backward.uy, EdgeMode.NO_WRAP)

    for x in range(grid.width):
        for y in range(grid.height):
            tile00 = grid.get_tile_instance(x, y)
            for direction in range(4):
                dx0, dy0 = direction_to_vec(direction)
                dx1, dy1 = direction_to_vec((direction + 1) % 4)

                tile10 = grid.get_tile_instance_offset(x, y, dx0, dy0, EdgeMode.NO_WRAP)
                tile01 = grid.get_tile_instance_offset(x, y, dx1, dy1, EdgeMode.NO_WRAP)
                tile11 = grid.get_tile_instance_offset(x, y, dx0 + dx1, dy0 + dy1, EdgeMode.NO_WRAP)

                if any(tile is None for tile in (tile00, tile10, tile01, tile11)):
                    continue


                grid.clauses.append([
                    -tile00.is_splitter_head,
                    -tile00.input_direction[direction],
                    tile00.output_direction[direction],
                    tile01.output_direction[direction],
                ])

                grid.clauses.append([
                    -tile00.is_splitter_head,
                    -tile00.input_direction[direction],
                    tile00.output_direction[direction],
                    tile01.input_direction[direction],
                ])

                grid.clauses.append([
                    -tile00.is_splitter_head,
                    -tile00.output_direction[direction],
                    tile00.input_direction[direction],
                    tile01.input_direction[direction],
                ])

                grid.clauses.append([
                    -tile00.is_splitter_head,
                    -tile00.output_direction[direction],
                    tile00.input_direction[direction],
                    tile01.output_direction[direction],
                ])

                fully_connected_precondition = [
                    tile00.is_splitter_head,
                    tile00.input_direction[direction], 
                    tile01.input_direction[direction],
                    tile00.output_direction[direction],
                    tile01.output_direction[direction],
                ]

                for in_flow0, in_flow1, flow_carry, out_flow0, out_flow1 in zip(tile00.forward.flow, tile01.forward.flow, tile00.forward.carry, tile10.forward.flow, tile11.forward.flow):
                    grid.clauses += implies(fully_connected_precondition, [
                        *add_numbers(in_flow0[1:], in_flow1[1:], out_flow0, make_fixed_allocator(flow_carry), in_flow0[0]),
                        *literals_same(in_flow0[0], in_flow1[0]),
                        *set_numbers_equal(out_flow0, out_flow1),
                    ])

                for out_flow0, out_flow1, flow_carry, in_flow0, in_flow1 in zip(tile00.backward.flow, tile01.backward.flow, tile00.backward.carry, tile10.backward.flow, tile11.backward.flow):
                    grid.clauses += implies(fully_connected_precondition, [
                        *add_numbers(in_flow0[1:], in_flow1[1:], out_flow0, make_fixed_allocator(flow_carry), in_flow0[0]),
                        *literals_same(in_flow0[0], in_flow1[0]),
                        *set_numbers_equal(out_flow0, out_flow1),
                    ])

                for flow_bit0, flow_bit1, diff_bit in zip(flatten([tile00.forward.flow, tile00.backward.flow]), flatten([tile01.forward.flow, tile01.backward.flow]), flatten([tile00.forward.diff, tile00.backward.diff])):
                    grid.clauses += implies(fully_connected_precondition, [
                        [ flow_bit0,  flow_bit1, -diff_bit],
                        [-flow_bit0, -flow_bit1, -diff_bit],
                    ])
                grid.clauses += implies(fully_connected_precondition, [flatten([tile00.forward.diff, tile00.backward.diff])])

                for connected_in_tile, connected_out_tile, unconnected_in_tile in ([tile00, tile10, tile01], [tile01, tile11, tile00]):
                    partial_input_precondition = [
                        tile00.is_splitter_head,
                        tile00.output_direction[direction],
                        connected_in_tile.input_direction[direction], 
                        -unconnected_in_tile.input_direction[direction]
                    ]

                    for in_flow, out_flow0, out_flow1 in zip(connected_in_tile.forward.flow, tile10.forward.flow, tile11.forward.flow):
                        grid.clauses += implies(partial_input_precondition, [
                            [-in_flow[0]],

                            *set_numbers_equal(in_flow[1:], out_flow0[:-1]),
                            *set_numbers_equal(in_flow[1:], out_flow1[:-1]),

                            [-out_flow0[-1]],
                            [-out_flow1[-1]],
                        ])

                    for in_flow0, in_flow1, out_flow, flow_carry in zip(tile10.backward.flow, tile11.backward.flow, connected_in_tile.backward.flow, tile00.backward.carry):
                        grid.clauses += implies(partial_input_precondition, add_numbers(in_flow0, in_flow1, out_flow, make_fixed_allocator(flow_carry)))

                    partial_output_precondition = [
                        tile00.is_splitter_head,
                        tile00.input_direction[direction],
                        connected_in_tile.output_direction[direction], 
                        -unconnected_in_tile.output_direction[direction]
                    ]

                    for in_flow0, in_flow1, out_flow, flow_carry in zip(tile00.forward.flow, tile01.forward.flow, connected_out_tile.forward.flow, tile00.forward.carry):
                        grid.clauses += implies(partial_output_precondition, add_numbers(in_flow0, in_flow1, out_flow, make_fixed_allocator(flow_carry)))

                    for in_flow, out_flow0, out_flow1 in zip(connected_out_tile.backward.flow, tile00.backward.flow, tile01.backward.flow):
                        grid.clauses += implies(partial_output_precondition, [
                            [-in_flow[0]],

                            *set_numbers_equal(in_flow[1:], out_flow0[:-1]),
                            *set_numbers_equal(in_flow[1:], out_flow1[:-1]),

                            [-out_flow0[-1]],
                            [-out_flow1[-1]],
                        ])

                    

                
    
    for y in range(grid.height):
        tile = grid.get_tile_instance(0, y)
        grid.clauses.append([-tile.is_splitter])
        for i, flow_component in enumerate(tile.forward.flow):
            grid.clauses += set_number(forward_input_flow if y % input_count == i else 0, flow_component)
        for flow_component in tile.backward.flow:
            grid.clauses += set_number(backward_output_flow, flow_component)

    for y in range(grid.height):
        tile = grid.get_tile_instance(grid.width - 1, y)
        grid.clauses.append([-tile.is_splitter])
        for flow_component in tile.forward.flow:
            grid.clauses += set_number(forward_output_flow, flow_component)
        
        for i, flow_component in enumerate(tile.backward.flow):
            grid.clauses += set_number(backward_input_flow if y % output_count == i else 0, flow_component)


    return grid

def setup_balancer_ends(grid: Grid, input_count: int, output_count: int, aligned: bool):
    start_offsets = []
    for offset in range(grid.height - input_count + 1):
        start_offset = grid.allocate_variable()
        start_offsets.append(start_offset)
        consequences = []
        for y in range(offset):
            tile = grid.get_tile_instance(0, y)
            consequences += set_all_false(tile.all_direction)
        for y in range(offset, offset+input_count):
            tile = grid.get_tile_instance(0, y)
            consequences += [[tile.input_direction[0]], [tile.output_direction[0]]]
        for y in range(offset+input_count, grid.height):
            tile = grid.get_tile_instance(0, y)
            consequences += set_all_false(tile.all_direction)
        grid.clauses += implies([start_offset], consequences) 
    grid.clauses += quadratic_one(start_offsets)

    end_offsets = []
    for offset in range(grid.height - output_count + 1):
        end_offset = grid.allocate_variable()
        end_offsets.append(end_offset)
        consequences = []
        for y in range(offset):
            tile = grid.get_tile_instance(grid.width - 1, y)
            consequences += set_all_false(tile.all_direction)
        for y in range(offset, offset+output_count):
            tile = grid.get_tile_instance(grid.width - 1, y)
            consequences += [[tile.input_direction[0]], [tile.output_direction[0]]]
        for y in range(offset+output_count, grid.height):
            tile = grid.get_tile_instance(grid.width - 1, y)
            consequences += set_all_false(tile.all_direction)
        grid.clauses += implies([end_offset], consequences) 
    grid.clauses += quadratic_one(end_offsets)

    if aligned:
        if input_count >= output_count:
            for i, start_offset in enumerate(start_offsets):
                grid.clauses += implies([start_offset], [end_offsets[i:(i + 1 + input_count - output_count)]])
        else:
            for i, end_offset in enumerate(end_offsets):
                grid.clauses += implies([end_offset], [start_offsets[i:(i + 1 + output_count - input_count)]])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates n to n belt balancers')
    parser.add_argument('width', type=int, help='Belt balancer maximum width')
    parser.add_argument('height', type=int, help='Belt balancer maximum height')
    parser.add_argument('input_count', type=int, help='Number of inputs')
    parser.add_argument('output_count', type=int, help='Number of outputs')
    parser.add_argument('--underground-length', type=int, default=4, help='Sets the maximum length of underground section (excludes ends)')
    parser.add_argument('--aligned', action='store_true', help='Enforces balancer input aligns with output')
    parser.add_argument('--all', action='store_true', help='Generate all belt balancers')
    parser.add_argument('--solver', type=str, default='Glucose3', help='Backend SAT solver to use')
    args = parser.parse_args()

    if args.underground_length == -1:
        args.underground_length = float('inf')

    if args.input_count != args.output_count:
        # raise RuntimeWarning('Different sized inputs does not always produce good/correct results')
        warnings.warn('Different sized inputs does not always produce good/correct results', RuntimeWarning)

    grid = create_n_to_m_balancer(args.width, args.height, args.underground_length, args.input_count, args.output_count)

    setup_balancer_ends(grid, args.input_count, args.output_count, args.aligned)
 
    grid.block_belts_through_edges((False, True))
    grid.prevent_intersection(EdgeMode.NO_WRAP)

    grid.enforce_maximum_underground_length(EdgeMode.NO_WRAP)

    optimisations.expand_underground(grid, min_x=1, max_x=grid.width-2)
    optimisations.apply_generic_optimisations(grid)
    # This optimisation likely conflicts with other optimisations
    #optimisations.break_vertical_symmetry(grid)
    belt_balancer.prevent_double_edge_belts(grid)

    for solution in grid.itersolve(solver=args.solver, ignore_colour=True):
        print(json.dumps(solution.tolist()))
        if not args.all:
            break