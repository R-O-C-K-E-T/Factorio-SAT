import argparse, json, math
from cardinality import quadratic_amo, quadratic_one

from fractions import Fraction
from collections import Counter
from template import EdgeMode, OneHotTemplate

import solver2
from util import *

def lcm(a: int, b: int):
    return abs(a * b) // math.gcd(a, b)

def calculate_common_fraction(a: Fraction, b: Fraction):
    return Fraction(math.gcd(a.numerator, b.numerator), lcm(a.denominator, b.denominator))

def expand_recipe(recipe_map, item, rate):
    recipe = recipe_map.get(item)
    if recipe is None:
        return Counter()
    
    _, amount, ingredients = recipe

    product_rate = Fraction(rate, amount)

    result = Counter({item: product_rate})
    for ingredient_name, quantity in ingredients:
        result += expand_recipe(recipe_map, ingredient_name, product_rate * quantity)
    return result

def compute_assembler_properties(recipe_map, amounts, max_belt_flowrate):
    input_items = set()
    output_items = set()

    item_unit_size = {}
    for item, product_rate in amounts.items():
        output_items.add(item)
        time, amount, ingredients = recipe_map[item]

        time = Fraction(time)
        amount = Fraction(amount)

        assembler_count = math.ceil(product_rate * time)

        output_per_assembler = product_rate * amount / assembler_count
        if item in item_unit_size:
            item_unit_size[item] = calculate_common_fraction(item_unit_size[item], output_per_assembler)
        else:
            item_unit_size[item] = output_per_assembler

        for ingredient, amount in ingredients:
            input_items.add(ingredient)
            input_per_assembler = product_rate * amount / assembler_count

            if ingredient in item_unit_size:
                item_unit_size[ingredient] = calculate_common_fraction(item_unit_size[ingredient], input_per_assembler)
            else:
                item_unit_size[ingredient] = input_per_assembler
    
    raw_inputs = input_items - output_items
    products = output_items - input_items
    del input_items, output_items
    
    maximum_flow = {}
    for item, unit_size in item_unit_size.items():
        maximum_flow[item] = math.floor(max_belt_flowrate / unit_size)

    total_input = Counter()
    total_output = {}

    assemblers = []
    for item, product_rate in amounts.items():
        time, amount, ingredients = recipe_map[item]

        time = Fraction(time)
        amount = Fraction(amount)

        assembler_count = math.ceil(product_rate * time)
        output_per_assembler = product_rate * amount / assembler_count

        output_units = output_per_assembler / item_unit_size[item]
        assert output_units.denominator == 1
        output_units = output_units.numerator
        
        total_output[item] = output_units * assembler_count

        assembler_inputs = []
        for ingredient, amount in ingredients:
            input_per_assembler = product_rate * amount / assembler_count
            input_units = input_per_assembler / item_unit_size[ingredient]
            assert input_units.denominator == 1
            input_units = input_units.numerator

            total_input[ingredient] += input_units

            assembler_inputs.append((ingredient, input_units))
        assemblers.append(((item, output_units), assembler_inputs))

    raw_inputs = dict((item, total_input[item]) for item in raw_inputs)
    products = dict((item, total_output[item]) for item in products)

    for item in maximum_flow:
        maximum_flow[item] = min(maximum_flow[item], max(total_output.get(item, 0), total_input.get(item, 0)))

    return raw_inputs, products, maximum_flow, assemblers, item_unit_size

def numbers_add_to(value: int, literals_a: List[LiteralType], literals_b: List[LiteralType]) -> ClauseList:
    assert len(literals_a) == len(literals_b)

    assert value < (1 << (len(literals_a) + 1))

    clauses = []
    for value_a in range(1 << len(literals_a)):
        for value_b in range(1 << len(literals_b)):
            if value_a + value_b != value:
                clauses.append(set_not_number(value_a, literals_a) + set_not_number(value_b, literals_b))
    return clauses

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a crafter for the given item')
    parser.add_argument('width', type=int, help='Block width')
    parser.add_argument('height', type=int, help='Block height')
    parser.add_argument('item_name', type=str, help='Name of the item to produce')
    parser.add_argument('speed', type=float, help='Rate of item production (item/s) given a crafting speed of 1')
    parser.add_argument('--all', action='store_true', help='Produce all crafters')
    parser.add_argument('--solver', type=str, default='Glucose3', help='Backend SAT solver to use')

    parser.add_argument('--expensive', action='store_const', dest='recipe_variant', default='normal', const='expensive', help='Whether to use the expensive recipe variants')

    #parser.add_argument('output', type=argparse.FileType('w'), nargs='?', help='Output file, if no file provided then results are sent to standard out')
    args = parser.parse_args()

    with open(f'assets/{args.recipe_variant}-recipes.json') as f:
        recipes = json.load(f)

    # TODO make adjustable
    max_belt_flowrate = 15

    
    recipe_map = {}
    for recipe in recipes:
        if len(recipe['results']) != 1:
            continue
        amounts = recipe['results'][0]
        result_item = amounts['name']

        is_recipe_invalid = False

        ingredients = []
        for ingredient in recipe['ingredients']:
            item = ingredient['name']
            if item == result_item:
                is_recipe_invalid = True
                break

            ingredients.append((item, ingredient['amount']))
        
        if is_recipe_invalid:
            continue

        recipe_map[result_item] = recipe['time'], amounts['amount'], ingredients
    del recipes
    

    amounts = expand_recipe(recipe_map, args.item_name, Fraction(args.speed))
    if len(amounts) == 0:
        raise RuntimeError('No recipe found for the given item')

    raw_inputs, products, maximum_flow, assemblers, item_unit_size = compute_assembler_properties(recipe_map, amounts, max_belt_flowrate)
    
    if any(raw_inputs[item] > maximum_flow[item] for item in raw_inputs):
        raise NotImplementedError

    flow_bits = max(maximum_flow.values()).bit_length()

    colour_mapping = set()
    for raw_input in raw_inputs:
        colour_mapping.add(raw_input)
    for product in products:
        colour_mapping.add(product)
    for (product, _), ingredients in assemblers:
        colour_mapping.add(product)
        for ingredient, _ in ingredients:
            colour_mapping.add(ingredient)
    colour_mapping = dict((item, i) for i, item in enumerate(colour_mapping))
    colour_bits = (len(colour_mapping) - 1).bit_length()
    
    grid = solver2.Grid(args.width, args.height, colour_bits, flow_bits, {
        'assembler_type' : OneHotTemplate(len(assemblers)),
    })

    grid.setup_multitile_entities(EdgeMode.BLOCK)

    grid.prevent_intersection(EdgeMode.IGNORE)
    grid.prevent_bad_undergrounding(EdgeMode.BLOCK)
    grid.set_maximum_underground_length(4, EdgeMode.BLOCK)
    grid.prevent_empty_along_underground(4, EdgeMode.BLOCK)

    grid.prevent_bad_colouring(EdgeMode.BLOCK)
    grid.prevent_bad_flow(EdgeMode.BLOCK)

    grid.prevent_bad_insertion(EdgeMode.BLOCK)
    grid.enforce_flow_summation(EdgeMode.IGNORE)
    grid.enforce_insertion_side()

    for colour in range(len(colour_mapping), 1<<colour_bits):
        grid.prevent_colour(colour)

    for item, max_flow in maximum_flow.items():
        grid.set_maximum_flow(colour_mapping[item], max_flow)

    for x in range(grid.width):
        for y in range(grid.height):
            tile = grid.get_tile_instance(x, y)
            grid.clauses += quadratic_amo(tile.assembler_type)
            grid.clauses += implies([tile.assembling_x[0], tile.assembling_y[0]], [tile.assembler_type])
            grid.clauses += implies([-tile.assembling_x[0]], set_all_false(tile.assembler_type))
            grid.clauses += implies([-tile.assembling_y[0]], set_all_false(tile.assembler_type))

    for x in (0, grid.width - 1):
        for y in (0, grid.height - 1):
            tile = grid.get_tile_instance(x, y)
            grid.clauses.append([tile.is_empty])

    edge_tiles = [(x, y) for y in range(1, grid.height - 1) for x in (0, grid.width - 1)] + [(x, y) for x in range(1, grid.width - 1) for y in (0, grid.height - 1)]
    output_literals = [[grid.allocate_variable() for _ in edge_tiles] for _ in products]
    input_literals = [[grid.allocate_variable() for _ in edge_tiles] for _ in raw_inputs]

    for pos, literals in zip(edge_tiles, zip(*output_literals, *input_literals)):
        tile = grid.get_tile_instance(*pos)
        grid.clauses += quadratic_amo(literals)
        grid.clauses += implies([tile.is_empty], set_all_false(literals))
        grid.clauses += implies(invert_components(literals), [[tile.is_empty]])


    for (item, flow), literal_set in zip(raw_inputs.items(), input_literals):
        colour = colour_mapping[item]
        grid.clauses += quadratic_one(literal_set)
        for (x, y), lit in zip(edge_tiles, literal_set):
            tile = grid.get_tile_instance(x, y)
            if x == 0:
                direction = 0
            elif y == grid.height - 1:
                direction = 1
            elif x == grid.width - 1:
                direction = 2
            elif y == 0:
                direction = 3
            else:
                assert False

            grid.clauses += implies([lit], [
                [tile.is_belt], 
                [tile.output_direction[direction]], 
                [tile.input_direction[direction]],
                *numbers_add_to(flow, tile.flow_out[0], tile.flow_out[1]),
                *set_number(colour, tile.colour[0]),
                *set_number(colour, tile.colour[1]),
            ])

    for (item, flow), literal_set in zip(products.items(), output_literals):
        colour = colour_mapping[item]
        grid.clauses += quadratic_one(literal_set)
        for (x, y), lit in zip(edge_tiles, literal_set):
            tile = grid.get_tile_instance(x, y)
            if x == 0:
                direction = 2
            elif y == grid.height - 1:
                direction = 3
            elif x == grid.width - 1:
                direction = 0
            elif y == 0:
                direction = 1
            else:
                assert False

            grid.clauses += implies([lit], [
                [tile.is_belt], 
                [tile.output_direction[direction]],
                [tile.input_direction[direction]],
                *numbers_add_to(flow, tile.flow_out[0], tile.flow_out[1]),
                *set_number(colour, tile.colour[0]),
                *set_number(colour, tile.colour[1]),
            ])

    for solution in grid.itersolve(solver=args.solver):
        print(json.dumps(solution.tolist()))
        if not args.all:
            break