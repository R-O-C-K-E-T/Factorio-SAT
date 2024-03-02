import argparse
import glob
import json
import shutil
import sys
import os

from os import path

ASSETS_DIR = path.join(os.getenv("XDG_DATA_HOME"), "factorio-sat/assets")

try:
    from luaparser import ast
except ModuleNotFoundError:
    print('"luaparser" not installed: recipe fetching will be disabled')
    ast = None


def copy_game_tilemaps(base_dir: str, assets_dir: str):
    graphics_dir = path.join(base_dir, 'graphics', 'entity')

    files = [
        ('assembling-machine-1', 'hr-assembling-machine-1.png'),
        ('assembling-machine-1', 'hr-assembling-machine-1-shadow.png'),

        ('burner-inserter', 'hr-burner-inserter-hand-base-shadow.png'),
        ('burner-inserter', 'hr-burner-inserter-hand-closed-shadow.png'),
        ('burner-inserter', 'hr-burner-inserter-hand-open-shadow.png'),

        ('inserter', 'hr-inserter-hand-base.png'),
        ('inserter', 'hr-inserter-hand-closed.png'),
        ('inserter', 'hr-inserter-hand-open.png'),
        ('inserter', 'hr-inserter-platform.png'),

        ('long-handed-inserter', 'hr-long-handed-inserter-hand-base.png'),
        ('long-handed-inserter', 'hr-long-handed-inserter-hand-closed.png'),
        ('long-handed-inserter', 'hr-long-handed-inserter-hand-open.png'),
        ('long-handed-inserter', 'hr-long-handed-inserter-platform.png'),

        ('splitter', 'hr-splitter-east-top_patch.png'),
        ('splitter', 'hr-splitter-east.png'),
        ('splitter', 'hr-splitter-north.png'),
        ('splitter', 'hr-splitter-south.png'),
        ('splitter', 'hr-splitter-west-top_patch.png'),
        ('splitter', 'hr-splitter-west.png'),

        ('transport-belt', 'hr-transport-belt.png'),

        ('underground-belt', 'hr-underground-belt-structure.png'),
    ]

    for file in files:
        source = path.join(graphics_dir, *file)

        destination = path.join(assets_dir, file[-1])

        print('Copying: {} -> {}'.format(source, destination))
        shutil.copyfile(source, destination)


def decode_lua_data(text):
    tree = ast.parse(text)

    invoke, = tree.body.body
    assert invoke.source.id == 'data' and invoke.func.id == 'extend'

    table, = invoke.args
    assert isinstance(table, ast.Table)

    def recurse(node):
        if isinstance(node, ast.Table):
            result = {}
            for field in node.fields:
                if isinstance(field.key, ast.Number):
                    key = field.key.n
                elif isinstance(field.key, ast.Name):
                    key = field.key.id
                else:
                    assert False
                result[key] = recurse(field.value)
            try:
                return [result[i + 1] for i in range(len(result))]
            except KeyError:
                return result
        elif isinstance(node, ast.String):
            return node.s
        elif isinstance(node, ast.FalseExpr):
            return False
        elif isinstance(node, ast.TrueExpr):
            return True
        elif isinstance(node, ast.Number):
            return node.n
        else:
            assert False
    return recurse(table)


def get_recipes_for_variant(data, variant):
    recipes = []
    for entry in data:
        if 'category' in entry and entry['category'] not in ('crafting', 'advanced-crafting'):
            continue

        time = entry.get('energy_required', 0.5)

        if variant in entry:
            entry = entry[variant]

        time = entry.get('energy_required', time)

        if 'ingredients' not in entry and ('result' not in entry or 'results' not in entry):
            entry = entry['normal']

        ingredients = []
        for item in entry['ingredients']:
            if isinstance(item, list):
                ingredients.append({'name': item[0], 'amount': item[1]})
            else:
                assert item['type'] == 'item'
                ingredients.append({'name': item['name'], 'amount': item['amount']})

        if 'results' in entry:
            results = []
            for item in entry['results']:
                assert item['type'] == 'item'
                results.append({'name': item['name'], 'amount': item['amount']})
        else:
            results = [{'name': entry['result'], 'amount': entry.get('result_count', 1)}]

        recipes.append({
            'time': time,
            'ingredients': ingredients,
            'results': results
        })
    return recipes


def copy_game_recipes(base_dir, assets_dir):
    data = []
    for file in glob.glob(path.join(base_dir, 'prototypes', 'recipe', '*.lua')):
        with open(file) as f:
            text = f.read()

        entry = decode_lua_data(text)
        assert isinstance(entry, list)

        data += entry

    for variant in ('normal', 'expensive'):
        with open(path.join(assets_dir, f'{variant}-recipes.json'), 'w') as f:
            json.dump(get_recipes_for_variant(data, variant), f)


def main():
    parser = argparse.ArgumentParser(description='Fetches Factorio tilemaps and recipes')
    parser.add_argument('path', type=str, nargs='?', help='Location of factorio installation')
    args = parser.parse_args()

    if args.path is None:
        if sys.platform.startswith('linux'):
            steam_directory = path.expanduser('~/.steam')
        elif sys.platform.startswith('win32'):
            steam_directory = 'C:\\Program Files (x86)\\Steam'
        elif sys.platform.startswith('darwin'):
            steam_directory = path.expanduser('~/Library/Application Support/Steam')
        else:
            raise RuntimeError('Unknown platform: {}'.format(sys.platform))

        if not path.exists(steam_directory):
            raise RuntimeError('No steam installation found at: {}'.format(steam_directory))
        game_directory = path.join(steam_directory, 'steamapps', 'common', 'Factorio')
    else:
        game_directory = args.path

    if not path.exists(game_directory):
        raise RuntimeError('Factorio not found at: {}'.format(game_directory))

    game_base_dir = path.join(game_directory, 'data', 'base')

    os.makedirs(ASSETS_DIR, exist_ok=True)
    copy_game_tilemaps(game_base_dir, ASSETS_DIR)
    if ast is not None:
        copy_game_recipes(game_base_dir, ASSETS_DIR)


if __name__ == '__main__':
    main()
