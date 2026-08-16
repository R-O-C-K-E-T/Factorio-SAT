import argparse
import glob
import json
import shutil
import sys
from os import path

from PIL import Image

try:
    from luaparser import ast
except ModuleNotFoundError:
    print('"luaparser" not installed: recipe fetching will be disabled')
    ast = None


ASSET_FILES = [
    ('assembling-machine-1', 'assembling-machine-1-shadow.png'),

    ('burner-inserter', 'burner-inserter-hand-base-shadow.png'),
    ('burner-inserter', 'burner-inserter-hand-closed-shadow.png'),
    ('burner-inserter', 'burner-inserter-hand-open-shadow.png'),

    ('inserter', 'inserter-hand-base.png'),
    ('inserter', 'inserter-hand-closed.png'),
    ('inserter', 'inserter-hand-open.png'),
    ('inserter', 'inserter-platform.png'),

    ('long-handed-inserter', 'long-handed-inserter-hand-base.png'),
    ('long-handed-inserter', 'long-handed-inserter-hand-closed.png'),
    ('long-handed-inserter', 'long-handed-inserter-hand-open.png'),
    ('long-handed-inserter', 'long-handed-inserter-platform.png'),

    ('splitter', 'splitter-east-top_patch.png'),
    ('splitter', 'splitter-east.png'),
    ('splitter', 'splitter-north.png'),
    ('splitter', 'splitter-south.png'),
    ('splitter', 'splitter-west-top_patch.png'),
    ('splitter', 'splitter-west.png'),

    ('transport-belt', 'transport-belt.png'),

    ('underground-belt', 'underground-belt-structure.png'),
]


def find_asset(graphics_directory: str, entity: str, filename: str):
    """Find an asset in either the Factorio 2.x or legacy 1.1 layout."""
    # Factorio 1.1 contains both standard- and high-resolution sprites. Prefer
    # the high-resolution one because the renderer's frame sizes are based on it.
    candidates = [f'hr-{filename}', filename]
    for candidate in candidates:
        source = path.join(graphics_directory, entity, candidate)
        if path.isfile(source):
            return source
    raise FileNotFoundError(
        f'Could not find {filename} for {entity}; tried: '
        + ', '.join(path.join(graphics_directory, entity, candidate) for candidate in candidates)
    )


def compose_assembling_machine(graphics_directory: str, destination: str):
    """Build the combined sprite sheet expected by the renderer from 2.x layers."""
    entity_directory = path.join(graphics_directory, 'assembling-machine-1')
    legacy_source = path.join(entity_directory, 'hr-assembling-machine-1.png')
    if path.isfile(legacy_source):
        print(f'Copying: {legacy_source} -> {destination}')
        shutil.copyfile(legacy_source, destination)
        return

    base_source = path.join(entity_directory, 'assembling-machine-1-base.png')
    animation_source = path.join(entity_directory, 'assembling-machine-1-anim.png')
    if not path.isfile(base_source) or not path.isfile(animation_source):
        raise FileNotFoundError(
            'Could not find either the Factorio 1.1 combined assembling-machine sprite '
            'or the Factorio 2.x base and animation sprites'
        )

    print(f'Composing: {base_source} + {animation_source} -> {destination}')
    with Image.open(base_source) as base_image, Image.open(animation_source) as animation_image:
        base_image = base_image.convert('RGBA')
        animation_image = animation_image.convert('RGBA')

        animation_frame_size = (140, 158)
        columns = animation_image.width // animation_frame_size[0]
        rows = animation_image.height // animation_frame_size[1]
        if columns * animation_frame_size[0] != animation_image.width or rows * animation_frame_size[1] != animation_image.height:
            raise RuntimeError(f'Unexpected assembling machine animation size: {animation_image.size}')

        # The legacy renderer uses a 214x226 frame at 64 source pixels per tile.
        # Factorio 2.x sprite shifts are expressed in 32-pixel game coordinates,
        # hence the factor of two when placing them in the high-resolution sheet.
        frame_size = (214, 226)
        base_shift = (0, 4)
        animation_shift = (0, -13.5)
        output = Image.new('RGBA', (columns * frame_size[0], rows * frame_size[1]))

        def position(image, shift):
            return (
                round((frame_size[0] - image.width) / 2 + 2 * shift[0]),
                round((frame_size[1] - image.height) / 2 + 2 * shift[1]),
            )

        base_position = position(base_image, base_shift)
        for row in range(rows):
            for column in range(columns):
                frame = Image.new('RGBA', frame_size)
                frame.alpha_composite(base_image, base_position)
                source_box = (
                    column * animation_frame_size[0],
                    row * animation_frame_size[1],
                    (column + 1) * animation_frame_size[0],
                    (row + 1) * animation_frame_size[1],
                )
                animation_frame = animation_image.crop(source_box)
                frame.alpha_composite(animation_frame, position(animation_frame, animation_shift))
                output.alpha_composite(frame, (column * frame_size[0], row * frame_size[1]))
        output.save(destination)


def fetch_tilemaps(base_directory: str, destination_directory: str = None):
    graphics_directory = path.join(base_directory, 'graphics', 'entity')
    if destination_directory is None:
        destination_directory = path.dirname(__file__)

    for entity, filename in ASSET_FILES:
        source = find_asset(graphics_directory, entity, filename)
        destination = path.join(destination_directory, filename)

        print(f'Copying: {source} -> {destination}')
        shutil.copyfile(source, destination)

    compose_assembling_machine(
        graphics_directory,
        path.join(destination_directory, 'assembling-machine-1.png'),
    )


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


def fetch_recipes(base_directory):
    recipe_directory = path.join(base_directory, 'prototypes', 'recipe')
    if not path.isdir(recipe_directory):
        print('Skipping legacy recipe export (Factorio 2.x no longer has difficulty-specific recipe files)')
        return

    data = []
    for file in glob.glob(path.join(recipe_directory, '*.lua')):
        with open(file) as f:
            text = f.read()

        entry = decode_lua_data(text)
        assert isinstance(entry, list)

        data += entry

    for variant in ('normal', 'expensive'):
        with open(path.join(path.dirname(__file__), f'{variant}-recipes.json'), 'w') as f:
            json.dump(get_recipes_for_variant(data, variant), f)


def default_game_directories(platform: str = sys.platform):
    if platform.startswith('linux'):
        return [
            path.expanduser('~/.steam/steam/steamapps/common/Factorio'),
            path.expanduser('~/.steam/steamapps/common/Factorio'),
            path.expanduser('~/.local/share/Steam/steamapps/common/Factorio'),
            path.expanduser('~/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Factorio'),
        ]
    elif platform.startswith('win32'):
        return [r'C:\Program Files (x86)\Steam\steamapps\common\Factorio']
    elif platform.startswith('darwin'):
        return [path.expanduser('~/Library/Application Support/Steam/steamapps/common/Factorio')]
    raise RuntimeError(f'Unknown platform: {platform}')


def main():
    parser = argparse.ArgumentParser(description='Fetches Factorio tilemaps and recipes')
    parser.add_argument('path', type=str, nargs='?', help='Location of factorio installation')
    args = parser.parse_args()

    if args.path is None:
        candidates = default_game_directories()
        game_directory = next((candidate for candidate in candidates if path.isdir(candidate)), None)
        if game_directory is None:
            raise RuntimeError('Factorio not found; checked: {}'.format(', '.join(candidates)))
    else:
        game_directory = args.path

    if not path.exists(game_directory):
        raise RuntimeError('Factorio not found at: {}'.format(game_directory))

    base_directory = path.join(game_directory, 'data', 'base')

    fetch_tilemaps(base_directory)
    if ast is not None:
        fetch_recipes(base_directory)


if __name__ == '__main__':
    main()
