from typing import Any, Dict, Optional, Tuple
import argparse
import base64
import copy
import enum
import json
import math
import struct
import zlib

import numpy as np

from .direction import Direction
from .tile import AssemblingMachine, BaseTile, Belt, BeltConnectedTile, EmptyTile, FillerTile, Inserter, Splitter, UndergroundBelt


class TransportBeltLevel(enum.Enum):
    NORMAL = 'transport-belt', 'underground-belt', 'splitter'
    FAST = 'fast-transport-belt', 'fast-underground-belt', 'fast-splitter'
    EXPRESS = 'express-transport-belt', 'express-underground-belt', 'express-splitter'

    def __init__(self, belt_variant: str, underground_variant: str, splitter_variant: str):
        self.belt_variant = belt_variant
        self.underground_variant = underground_variant
        self.splitter_variant = splitter_variant


def encode_factorio_version(major: int, minor: int, patch: int, developer: int) -> int:
    return struct.unpack('<Q', struct.pack('<HHHH', major, minor, patch, developer))[0]


def decode_factorio_version(value: int) -> Tuple[int, int, int, int]:
    return struct.unpack('<HHHH', struct.pack('<Q', value))


def decode_blueprint(string: str):
    version = string[0]
    if version != '0':
        raise RuntimeError('Invalid blueprint version')
    string = string[1:]
    compressed = base64.b64decode(string)
    raw_bytes = zlib.decompress(compressed)
    data = json.loads(raw_bytes)
    return data


def encode_blueprint(data) -> str:
    raw_bytes = json.dumps(data).encode('utf-8')
    compressed = zlib.compress(raw_bytes, level=9)
    string = base64.b64encode(compressed).decode('utf-8')
    return '0' + string


BLUEPRINT_TEMPLATE = {'blueprint': {'icons': [{'signal': {'type': 'item', 'name': 'splitter'}, 'index': 1}], 'item': 'blueprint', 'version': 281474976710656}}


def make_blueprint(tiles, label: Optional[str] = None, level: TransportBeltLevel = TransportBeltLevel.NORMAL):
    entities = []
    entity_number = 1
    for (y, x), tile in np.ndenumerate(tiles):
        if isinstance(tile, EmptyTile) or isinstance(tile, FillerTile):
            continue

        entity = {'entity_number': entity_number, 'position': {'x': x + 0.5, 'y': y + 0.5}}
        if isinstance(tile, Belt):
            entity['name'] = level.belt_variant

            direction = tile.output_direction.factorio_direction
            if direction != 0:
                entity['direction'] = direction
        elif isinstance(tile, UndergroundBelt):
            entity['name'] = level.underground_variant

            direction = tile.direction.factorio_direction
            if direction != 0:
                entity['direction'] = direction

            entity['type'] = 'input' if tile.is_input else 'output'
        elif isinstance(tile, Splitter):
            if not tile.is_head:
                continue
            entity['name'] = level.splitter_variant

            direction = tile.direction.factorio_direction
            if direction != 0:
                entity['direction'] = direction

            dx, dy = tile.direction.next.vec
            entity['position']['x'] += dx / 2
            entity['position']['y'] += dy / 2
        elif isinstance(tile, Inserter):
            # invert direction

            if tile.type == 0:  # Normal
                entity['name'] = 'inserter'
            elif tile.type == 1:  # Long
                entity['name'] = 'long-handed-inserter'

            direction = tile.direction.reverse.factorio_direction
            if direction != 0:
                entity['direction'] = direction
        else:
            raise RuntimeError(f'Unencodable tile: {tile}')
        entities.append(entity)
        entity_number += 1

    result = copy.deepcopy(BLUEPRINT_TEMPLATE)
    if label is not None:
        result['blueprint']['label'] = label
    result['blueprint']['entities'] = entities
    return result


def resolve_belt_input_directions(tiles):
    for (y, x), tile in np.ndenumerate(tiles):
        if not isinstance(tile, Belt):
            continue

        input_direction = None
        for direction in Direction:
            x1 = x - direction.dx
            y1 = y - direction.dy
            if x1 < 0 or x1 >= tiles.shape[1] or y1 < 0 or y1 >= tiles.shape[0]:
                continue

            neighbour = tiles[y1, x1]
            if not isinstance(neighbour, BeltConnectedTile):
                continue

            if neighbour.output_direction != direction:
                continue

            if input_direction is None:
                input_direction = direction
            else:
                input_direction = None
                break

        if input_direction is None:
            input_direction = tile.output_direction

        tiles[y, x] = Belt(input_direction, tile.output_direction)


def import_blueprint(data: Any):
    if len(data['blueprint']['entities']) == 0:
        return np.full((0, 0), EmptyTile())

    entities = {}

    for entity in data['blueprint']['entities']:
        pos = entity['position']['x'], entity['position']['y']
        name = entity['name']
        direction = Direction.from_factorio(entity.get('direction', 0))

        floor_pos = math.floor(pos[0]), math.floor(pos[1])

        if any(name == level.splitter_variant for level in TransportBeltLevel):
            dx, dy = direction.next.vec
            x, y = pos
            entities[math.floor(x - dx / 2), math.floor(y - dy / 2)] = Splitter(direction, True)
            entities[math.floor(x + dx / 2), math.floor(y + dy / 2)] = Splitter(direction, False)
        elif any(name == level.belt_variant for level in TransportBeltLevel):
            entities[floor_pos] = Belt(direction, direction)
        elif any(name == level.underground_variant for level in TransportBeltLevel):
            entities[floor_pos] = UndergroundBelt(direction, entity['type'] == 'input')
        elif name in ('burner-inserter', 'inserter', 'fast-inserter', 'filter-inserter', 'stack-inserter', 'stack-filter-inserter'):
            entities[floor_pos] = Inserter(direction.reverse, 0)
        elif name == 'long-handed-inserter':
            entities[floor_pos] = Inserter(direction.reverse, 1)
        elif name.startswith('assembling-machine-'):
            x0 = floor_pos[0] - 1
            y0 = floor_pos[1] - 1
            for dx in range(3):
                for dy in range(3):
                    entities[x0 + dx, y0 + dy] = AssemblingMachine(dx, dy)
        elif name == 'wooden-chest':
            entities[floor_pos] = FillerTile()
        else:
            raise RuntimeError('Unsupported entity: ' + name)

    min_x = min(x for x, _ in entities)
    min_y = min(y for _, y in entities)

    old_entities = entities.copy()
    entities.clear()
    for (x, y), tile in old_entities.items():
        entities[x - min_x, y - min_y] = tile

    width = max(x for x, _ in entities) + 1
    height = max(y for _, y in entities) + 1

    tiles = np.full((height, width), EmptyTile())
    for pos, entity in entities.items():
        tiles[pos[::-1]] = entity
    del entities

    resolve_belt_input_directions(tiles)
    return tiles


def read_tile(item) -> BaseTile:
    if 'tile' in item:
        return BaseTile.read(item['tile'])

    input_direction = item['input_direction']
    output_direction = item['output_direction']
    if 'is_empty' in item:
        if item['is_empty']:
            tile = EmptyTile()
        elif item['is_belt']:
            if input_direction is None:
                input_direction = item['colour_direction']
            if output_direction is None:
                output_direction = item['colour_direction']

            assert input_direction is not None or output_direction is not None

            tile = Belt(input_direction, output_direction)
        elif item['is_underground_in']:
            tile = UndergroundBelt(input_direction, True)
        elif item['is_underground_out']:
            tile = UndergroundBelt(output_direction, False)
        elif item['is_splitter'] is not None and not item['is_splitter'] is False:
            try:
                tile = Splitter(item['colour_direction'], item['is_splitter'])
            except KeyError:
                tile = Splitter(item['splitter_direction'], item['splitter_side'])
        elif item['is_inserter'] is not None:
            tile = Inserter(item['inserter_direction'], item['is_inserter'])
        elif item['is_assembling_machine']:
            tile = AssemblingMachine(item['assembling_x'], item['assembling_y'])
        else:
            assert False
    else:
        if item['is_splitter']:
            direction = input_direction
            if direction is None:
                direction = output_direction
                assert direction is not None
            tile = Splitter(direction, item['is_splitter_head'])
        elif input_direction is None and output_direction is None:
            tile = EmptyTile()
        elif input_direction is None or output_direction is None:
            direction = input_direction
            if direction is None:
                direction = output_direction
                assert direction is not None
            tile = UndergroundBelt(direction, output_direction is None)
        else:
            tile = Belt(input_direction, output_direction)
    return tile


def write_tile(tile: BaseTile) -> Dict[str, Any]:
    return {'tile': tile.write()}


def convert_to_tiles(blueprint_or_json: str) -> np.ndarray:
    try:
        decoded_blueprint = decode_blueprint(blueprint_or_json)
    except Exception:
        try:
            tiles = json.loads(blueprint_or_json)
        except json.JSONDecodeError:
            raise RuntimeError('Failed to decode string as either blueprint or json')
        tiles = np.vectorize(read_tile)(tiles)
    else:
        tiles = import_blueprint(decoded_blueprint)

    return tiles


def main():
    parser = argparse.ArgumentParser(description='Encode/Decode blueprint strings')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    encode_parser = subparsers.add_parser('encode', help='Convert solver output into blueprint')
    encode_parser.add_argument('--label', type=str, help='Label for created blueprint')
    encode_parser.add_argument('--level', choices=[level.name.lower() for level in TransportBeltLevel], default='normal', help='Belt technology level to use')

    subparsers.add_parser('decode', help='Convert blueprint to solver output format')

    args = parser.parse_args()

    if args.mode == 'encode':
        while True:
            tiles = np.array(json.loads(input()))
            tiles = np.vectorize(read_tile)(tiles)
            print(encode_blueprint(make_blueprint(tiles, args.label, TransportBeltLevel[args.level.upper()])))

    elif args.mode == 'decode':
        while True:
            decoded = decode_blueprint(input())
            tiles = np.vectorize(write_tile)(import_blueprint(decoded))
            print(json.dumps(tiles.tolist()))
    else:
        assert False


if __name__ == '__main__':
    main()
