import argparse
import json
import sys

import numpy as np

from blueprint import read_tile, write_tile
from direction import Direction
from tile import Belt, EmptyTile, Splitter, UndergroundBelt


def raw_print(data):
    sys.stdout.buffer.write(data)


def attrib_string(attribs):
    return b'\x1b[' + b';'.join(bytes(str(attrib), 'utf-8') for attrib in attribs) + b'm'


def style_seq(fg=None, bg=None, bold=False, underlined=False):
    attribs = []

    if bold:
        attribs.append(1)
    if underlined:
        attribs.append(4)

    if fg is not None:
        if fg < 0:
            raise ValueError(f'Invalid foreground colour: {fg}')
        elif fg < 8:
            attribs.append(30 + fg)
        elif fg < 16:
            attribs.append(90 + fg - 8)
        else:
            raise ValueError(f'Invalid foreground colour: {fg}')

    if bg is not None:
        if bg < 0:
            raise ValueError(f'Invalid background colour: {bg}')
        elif bg < 8:
            attribs.append(40 + bg)
        elif bg < 16:
            attribs.append(100 + bg - 8)
        else:
            raise ValueError(f'Invalid background colour: {bg}')

    return attrib_string(attribs)


MAPPING = {
    EmptyTile(): ' ',
    Belt(Direction.RIGHT, Direction.RIGHT): '→',
    Belt(Direction.UP, Direction.UP): '↑',
    Belt(Direction.LEFT, Direction.LEFT): '←',
    Belt(Direction.DOWN, Direction.DOWN): '↓',

    Belt(Direction.RIGHT, Direction.UP): 'h',  # '\u2b0f',
    Belt(Direction.UP, Direction.LEFT): 't',  # '\u21b0',
    Belt(Direction.LEFT, Direction.DOWN): 'H',  # '\u2b10',
    Belt(Direction.DOWN, Direction.RIGHT): 'T',  # '\u21b3',

    Belt(Direction.RIGHT, Direction.DOWN): 'f',  # '\u2b0e',
    Belt(Direction.DOWN, Direction.LEFT): 'g',  # '\u21b2',
    Belt(Direction.LEFT, Direction.UP): 'F',  # '\u2b11',
    Belt(Direction.UP, Direction.RIGHT): 'G',  # '\u21b1',

    UndergroundBelt(Direction.RIGHT,  True): 'l',  # '⇥',
    UndergroundBelt(Direction.RIGHT, False): 'L',  # '↦',
    UndergroundBelt(Direction.UP,  True): 'i',  # '⤒',
    UndergroundBelt(Direction.UP, False): 'I',  # '↥',
    UndergroundBelt(Direction.LEFT,  True): 'j',  # '⇤',
    UndergroundBelt(Direction.LEFT, False): 'J',  # '↤',
    UndergroundBelt(Direction.DOWN,  True): 'k',  # '⤓',
    UndergroundBelt(Direction.DOWN, False): 'K',  # '↧',

    Splitter(Direction.RIGHT,  True): 'd',  # '⥟',
    Splitter(Direction.RIGHT, False): 'D',  # '⥛',
    Splitter(Direction.UP,  True): 'w',  # '⥜',
    Splitter(Direction.UP, False): 'W',  # '⥠',
    Splitter(Direction.LEFT,  True): 'a',  # '⥚',
    Splitter(Direction.LEFT, False): 'A',  # '⥞',
    Splitter(Direction.DOWN,  True): 's',  # '⥡',
    Splitter(Direction.DOWN, False): 'S',  # '⥝',
}

INV_MAPPING = dict((val, key) for key, val in MAPPING.items())

END_STOP = '┘'


@np.vectorize
def encode_tile(tile):
    return MAPPING.get(tile, 'U')


def decode_char(character):
    return INV_MAPPING[character]


def encode(grid):
    char_grid = np.full((grid.shape[0] + 2, 2 * grid.shape[1] + 1), ' ', dtype='<U1')
    char_grid[1:-1, 1:-1:2] = encode_tile(grid)

    # raw_print(style_seq(bold=True))

    char_grid[0, :] = char_grid[-1, :] = '─'
    char_grid[:, 0] = char_grid[:, -1] = '│'
    char_grid[0, 0] = '┌'
    char_grid[-1, 0] = '└'
    char_grid[0, -1] = '┐'
    char_grid[-1, -1] = END_STOP

    return '\n'.join(''.join(row) for row in char_grid)


def decode(input_lines):
    grid = np.full((len(input_lines) - 2, len(input_lines[0]) // 2), None, dtype=object)

    for row, line in enumerate(input_lines[1:-1]):
        for col, character in enumerate(line[1:-1]):
            if col % 2 != 0:
                if character != ' ':
                    raise RuntimeError('Unexpected non empty column')
                continue
            grid[row, col // 2] = decode_char(character)
    return grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts grids to and from an ascii representation')
    parser.add_argument('mode', choices=['encode', 'decode'])
    args = parser.parse_args()

    if args.mode == 'encode':
        while True:
            tiles = np.array(json.loads(input()))
            tiles = np.vectorize(read_tile)(tiles)
            print(encode(tiles))
    else:
        while True:
            lines = []
            while True:
                lines.append(input().strip())
                if lines[-1].endswith(END_STOP):
                    break
            grid = np.vectorize(write_tile)(decode(lines))
            print(json.dumps(grid.tolist()))
