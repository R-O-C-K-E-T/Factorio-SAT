import argparse
import json
import sys

import numpy as np

from blueprint import read_tile, write_tile_simple
from tile import Belt, Splitter, UndergroundBelt


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
    None: ' ',
    Belt(0, 0): '→',
    Belt(1, 1): '↑',
    Belt(2, 2): '←',
    Belt(3, 3): '↓',

    Belt(0, 1): 'h',  # '\u2b0f',
    Belt(1, 2): 't',  # '\u21b0',
    Belt(2, 3): 'H',  # '\u2b10',
    Belt(3, 0): 'T',  # '\u21b3',

    Belt(0, 3): 'f',  # '\u2b0e',
    Belt(3, 2): 'g',  # '\u21b2',
    Belt(2, 1): 'F',  # '\u2b11',
    Belt(1, 0): 'G',  # '\u21b1',

    UndergroundBelt(0,  True): 'l',  # '⇥',
    UndergroundBelt(0, False): 'L',  # '↦',
    UndergroundBelt(1,  True): 'i',  # '⤒',
    UndergroundBelt(1, False): 'I',  # '↥',
    UndergroundBelt(2,  True): 'j',  # '⇤',
    UndergroundBelt(2, False): 'J',  # '↤',
    UndergroundBelt(3,  True): 'k',  # '⤓',
    UndergroundBelt(3, False): 'K',  # '↧',

    Splitter(0,  True): 'd',  # '⥟',
    Splitter(0, False): 'D',  # '⥛',
    Splitter(1,  True): 'w',  # '⥜',
    Splitter(1, False): 'W',  # '⥠',
    Splitter(2,  True): 'a',  # '⥚',
    Splitter(2, False): 'A',  # '⥞',
    Splitter(3,  True): 's',  # '⥡',
    Splitter(3, False): 'S',  # '⥝',
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
            for y, row in enumerate(tiles):
                for x, entry in enumerate(row):
                    tiles[y, x] = read_tile(entry)

            print(encode(tiles))
    else:
        while True:
            lines = []
            while True:
                lines.append(input().strip())
                if lines[-1].endswith(END_STOP):
                    break
            grid = decode(lines)
            for y, row in enumerate(grid):
                for x, tile in enumerate(row):
                    grid[y, x] = write_tile_simple(tile)
            print(json.dumps(grid.tolist()))
