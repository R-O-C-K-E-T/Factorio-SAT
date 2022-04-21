import argparse, json, sys
import numpy as np
from template import EdgeMode

from util import *
from network import open_network, deduplicate_network
import belt_balancer, blueprint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Checks that the provided balancer meshes with the provided network')
    parser.add_argument('network', type=argparse.FileType('r'), help='Splitter network')
    parser.add_argument('--no-output', action='store_true', help='Run without printing colourised balancer (only exit code)')
    parser.add_argument('--underground-length', type=int, default=4, help='Sets the maximum length of underground section (excludes ends)')
    args = parser.parse_args()

    tiles = np.array(json.loads(input()))
    for y, row in enumerate(tiles):
        for x, entry in enumerate(row):
            tiles[y, x] = blueprint.read_tile(entry)

    network = open_network(args.network)
    args.network.close()

    network = deduplicate_network(network)

    grid = belt_balancer.create_balancer(network, tiles.shape[1], tiles.shape[0], args.underground_length)
    for y in range(grid.height):
        for x in range(grid.width):
            grid.set_tile(x, y, tiles[y, x])

    print(len(grid.clauses), file=sys.stderr)
    grid.enforce_maximum_underground_length(EdgeMode.NO_WRAP)
    grid.prevent_intersection(EdgeMode.NO_WRAP)

    solution = grid.solve()
    if solution is not None:
        if not args.no_output:
            print(json.dumps(solution.tolist()))
    else:
        exit(1)