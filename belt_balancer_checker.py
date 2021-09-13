import argparse, json, sys
import numpy as np

from util import *
from network import open_network, deduplicate_network
import belt_balancer, blueprint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Checks that the provided balancer meshes with the provided network')
    parser.add_argument('network', type=argparse.FileType('r'), help='Splitter network')
    parser.add_argument('--no-output', action='store_true', help='Run without printing colourised balancer')
    parser.add_argument('--underground-length', type=int, default=4, help='Sets the maximum length of underground section (excludes ends)')
    args = parser.parse_args()

    tiles = np.array(json.loads(input()))
    for i, row in enumerate(tiles):
        for j, entry in enumerate(row):
            tiles[i, j] = blueprint.read_tile(entry)

    network = open_network(args.network)
    args.network.close()

    network = deduplicate_network(network)

    grid = belt_balancer.create_balancer(network, tiles.shape[0], tiles.shape[1])
    for y in range(grid.height):
        for x in range(grid.width):
            grid.set_tile(x, y, tiles[x,y])

    print(len(grid.clauses), file=sys.stderr)
    grid.set_maximum_underground_length(args.underground_length, EDGE_MODE_BLOCK)
    grid.prevent_intersection(EDGE_MODE_IGNORE)

    solution = grid.solve()
    if solution is not None:
        if not args.no_output:
            print(json.dumps(solution.tolist()))
    else:
        exit(1)