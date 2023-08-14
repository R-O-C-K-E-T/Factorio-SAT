import argparse
import asyncio
import concurrent.futures
import json
import math
import os
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple

import numpy as np

from . import belt_balancer
from . import blueprint
from . import optimisations
from .network import deduplicate_network, get_input_output_colours, open_network
from .template import EdgeMode

MAXIMUM_UNDERGROUND_LENGTHS = {
    'normal': 4,
    'fast': 6,
    'express': 8,
}


def factors(value: int) -> Iterator[Tuple[int, int]]:
    for test in reversed(range(1, math.floor(math.sqrt(value)) + 1)):
        if value % test == 0:
            a = test
            b = value // test
            yield a, b
            if a != b:
                yield b, a


def solve_balancer(network, size: Tuple[int, int, int], solver: str):
    maximum_underground_length, width, height = size

    network = deduplicate_network(network)
    grid = belt_balancer.create_balancer(network, width, height, maximum_underground_length)
    grid.prevent_intersection(EdgeMode.NO_WRAP)
    belt_balancer.setup_balancer_ends(grid, network, True, False)

    optimisations.expand_underground(grid, min_x=1, max_x=grid.width - 2)
    optimisations.apply_generic_optimisations(grid)

    belt_balancer.enforce_edge_splitters(grid, network)
    grid.enforce_maximum_underground_length(EdgeMode.NO_WRAP)

    solution = grid.solve(solver)
    if solution is None:
        return None
    return solution.tolist()


class NetworkSolutionStore:
    def __init__(self, network_path: str):
        self.network = open_network(network_path)
        self.network_name = os.path.split(network_path)[1]

        self.exist: Dict[Tuple[int, int, int], bool] = dict()
        self.solutions: Dict[Tuple[int, int, int], Any] = dict()

    @property
    def ordering_key(self):
        match = re.match('(\\d+)[x-](\\d+)(.*)', self.network_name)
        if match is not None:
            return int(match.group(1)), int(match.group(2)), match.group(3)
        else:
            return float('inf'), float('inf'), self.network_name

    def does_balancer_exist(self, size: Tuple[int, int, int]):
        for other_size, exist in self.exist.items():
            if exist:
                if all(d1 >= d2 for d1, d2 in zip(size, other_size)):
                    return True
            else:
                if all(d1 <= d2 for d1, d2 in zip(size, other_size)):
                    return False
        return None

    def clean(self):
        for size, exist in list(self.exist.items()):
            if exist is not False:
                continue
            del self.exist[size]

            if self.does_balancer_exist(size) is not False:
                self.exist[size] = exist

    def from_json(self, data):
        self.exist = {}
        for key, val in data.get('exist', {}).items():
            underground_length, width, height = map(int, key.split(','))
            self.exist[underground_length, width, height] = val

        self.solutions = {}
        for key, val in data.get('solutions', {}).items():
            underground_length, width, height = map(int, key.split(','))
            self.solutions[underground_length, width, height] = val

    def to_json(self):
        return {
            'exist': dict((','.join(map(str, key)), val) for key, val in self.exist.items()),
            'solutions': dict((','.join(map(str, key)), val) for key, val in self.solutions.items())
        }

    def add_solution(self, size: Tuple[int, int, int], solution: Optional[Any]):
        self.exist[size] = solution is not None
        if solution is not None:
            self.solutions[size] = solution

    def best_current_solution(self, loss: Callable[[Tuple[int, int]], Any], underground_length: int):
        found_solutions = ((size, solution) for size, solution in self.solutions.items() if size[0] <= underground_length and loss(size[1:]) != float('inf'))
        return min(found_solutions, key=lambda v: loss(v[0][1:]), default=[None] * 2)[1]


class OptimisationObjective(Protocol):
    def next_size(self, store: NetworkSolutionStore, underground_length: int) -> Optional[Tuple[int, int, int]]:
        ...

    def loss(self, size: Tuple[int, int]) -> Any:
        ...


class LengthObjective(OptimisationObjective):
    def next_size(self, store: NetworkSolutionStore, underground_length: int) -> Optional[Tuple[int, int, int]]:
        (_, input_count), (_, output_count) = get_input_output_colours(store.network)
        height = max(input_count, output_count)

        width = 3
        while True:
            size = underground_length, width, height
            existence = store.does_balancer_exist(size)
            if existence:
                return None

            if existence is None:
                return size
            width += 1

    def loss(self, size: Tuple[int, int]) -> Tuple[int, int]:
        return size[1], size[0]


class AreaObjective(OptimisationObjective):
    def next_size(self, store: NetworkSolutionStore, underground_length: int) -> Optional[Tuple[int, int, int]]:
        (_, input_count), (_, output_count) = get_input_output_colours(store.network)
        min_height = max(input_count, output_count)
        area = min_height
        while True:
            for width, height in factors(area):
                if height < min_height or height > 2 * min_height:
                    continue

                size = underground_length, width + 2, height
                existence = store.does_balancer_exist(size)
                if existence:
                    return None
                if existence is None:
                    return size
            area += 1

    def loss(self, size: Tuple[int, int]) -> int:
        return (size[0] - 2) * size[1]


def get_belt_level(underground_length: int):
    for belt_level, length in sorted(MAXIMUM_UNDERGROUND_LENGTHS.items(), key=lambda i: i[1]):
        if underground_length <= length:
            return belt_level
    return belt_level  # If none work, then just use the biggest


def export_crosstable(stores: List[NetworkSolutionStore], crosstable_filename: str):
    belt_levels = [
        (4, '\U0001f7e8'),
        (6, '\U0001f7e5'),
        (8, '\U0001f7e6'),
    ]
    impossible = '\u274c'
    # unknown = '\u2753'
    unknown = '\u2754'

    preamble = ''
    try:
        with open(crosstable_filename, 'r') as f:
            while True:
                line = f.readline()
                if len(line) == 0 or line.startswith('# Balancers'):
                    break
                preamble += line
    except FileNotFoundError:
        pass

    with open(crosstable_filename, 'w') as f:
        f.write(preamble)
        f.write('# Balancers\n')

        for store in sorted(stores, key=lambda store: store.ordering_key):
            if len(store.exist) == 0:
                continue
            f.write(f'## {store.network_name}\n')
            max_width = max(width for _, width, _ in store.exist) - 2
            max_height = max(height for _, _, height in store.exist)
            (_, input_count), (_, output_count) = get_input_output_colours(store.network)
            min_height = max(input_count, output_count)

            f.write('|     |' + ''.join(f' {i:<3} |' for i in range(1, max_width + 1)) + '\n')
            f.write('|' + '-----|' * (max_width + 1) + '\n')

            for height in range(min_height, max_height + 1):
                f.write(f'| {height:<3} |')
                for width in range(1, max_width + 1):
                    is_known = False
                    for underground_length, character in belt_levels:
                        exists = store.does_balancer_exist((underground_length, width + 2, height))
                        if exists:
                            break
                        if exists is False:
                            is_known = True
                    else:
                        character = impossible if is_known else unknown

                    f.write(f' {character:<3} |')
                f.write('\n')
            f.write('\n')


def main():
    base_path = 'networks'

    parser = argparse.ArgumentParser(description='Calculates optimal balancers')
    parser.add_argument('--database', type=str, default='optimal_balancers.json', help='File for storing/querying results')

    subparsers = parser.add_subparsers(dest='mode', required=True)
    query_parser = subparsers.add_parser('query')
    compute_parser = subparsers.add_parser('compute')
    export_crosstable_parser = subparsers.add_parser('export-crosstable')

    for subparser in (query_parser, compute_parser):
        subparser.add_argument('underground_length', type=int, help='Maximum underground length')
        subparser.add_argument('objective', choices=['area', 'length'], help='Optimisation objective')

    query_parser.add_argument('--export-blueprints', action='store_true', help='Return query results as blueprints')
    query_parser.add_argument('--allow-imperfect', action='store_true', help='Return balancers that are not known to be optimal')

    compute_parser.add_argument('--threads', type=int, help='Number of compute threads')
    compute_parser.add_argument('--solver', type=str, default='g4', help='Backend SAT solver to use')

    export_crosstable_parser.add_argument('filename', type=str, help='Name of file to export crosstable markdown as')
    args = parser.parse_args()

    result_file: str = args.database

    if 'objective' in args:
        if args.objective == 'area':
            args.objective = AreaObjective()
        elif args.objective == 'length':
            args.objective = LengthObjective()
        else:
            assert False

    stores: List[NetworkSolutionStore] = []
    for file in os.listdir(base_path):
        stores.append(NetworkSolutionStore(os.path.join(base_path, file)))

    try:
        with open(result_file) as f:
            data = json.load(f)
        for store in stores:
            item = data.get(store.network_name)
            if item is None:
                continue
            store.from_json(item)
    except FileNotFoundError:
        pass

    def save_progress():
        data = {}
        for store in stores:
            data[store.network_name] = store.to_json()
        with open(result_file, 'w') as f:
            json.dump(data, f)

    if args.mode == 'query':
        if args.export_blueprints:
            def encode_solution(solution, name):
                label = ' to '.join(name.split('x'))
                tiles = np.array(solution)
                for i, row in enumerate(tiles):
                    for j, entry in enumerate(row):
                        tiles[i, j] = blueprint.read_tile(entry)
                return blueprint.encode_blueprint(blueprint.make_blueprint(tiles, label, get_belt_level(args.underground_length)))
        else:
            def encode_solution(solution, _):
                return json.dumps(solution)

        for store in sorted(stores, key=lambda store: store.ordering_key):
            if not args.allow_imperfect and args.objective.next_size(store, args.underground_length) is not None:
                continue

            solution = store.best_current_solution(args.objective.loss, args.underground_length)
            if solution is None:
                continue

            print(encode_solution(solution, store.network_name))
    elif args.mode == 'compute':
        async def optimise(executor: concurrent.futures.ProcessPoolExecutor, store: NetworkSolutionStore):
            while True:
                next_size = args.objective.next_size(store, args.underground_length)
                if next_size is None:
                    break
                print(f'{store.network_name}: Start {next_size}')
                solution = await loop.run_in_executor(executor, solve_balancer, store.network, next_size, args.solver)

                store.add_solution(next_size, solution)
                store.clean()

                save_progress()

            print(f'{store.network_name}: Solution found')

        async def main():
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
                tasks = [optimise(executor, store) for store in stores]
                await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(main())
    elif args.mode == 'export-crosstable':
        export_crosstable(stores, args.filename)
    else:
        assert False


if __name__ == '__main__':
    main()
