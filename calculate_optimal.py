from asyncio.tasks import gather
import json, math, time, os, asyncio, argparse
import concurrent.futures
from functools import partial
import numpy as np

from util import EDGE_MODE_BLOCK, EDGE_MODE_IGNORE
from network import open_network, get_input_output_colours
import belt_balancer, blueprint

MAXIMUM_UNDERGROUND_LENGTHS = {
    'normal'  : 4,
    'fast'    : 6,
    'express' : 8,
}

def factors(value):
    for test in reversed(range(1, math.floor(math.sqrt(value)) + 1)):
        if value % test == 0:
            a = test
            b = value // test
            yield a, b
            if a != b:
                yield b, a


def get_offsets(height, input_size, output_size):
    if input_size > output_size:
        for start_offset, end_offset in get_offsets(height, output_size, input_size):
            yield end_offset, start_offset
        return
    
    for end_offset in range(height - output_size + 1):
        for i in range(output_size - input_size + 1):
            yield i + end_offset, end_offset

def solve_balancer(network, size, solver):
    maximum_underground_length, width, height = size

    grid = belt_balancer.create_balancer(network, width, height)
    grid.prevent_intersection((EDGE_MODE_IGNORE, EDGE_MODE_BLOCK))
    belt_balancer.setup_balancer_ends(grid, network, True)
    belt_balancer.enforce_edge_splitters(grid, network)
    grid.set_maximum_underground_length(maximum_underground_length, EDGE_MODE_BLOCK)
    grid.prevent_empty_along_underground(maximum_underground_length, EDGE_MODE_BLOCK)
    
    solution = grid.solve(solver)
    if solution is None:
        return None
    return solution.tolist()


class NetworkSolutionStore:
    def __init__(self, network_path):
        self.network = open_network(network_path)
        self.network_name = os.path.split(network_path)[1]

        self.exist = dict()
        self.solutions = dict()

    def does_balancer_exist(self, size):
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
            if exist != False:
                continue
            del self.exist[size]

            if self.does_balancer_exist(size) != False:
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

    def add_solution(self, size, solution):
        self.exist[size] = solution is not None
        if solution is not None:
            self.solutions[size] = solution

    def best_current_solution(self, loss, underground_length):
        return min(((size, solution) for size, solution in self.solutions.items() if size[0] <= underground_length and loss(size[1:]) != float('inf')), key=lambda v: loss(v[0][1:]), default=[None]*2)[1]

    def next_length_size(self, underground_length):
        (_, input_count), (_, output_count) = get_input_output_colours(self.network)
        height = max(input_count, output_count)

        width = 3
        while True:
            size = underground_length, width, height
            existence = self.does_balancer_exist(size)
            if existence:
                return None

            if existence is None:
                return size
            width += 1

    def next_area_size(self, underground_length):
        (_, input_count), (_, output_count) = get_input_output_colours(self.network)
        min_height = max(input_count, output_count)
        area = min_height
        while True:
            for width, height in factors(area):
                if height < min_height or height > 2*min_height:
                    continue

                size = underground_length, width + 2, height
                existence = self.does_balancer_exist(size)
                if existence:
                    return None
                if existence is None:
                    return size
            area += 1

def get_belt_level(underground_length: int):
    for belt_level, length in sorted(MAXIMUM_UNDERGROUND_LENGTHS.items(), key=lambda i: i[1]):
        if underground_length <= length:
            return belt_level
    return belt_level # If none work, then just use the biggest

if __name__ == '__main__':
    base_path = 'networks'
    result_file = 'optimal_balancers.json'

    parser = argparse.ArgumentParser(description='Calculates optimal balancers')
    parser.add_argument('mode', choices=['query', 'compute'])
    parser.add_argument('underground_length', type=int, help='Maximum underground length')
    parser.add_argument('objective', choices=['area', 'length'], help='Optimisation objective')
    parser.add_argument('--export-blueprints', action='store_true', help='Return query results as blueprints')
    parser.add_argument('--threads', type=int, help='Number of compute threads')
    parser.add_argument('--solver', type=str, default='g4', help='Backend SAT solver to use')
    args = parser.parse_args()

    stores = []
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
            encode_solution = lambda solution, _: json.dumps(solution)

        if args.objective == 'length':
            get_next_size = store.next_length_size
        elif args.objective == 'area':
            get_next_size = store.next_area_size
        
        
        for store in sorted(stores, key=lambda store: [int(s) for s in store.network_name.split('x')]):
            if get_next_size(args.underground_length) is not None:
                continue

            if args.objective == 'length':
                (_, input_count), (_, output_count) = get_input_output_colours(store.network)
                min_height = max(input_count, output_count)
                loss = lambda size: size[0] if size[1] == min_height else float('inf')
            elif args.objective == 'area':
                loss = lambda size: (size[0]-2) * size[1]
            else:
                assert False
            
            solution = store.best_current_solution(loss, args.underground_length)
            if solution is None:
                continue

            print(encode_solution(solution, store.network_name))
    elif args.mode == 'compute':
        async def optimise(executor, store):
            next_size = None
            while True:
                if args.objective == 'length':
                    next_size = store.next_length_size(args.underground_length)
                elif args.objective == 'area':
                    next_size = store.next_area_size(args.underground_length)
                else:
                    assert False
                
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

        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    else:
        assert False