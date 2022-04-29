import json

import numpy as np

if __name__ == '__main__':
    while True:
        solution = np.array(json.loads(input()))

        solution = np.rot90(solution)
        for tile in solution.reshape(-1):
            for key in ('input_direction', 'output_direction'):
                direction = tile[key]
                if direction is not None:
                    tile[key] = (direction + 1) % 4

        print(json.dumps(solution.tolist()))
