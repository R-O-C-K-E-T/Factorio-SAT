from dataclasses import dataclass
from collections import namedtuple
import re
from util import EDGE_MODE_BLOCK, EDGE_MODE_IGNORE
from solver import Grid
from typing import *

from network import open_network, deduplicate_network
import stringifier, belt_balancer

TEST_EDGE_START  = '┌'
TEST_EDGE_MIDDLE = '│'
TEST_EDGE_STOP   = '└'
TEST_EDGES = {TEST_EDGE_START, TEST_EDGE_MIDDLE, TEST_EDGE_STOP}
SUITE_FILENAME = 'test_suite'

def remap_characters(filename, mapping):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    new_lines = []
    for line in lines:
        if line[0] in TEST_EDGES:
            line = ''.join(mapping.get(character, character) for character in line)
        new_lines.append(line)
    
    with open(filename + '.new', 'w') as f:
        for line in new_lines:
            f.write(line)
            f.write('\n')

@dataclass
class TestCase:
    index: int
    params: Dict[str, str]
    is_positive: bool
    tiles: Any

    def run(self):
        if 'network' in self.params:
            network = open_network(self.params['network'])
            network = deduplicate_network(network)
            grid = belt_balancer.create_balancer(network, self.tiles.shape[1], self.tiles.shape[0])
        else:
            grid = Grid(self.tiles.shape[1], self.tiles.shape[0], 1)
            grid.prevent_bad_undergrounding(EDGE_MODE_BLOCK)
        grid.prevent_intersection(EDGE_MODE_IGNORE)
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                grid.set_tile(x, y, tile)
        return grid.check() == self.is_positive

@dataclass
class TestSuite:
    label: str
    sub_suites: List['TestSuite']
    cases: List[TestCase]

    def run(self, indent=-1):
        passed = 0
        failed = 0

        if self.label is not None:
            print('  ' * indent + self.label)
        for suite in self.sub_suites:
            sub_passed, sub_failed = suite.run(indent + 1)
            passed += sub_passed
            failed += sub_failed
        
        fail_list = []
        for i, test in enumerate(self.cases):
            if test.run():
                passed += 1
            else:
                failed += 1
                fail_list.append(i + 1)

        failed_str = ''
        if len(fail_list) != 0:
            failed_str = ' (' + ','.join(map(str, fail_list)) + ')'
        print('  ' * (indent+1) + f'{passed} passed, {failed} failed' + failed_str)
        return passed, failed

def tokenise(lines):
    line_iter = enumerate(lines)
    tokenised = []

    block_start = re.compile('^\\[([^\\[\\]]*)\\](\\([^\\(\\)]*\\))?\\s*(\\+|-)?\\s*{$')

    for col, line in line_iter:
        if not line.startswith(TEST_EDGE_START):
            if len(line) > 0 and line[0] in TEST_EDGES:
                raise RuntimeError(f'Invalid line start character "{line[0]}" at line {col+1}')
            
            match = block_start.match(line)
            if match is not None:
                if match[3] == '+':
                    is_positive = True
                elif match[3] == '-':
                    is_positive = False
                else:
                    is_positive = None
                params = {}
                if match[2] is not None:
                    for entry in match[2][1:-1].split(','):
                        key, val = entry.strip().split('=')
                        params[key.strip()] = val.strip()
                tokenised.append(('open', (match[1], params, is_positive)))
            elif line == '}':
                tokenised.append(('close', None))
        else:
            test_case = [line]
            while True:
                col, line = next(line_iter)
                test_case.append(line)
                if not line.startswith(TEST_EDGE_MIDDLE):
                    break
            if not line.startswith(TEST_EDGE_STOP):
                raise RuntimeError(f'Invalid test instance format at line {col+1}')
            tokenised.append(('test', stringifier.decode(test_case)))
    return tokenised

def open_suite(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    tokens = tokenise(lines)

    stack = []
    stack_entry = namedtuple('stack_entry', ['label', 'params', 'is_positive', 'test_cases', 'sub_blocks'])

    result = TestSuite(None, [], [])
    for name, data in tokens:
        if name == 'open':
            label, params, is_positive = data
            if is_positive is None:
                is_positive = stack[-1].is_positive
            stack.append(stack_entry(label, params, is_positive, [], []))
        elif name == 'close':
            last = stack.pop()
            test_block = TestSuite(last.label, last.sub_blocks, last.test_cases)
            if len(stack) == 0:
                result.sub_suites.append(test_block)
            else:
                stack[-1].sub_blocks.append(test_block)
        elif name == 'test':
            last = stack[-1]
            last.test_cases.append(TestCase(len(last.test_cases) + 1, last.params, last.is_positive, data))
    
    return result

if __name__ == '__main__':
    #remap_characters(SUITE_FILENAME, {})
    suite = open_suite(SUITE_FILENAME)
    _, failed = suite.run()
    if failed != 0:
        exit(1)