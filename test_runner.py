from dataclasses import dataclass
from collections import namedtuple
import re
from template import EdgeMode
from solver import Grid
from typing import *

from network import open_network, deduplicate_network
import stringifier, belt_balancer, optimisations

TEST_EDGE_START  = '┌'
TEST_EDGE_MIDDLE = '│'
TEST_EDGE_STOP   = '└'
TEST_EDGES = {TEST_EDGE_START, TEST_EDGE_MIDDLE, TEST_EDGE_STOP}
SUITE_FILENAME = 'test_suite'

def remap_characters(filename: str, mapping: Dict[str, str]):
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
        underground_length = int(self.params.get('underground-length', 4))

        edges = self.params.get('edges', 'ignore')
        edge_mode = EdgeMode.WRAP if edges == 'tile' else EdgeMode.NO_WRAP

        if 'network' in self.params:
            assert edges == 'ignore'
            network = open_network(self.params['network'])
            network = deduplicate_network(network)
            grid = belt_balancer.create_balancer(network, self.tiles.shape[1], self.tiles.shape[0], underground_length)
        else:
            grid = Grid(self.tiles.shape[1], self.tiles.shape[0], 1, underground_length)
            grid.prevent_bad_undergrounding(edge_mode)

        rule = self.params.get('rule')
        if rule is not None:
            if rule == 'expand-underground':
                optimisations.expand_underground(grid)
            elif rule == 'prevent-mergeable-underground':
                optimisations.prevent_mergeable_underground(grid, edge_mode)
            elif rule == 'glue-splitters':
                optimisations.glue_splitters(grid)
            elif rule == 'prevent-belt-hooks':
                optimisations.prevent_belt_hooks(grid, edge_mode)
            elif rule == 'prevent-small-loops':
                optimisations.prevent_small_loops(grid)
            elif rule == 'prevent-semicircles':
                optimisations.prevent_semicircles(grid, edge_mode)
            elif rule == 'prevent-underground-hook':
                optimisations.prevent_underground_hook(grid, edge_mode)
            elif rule == 'prevent-zigzags':
                optimisations.prevent_zigzags(grid, edge_mode)
            elif rule == 'break-symmetry':
                optimisations.break_vertical_symmetry(grid)
            elif rule == 'prevent-belt-parallel-splitter':
                optimisations.prevent_belt_parallel_splitter(grid, edge_mode)
            elif rule == 'glue-partial-splitters':
                optimisations.glue_partial_splitters(grid, edge_mode)
            else:
                raise RuntimeError(f'Unknown rule "{rule}"')

        grid.enforce_maximum_underground_length(edge_mode)
        
        grid.prevent_intersection(edge_mode)
        if edges == 'block':
            grid.block_belts_through_edges()
            grid.block_underground_through_edges()
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

@dataclass(frozen=True)
class Token:
    name: str
    lineno: int
    data: Any

def tokenise(lines: List[str]) -> List[Token]:
    line_iter = enumerate(lines)
    tokenised: List[Token] = []

    block_start = re.compile('^\\[([^\\[\\]]*)\\](\\([^\\(\\)]*\\))?\\s*(\\+|-)?\\s*{$')

    for row, line in line_iter:
        if not line.startswith(TEST_EDGE_START):
            if len(line) > 0 and line[0] in TEST_EDGES:
                raise RuntimeError(f'Invalid line start character "{line[0]}" at line {row+1}')
            
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
                tokenised.append(Token('open', row + 1, (match[1], params, is_positive)))
            elif line == '}':
                tokenised.append(Token('close', row + 1, None))
        else:
            test_case = [line]
            while True:
                row, line = next(line_iter)
                test_case.append(line)
                if not line.startswith(TEST_EDGE_MIDDLE):
                    break
            if not line.startswith(TEST_EDGE_STOP):
                raise RuntimeError(f'Invalid test instance format at line {row+1}')
            tokenised.append(Token('test', row + 1, stringifier.decode(test_case)))
    return tokenised

def open_suite(filename: str):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    tokens = tokenise(lines)

    stack = []
    stack_entry = namedtuple('stack_entry', ['label', 'params', 'is_positive', 'test_cases', 'sub_blocks'])

    result = TestSuite(None, [], [])
    for token in tokens:
        if token.name == 'open':
            label, params, is_positive = token.data
            if len(stack) > 0:
                if is_positive is None:
                    is_positive = stack[-1].is_positive
                params = {**stack[-1].params, **params}

            stack.append(stack_entry(label, params, is_positive, [], []))
        elif token.name == 'close':
            if len(stack) == 0:
                raise RuntimeError(f'Block end token at line {token.lineno} does not have a corresponding test block')
            last = stack.pop()
            test_block = TestSuite(last.label, last.sub_blocks, last.test_cases)
            if len(stack) == 0:
                result.sub_suites.append(test_block)
            else:
                stack[-1].sub_blocks.append(test_block)
        elif token.name == 'test':
            if len(stack) == 0:
                raise RuntimeError(f'Test at line {token.lineno} does not have a surrounding test block')
            last = stack[-1]
            if last.is_positive is None:
                raise RuntimeError(f'Test at line {token.lineno} does is neither positive nor negative')
            last.test_cases.append(TestCase(len(last.test_cases) + 1, last.params, last.is_positive, token.data))
    
    return result

if __name__ == '__main__':
    #remap_characters(SUITE_FILENAME, {})
    suite = open_suite(SUITE_FILENAME)
    _, failed = suite.run()
    if failed != 0:
        exit(1)