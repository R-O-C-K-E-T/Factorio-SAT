from typing import *

import inspect, shlex, subprocess, tempfile, sys, json, io
from dataclasses import dataclass

import numpy as np

from pysat.solvers import Solver
from pysat.formula import CNF, IDPool

from ipasir import IPASIRLibrary

from util import *

EDGE_MODE_IGNORE = 'EDGE_IGNORE'
EDGE_MODE_BLOCK  = 'EDGE_BLOCK'
EDGE_MODE_TILE   = 'EDGE_TILE'
IGNORED_TILE     = 'TILE_IGNORE'
BLOCKED_TILE     = 'TILE_BLOCK'

EdgeModeEnumType = Literal['EDGE_IGNORE', 'EDGE_BLOCK', 'EDGE_TILE']
EdgeModeType = Union[Tuple[EdgeModeEnumType, EdgeModeEnumType], EdgeModeEnumType]

def expand_edge_mode(edge_mode: EdgeModeType) -> Tuple[EdgeModeEnumType, EdgeModeEnumType]:
    if isinstance(edge_mode, str):
        return edge_mode, edge_mode
    else:
        return edge_mode

def run_command_solver(cmd: str, clauses: ClauseList) -> Optional[List[LiteralType]]:
    def interpret_solver_answer(stdout):
        result = io.TextIOWrapper(stdout)
        # partials = []
        while True:
            line = result.readline()
            if line.startswith('s'):
                break
            # if line.startswith('c partial'):
            #     pieces = line.split(' ')[2:-1]
            #     partials.append([int(x) for x in pieces])
            
            print(line, file=sys.stderr, end='')
        
        # with open('partials.json', 'w') as f:
        #     json.dump(partials, f)

        if line.startswith('s UNSATISFIABLE'):
            return None
        
        if not line.startswith('s SATISFIABLE'):
            raise RuntimeError('Unknown solution status: ' + line)
        
        model = []
        while True:
            line = result.readline()
            variables = line.split(' ')
            if variables[0] != 'v':
                raise RuntimeError('Solution not returned correctly: ' + line)
            model += [int(v) for v in variables[1:]]
            if model[-1] == 0:
                model.pop()
                break
        return model

    formula = CNF(from_clauses=clauses)
    del clauses
    pieces = shlex.split(cmd)
    if '$FILE' in pieces:
        with tempfile.NamedTemporaryFile('w', suffix='.cnf') as file:
            formula.to_file(file.name)
            del formula
            file.flush()

            pieces = [file.name if piece == '$FILE' else piece for piece in pieces]

            with subprocess.Popen(pieces, stdout=subprocess.PIPE) as process:
                return interpret_solver_answer(process.stdout)
    else:
        with subprocess.Popen(pieces, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
            formula.to_fp(io.TextIOWrapper(process.stdin))
            del formula
            process.stdin.close()
            return interpret_solver_answer(process.stdout)


T = TypeVar('T')
NestedArray = Union[T, List['NestedArray']]

def flatten(tile: Union[NamedTuple, NestedArray[LiteralType]]) -> List[LiteralType]:
    if isinstance(tile, LiteralType):
        return [tile]
    
    if isinstance(tile, NamedTuple):
        tile = tile._asdict().values()
    
    result = []
    for member in tile:
        result += flatten(member)
    return result


InstanceType = TypeVar('InstanceType')
ParsedType = TypeVar('ParsedType')

class Template(Protocol[InstanceType, ParsedType]):
    shape: Tuple[int, ...]
    variable_count: int

    def instantiate(self, pool: IDPool) -> InstanceType:
        ...

    def parse(self, instance: InstanceType, mapping: Dict[int, bool]) -> ParsedType:
        ...

@dataclass(frozen=True)
class BoolTemplate(Template[LiteralType, bool]):
    @property
    def variable_count(self):
        return 1

    def instantiate(self, pool: IDPool) -> LiteralType:
        return pool._next()

    def parse(self, instance: LiteralType, mapping: Dict[int, bool]) -> bool:
        return mapping[instance]

I = TypeVar('I')
P = TypeVar('P')

@dataclass(frozen=True)
class ArrayTemplate(Template[NestedArray[I], NestedArray[P]]):
    component: Template[I, P]
    shape: Tuple[int, ...]

    @property
    def variable_count(self):
        return int(np.product(self.shape)) * self.component.variable_count

    def instantiate(self, pool: IDPool) -> NestedArray[I]:
        composed = np.empty(np.product(self.shape), dtype=object)
        for i in range(len(composed)):
            composed[i] = self.component.instantiate(pool)
        return np.reshape(composed, self.shape).tolist()

    def parse(self, instance: NestedArray[I], mapping: Dict[int, bool]) -> NestedArray[P]:
        assert isinstance(instance, list)
        def recurse(sub_instance: NestedArray[bool], shape: Tuple[int, ...]):
            if len(shape) == 0:
                return self.component.parse(sub_instance, mapping)
            else:
                size, *sub_shape = shape
                return [recurse(sub_instance[i], sub_shape) for i in range(size)]

        return recurse(instance, self.shape)

T = TypeVar('T')
@dataclass(frozen=True)
class SizedTemplate(Template[List[LiteralType], T]):
    size: int

    @property
    def variable_count(self):
        return self.size

    def instantiate(self, pool: IDPool) -> List[LiteralType]:
        return [pool._next() for _ in range(self.size)]

@dataclass(frozen=True)
class ManyHotTemplate(SizedTemplate[List[int]]):
    def parse(self, instance: List[LiteralType], mapping: Dict[int, bool]) -> List[int]:
        assert isinstance(instance, list)
        return [j for j, lit in enumerate(instance) if mapping[lit]]

@dataclass(frozen=True)
class OneHotTemplate(SizedTemplate[Optional[int]]):
    def parse(self, instance: List[LiteralType], mapping: Dict[int, bool]) -> Optional[int]:
        assert isinstance(instance, list)
        for i, lit in enumerate(instance):
            if mapping[lit]:
                return i
        return None

@dataclass(frozen=True)
class NumberTemplate(SizedTemplate[int]):
    is_signed: bool = False

    def parse(self, instance: List[LiteralType], mapping: Dict[int, bool]) -> int:
        assert isinstance(instance, list)
        return read_number([mapping[lit] for lit in instance], self.is_signed)

CompositeTemplateParams = Dict[str, Union[Template[Any, Any], Callable, 'CompositeTemplateParams']]

class CompositeTemplate(Template[NamedTuple, Dict[str, Any]]):
    def __init__(self, template: CompositeTemplateParams):
        self._atomics: Dict[str, Template] = {}
        self._aliases: Dict[str, Callable] = {}
        for name, val in template.items():
            if callable(val):
                self._aliases[name] = val
            elif isinstance(val, dict):
                self._atomics[name] = CompositeTemplate(val)
            else:
                self._atomics[name] = val
        self.variable_count = sum(entry.variable_count for entry in self._atomics.values())

        self.tile_type = collections.namedtuple('CompositeInstance', template.keys(), rename=True)

    def parse(self, instance: NamedTuple, mapping: Dict[int, bool]) -> Dict[str, Any]:
        tile_dict = instance._asdict()
        result = {}

        for name, item_type in self._atomics.items():
            result[name] = item_type.parse(tile_dict[name], mapping)
        
        return result

    def instantiate(self, pool: IDPool) -> NamedTuple:
        members: Dict[str, Any] = {}
        
        for name, item_type in self._atomics.items():
            members[name] = item_type.instantiate(pool)

        for name, function in self._aliases.items():
            if function.__code__.co_flags & inspect.CO_VARKEYWORDS:
                result = function(**members)
            else:
                inputs = set(function.__code__.co_varnames)
                result = function(**{name: val for name, val in members.items() if name in inputs})

            if isinstance(result, np.ndarray):
                result = result.tolist()
            members[name] = result
        return self.tile_type(**members)

    def __repr__(self) -> str:
        return 'CompositeTemplate(' + repr({
            **self._atomics,
            **self._aliases,
        }) + ')'

    __str__ = __repr__

I = TypeVar('I')
P = TypeVar('P')

class BaseGrid(Generic[I, P]):
    def __init__(self, template: Template[I, P], width: int, height: int, pool: Optional[IDPool]=None):
        assert width > 0 and height > 0
        self.template = template
        self.width = width
        self.height = height

        if pool is None:
            self.pool = IDPool()
        else:
            self.pool = pool

        self.tiles = np.frompyfunc(lambda i, j: template.instantiate(self.pool), 2, 1)(*np.ogrid[0:height,0:width])
        
        self.clauses: ClauseList = []

    @property
    def total_variables(self):
        return self.width * self.height * self.template.variable_count

    @property
    def tile_size(self):
        return self.template.variable_count

    def iterate_tiles(self) -> Iterator[I]:
        for x in range(self.width):
            for y in range(self.height):
                yield self.get_tile_instance(x, y)

    def iterate_tile_blocks(self, columnwise_dir: Tuple[int, int], column_count: int, rowwise_dir: Tuple[int, int], row_count: int, edge_mode: EdgeModeType) -> Iterator[np.ndarray]:
        cx, cy = columnwise_dir
        rx, ry = rowwise_dir
        assert abs(cx) + abs(cy) == 1
        assert abs(rx) + abs(ry) == 1
        assert column_count > 0
        assert row_count > 0

        for x in range(self.width):
            for y in range(self.height):
                yield np.frompyfunc(lambda i, j: self.get_tile_instance_offset(x, y, rx*i + cx*j, ry*i + cy*j, edge_mode), 2, 1)(*np.ogrid[0:row_count, 0:column_count])

    def iterate_tile_lines(self, direction: Tuple[int, int], length: int, edge_mode: EdgeModeType) -> Iterator[Sequence[I]]:
        dx, dy = direction
        assert abs(dx) + abs(dy) == 1
        assert length > 0

        for x in range(self.width):
            for y in range(self.height):
                yield np.frompyfunc(lambda i: self.get_tile_instance_offset(x, y, dx*i, dy*i, edge_mode), 1, 1)(np.arange(length))

    def allocate_variable(self) -> LiteralType:
        return self.pool._next()

    def get_tile_instance(self, x: int, y: int) -> I:
        assert x >= 0 and y >= 0 and x < self.width and y < self.height
        return self.tiles[y, x]

    def get_tile_instance_offset(self, x: int, y: int, dx: int, dy: int, edge_mode: EdgeModeType) -> Union[Literal['TILE_IGNORE', 'TILE_BLOCK'], I]:
        edge_mode = expand_edge_mode(edge_mode)
        assert all(mode in (EDGE_MODE_IGNORE, EDGE_MODE_BLOCK, EDGE_MODE_TILE) for mode in edge_mode)
        
        pos = [x + dx, y + dy]
        size = self.width, self.height

        is_ignored = False
        for i in range(2):
            if pos[i] < 0 or pos[i] >= size[i]:
                if edge_mode[i] == EDGE_MODE_TILE:
                    pos[i] = pos[i] % size[i]
                elif edge_mode[i] == EDGE_MODE_BLOCK:
                    return BLOCKED_TILE
                elif edge_mode[i] == EDGE_MODE_IGNORE:
                    is_ignored = True
                else:
                    assert False
        
        if is_ignored:
            return IGNORED_TILE

        return self.get_tile_instance(*pos)

    def parse_solution(self, solution: List[LiteralType]) -> np.ndarray:
        mapping = {abs(lit): lit > 0 for lit in solution}
        return np.frompyfunc(lambda tile: self.template.parse(tile, mapping), 1, 1)(self.tiles)

    def check(self, solver: str='g3'):
        return self.solve(solver) is not None
    
    def solve(self, solver: str='g3'):
        if solver.startswith('cmd:'):
            solution = run_command_solver(solver[4:], self.clauses)
            if solution is None:
                return None
            return self.parse_solution(solution)
        else:
            if solver.startswith('lib:'):
                s = IPASIRLibrary(solver[4:]).create_solver()
                s.add_clauses(self.clauses)
            else:
                s = Solver(name=solver, bootstrap_with=self.clauses)
            
            with s:
                if s.solve():
                    return self.parse_solution(s.get_model())
                else:
                    return None

    def itersolve(self, important_variables=set(), solver: str='g3') -> Iterator[np.ndarray]:
        if solver.startswith('cmd:'):
            solution = run_command_solver(solver[4:], self.clauses)
            if solution is None:
                return
            yield self.parse_solution(solution)
        else:
            if solver.startswith('lib:'):
                s = IPASIRLibrary(solver[4:]).create_solver()
                s.add_clauses(self.clauses)
            else:
                s = Solver(name=solver, bootstrap_with=self.clauses)
            
            with s:
                while s.solve():
                    solution = s.get_model()
                    yield self.parse_solution(solution)
                    
                    s.add_clause([-lit for lit in solution if abs(lit) in important_variables])

    def write(self, filename: str, comments: Optional[List[str]]=None):
        cnf = CNF(from_clauses=self.clauses)
        cnf.to_file(filename, comments)