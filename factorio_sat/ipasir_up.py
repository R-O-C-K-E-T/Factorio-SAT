import json
import sys
import time
from typing import Callable, List, Optional, Set, Tuple

import numpy as np

from .direction import Direction

from .coord import Coord
from . import stringifier
from . import optimisations

from .trail_logger import EliminationMap, PathLocation, PathTracer, determine_tile, find_shortcuts

from .solver import Grid

from .network import deduplicate_network, open_network
from .ipasir import IPASIRExternalPropagate
from . import tile as tile_type
from . import belt_balancer
from . import interchange
from . import blueprint

import ipasir as ipasir_rs


class TilePropagator:
    def __init__(self, grid: Grid, add_propagate: Callable[[Tuple[int, List[int]]], None]) -> None:
        super().__init__()
        self.grid = grid
        self.add_propagate = add_propagate
        self.map = EliminationMap(grid)
        self._tile_levels: List[Optional[Tuple[int, int]]] = []
        self._tile_assignment: np.ndarray = np.full((grid.height, grid.width), None, dtype=object)
        self._underground_levels: List[Optional[Tuple[int, int, int]]] = []
        self._underground_assignment = np.full((grid.height, grid.width, 4), False, dtype=bool)

        self._show = False

    def notify_assignment(self, lit: int, assignment: Set[int]) -> None:
        label = self.map.label(lit)
        if label is None:
            return

        idx = label.location[::-1]
        tile_template = self.grid.get_tile_instance(idx[1], idx[0])
        # if is_fixed and self._current_level != 0:
        #     print(label, self._current_level, file=sys.stderr)
        #     print(stringifier.encode(self._tile_assignment), file=sys.stderr)
        # assert not is_fixed or self._current_level == 0 # TODO

        if self._tile_assignment[idx] is None:
            tile = determine_tile(assignment, tile_template)

            if tile is not None:
                self._tile_assignment[idx] = tile
                self._tile_levels.append(idx)

                self.notify_new_location(lit, PathLocation(Coord(idx[1], idx[0]), None))

        for i, underground_lit in enumerate(tile_template.underground):
            if underground_lit == lit:
                u_idx = *idx, i
                self._underground_assignment[u_idx] = True
                self._underground_levels.append(u_idx)

                self.notify_new_location(lit, PathLocation(Coord(idx[1], idx[0]), Direction(i)))
                break

    def notify_new_location(self, lit: int, location: PathLocation):
        tracer = PathTracer(self._tile_assignment, self._underground_assignment)

        loop = tracer.get_loop(location)
        if loop is not None:
            # print('loop', loop, file=sys.stderr)
            clause = tracer.conflict_path(self.grid, loop)
            # if -lit not in clause:
            #     print(stringifier.encode(self._tile_assignment), file=sys.stderr)
            #     print(self.map.label(lit), '[' + ','.join(str(self.map.label(c)) for c in clause) + ']', file=sys.stderr)
            #     clause.append(-lit)
            # assert -lit in clause
            if -lit in clause:
                self.add_propagate((-lit, clause))
            else:
                self.add_propagate((clause[0], clause))
            return

        full_path = [*list(tracer.trace_backward(location))[::-1], *tracer.trace_forward(tracer.next(location))]

        if not tracer.on_path(full_path[0]):
            del full_path[0]

        if len(full_path) > 0 and not tracer.on_path(full_path[-1]):
            del full_path[-1]

        for shortcut in find_shortcuts(tracer, full_path):
            # print('full_path', full_path, file=sys.stderr)
            # print('shortcut', shortcut, file=sys.stderr)
            # print(stringifier.encode(self._tile_assignment), file=sys.stderr)
            # exit(1)
            self._show = True
            clause = tracer.conflict_path(self.grid, shortcut)
            # if -lit not in clause:
            #     print(stringifier.encode(self._tile_assignment), file=sys.stderr)
            #     print(self.map.label(lit), [str(self.map.label(c)) for c in clause], file=sys.stderr)
            # assert -lit in clause, f'{lit}, {clause}'
            if -lit in clause:
                self.add_propagate((-lit, clause))
            else:
                self.add_propagate((clause[0], clause))

    def notify_new_decision_level(self) -> None:
        self._tile_levels.append(None)
        self._underground_levels.append(None)

    def backtrack_single_level(self):
        tile_loc = self._tile_levels.pop()
        while tile_loc is not None:
            self._tile_assignment[tile_loc] = None
            tile_loc = self._tile_levels.pop()

        underground_loc = self._underground_levels.pop()
        while underground_loc is not None:
            self._underground_assignment[underground_loc] = False
            underground_loc = self._underground_levels.pop()

    def before_backtrack(self):
        # if not self._show:
        if True:
            return
        self._show = False

        tile_grid = np.full((self.grid.height, self.grid.width), None)
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                tile_instance = self._tile_assignment[y, x]

                tile = blueprint.write_tile(tile_instance or tile_type.EmptyTile())
                tile['underground'] = self._underground_assignment[y, x].tolist()
                tile_grid[y, x] = tile

        print(json.dumps(tile_grid.tolist()))


class Propagator(IPASIRExternalPropagate):
    def __init__(self, grid: Grid):
        super().__init__()
        self.assignment: Set[int] = set()
        self._levels: List[int] = []
        self._current_level = 0
        self._to_propagate: List[Tuple[int, List[int]]] = []

        self.tile_propagator = TilePropagator(grid, self._to_propagate.append)
        self._stashed_lits: List[int] = []

    @property
    def current_level(self):
        return self._current_level

    def notify_assignment(self, lit: int, is_fixed: bool) -> None:
        if is_fixed:
            self._levels.insert(0, lit)
        else:
            self._levels.append(lit)

        assert lit not in self.assignment
        assert -lit not in self.assignment
        self.assignment.add(lit)

        if is_fixed and self._current_level != 0:
            self._stashed_lits.append(lit)
            # for _ in range(self._current_level):
            #     self.tile_propagator.backtrack_single_level()

            # partial_assignment = set()
            # level = 0
            # for lit in self._levels:
            #     if lit == 0:
            #         level += 1
            #         self.tile_propagator.notify_new_decision_level()
            #         continue

            #     partial_assignment.add(lit)
            #     if level > 0:
            #         self.tile_propagator.notify_assignment(lit, partial_assignment)
        else:
            self.tile_propagator.notify_assignment(lit, self.assignment)

    def notify_new_decision_level(self) -> None:
        self._current_level += 1
        self._levels.append(0)
        self.tile_propagator.notify_new_decision_level()

    def notify_backtrack(self, new_level: int) -> None:
        self.tile_propagator.before_backtrack()
        self._to_propagate.clear()

        while self._current_level > new_level:
            lit = self._levels.pop()
            while lit != 0:
                self.assignment.remove(lit)
                lit = self._levels.pop()

            self._current_level -= 1

            self.tile_propagator.backtrack_single_level()

        if new_level == 0:
            for lit in self._stashed_lits:
                self.tile_propagator.notify_assignment(lit, self.assignment)
            self._stashed_lits.clear()

    def propagate(self) -> Optional[Tuple[int, List[int]]]:
        if len(self._to_propagate) > 0:
            return self._to_propagate.pop()
            self._to_propagate.clear()
            return None
        return None


if __name__ == '__main__':
    # solver_lib = IPASIRLibrary('/home/nathan/SAT_Solvers/cadical/build/libcadical.so')

    # solver = solver_lib.create_solver()

    if False:
        network = open_network('networks/4x4')
        network = deduplicate_network(network)
        grid = belt_balancer.create_balancer(network, 10, 4, 4)
        grid.prevent_intersection()
        grid.enforce_maximum_underground_length()
        grid.block_belts_through_edges((False, True))
        belt_balancer.setup_balancer_ends(grid, network, True, False)
    else:
        # grid = interchange.create_grid(10, 8, 4, True)
        # grid = interchange.create_grid(11, 30, 4, True)
        # UNSAT old 89.30845069885254, new 84.62783432006836,
        # old optimised 53.20967721939087, new optimised 90.33s
        grid = interchange.create_grid(11, 26, 4, True)  # SAT old 11.739898443222046, new 44.937527894973755
        # grid = interchange.create_grid(10, 26, 4, True) # UNSAT old 4.228454113006592, new 8.130547523498535
        interchange.require_correct_transport_through_edges(grid)
        interchange.prevent_awkward_underground_entry(grid)
        interchange.prevent_passing(grid)

    optimisations.prevent_empty_along_underground(grid)
    optimisations.expand_underground(grid, min_x=1, max_x=grid.width - 2)

    optimisations.apply_generic_optimisations(grid)

    # optimisations.expand_underground(grid)

    solver = ipasir_rs.FactorioSolver(grid.tiles.tolist())
    # solver = Solver(name='cadical153')

    # grid.clauses.append([grid.get_tile_instance(3, 3).underground[2]])

    for clause in grid.clauses:
        solver.add_clause(clause)

    # solver.add_clause(clause)

    t0 = time.time()
    outcome = solver.solve()
    t1 = time.time()
    print(outcome, t1 - t0)
    if outcome:
        solution = grid.parse_solution(solver.get_model())
        solution = np.vectorize(blueprint.read_tile)(solution)
        print(stringifier.encode(solution))
    exit()

    # solver.add_clauses(grid.clauses)

    # lits = list(range(1, 10))
    # clauses = quadratic_one(lits)

    solver.set_propagator(Propagator(grid))

    for tile in grid.iterate_tiles():
        for lit in [*tile.all_direction, *tile.type, tile.is_splitter_head, *tile.underground]:
            solver.add_observed(lit)

    # clauses.append([-9])

    important_lits = set(abs(lit) for tile in grid.iterate_tiles() for lit in [*tile.all_direction, *tile.underground, *tile.type, tile.is_splitter_head])

    t0 = time.time()
    outcome = solver.solve()
    t1 = time.time()
    print(outcome, t1 - t0, file=sys.stderr)
    # exit()

    # while outcome:
    #     model = solver.get_model()
    #     # print(json.dumps(grid.parse_solution(model).tolist()))
    #     solution = grid.parse_solution(model)
    #     print(json.dumps({
    #         'metadata': {
    #         },
    #         'template': grid.template.write(),
    #         'tiles': solution.tolist()
    #     }))
    #     sys.stdout.flush()

    #     solver.add_clause([-lit for lit in model if abs(lit) in important_lits])

    #     outcome = solver.solve()
    #     print(outcome, file=sys.stderr)
