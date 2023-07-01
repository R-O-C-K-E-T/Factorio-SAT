import ctypes
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .util import ClauseList, ClauseType, LiteralType


class IPASIRExternal:
    def __init__(self):
        super().__init__()
        self._solver: Optional['IPASIRSolver'] = None
        self._notify_assignment = None
        self._notify_new_decision_level = None
        self._notify_backtrack = None
        self._cb_decide = None
        self._cb_propagate = None
        self._cb_add_reason_clause_lit = None

    def _connect(self, solver: 'IPASIRSolver'):
        if self._solver is not None:
            raise RuntimeError('Solver already connected')
        self._solver = solver

        notify_assignment_type = solver.lib.ipasir_ext_connect_external_propagator.argtypes[2]
        notify_new_decision_level_type = solver.lib.ipasir_ext_connect_external_propagator.argtypes[3]
        notify_backtrack_type = solver.lib.ipasir_ext_connect_external_propagator.argtypes[4]
        cb_decide_type = solver.lib.ipasir_ext_connect_external_propagator.argtypes[5]
        cb_propagate_type = solver.lib.ipasir_ext_connect_external_propagator.argtypes[6]
        cb_add_reason_clause_lit_type = solver.lib.ipasir_ext_connect_external_propagator.argtypes[7]

        self._notify_assignment = notify_assignment_type(solver.wrap_callback(self.notify_assignment))
        self._notify_new_decision_level = notify_new_decision_level_type(solver.wrap_callback(self.notify_new_decision_level))
        self._notify_backtrack = notify_backtrack_type(solver.wrap_callback(self.notify_backtrack))

        solver.lib.ipasir_ext_connect_external_propagator(
            solver.solver_p,
            None,
            self._notify_assignment,
            self._notify_new_decision_level,
            self._notify_backtrack,
            self._cb_decide or cb_decide_type(0),
            self._cb_propagate or cb_propagate_type(0),
            self._cb_add_reason_clause_lit or cb_add_reason_clause_lit_type(0),
        )

    def _disconnect(self):
        if self._solver is not None:
            raise RuntimeError('Already disconnected')

        self._solver.lib.ipasir_ext_disconnect_external_propagator(self._solver.solver_p)
        self._solver = None
        self._notify_assignment = None
        self._notify_new_decision_level = None
        self._notify_backtrack = None
        self._cb_decide = None
        self._cb_propagate = None
        self._cb_add_reason_clause_lit = None

    def notify_assignment(self, lit: int, is_fixed: bool) -> None:
        raise NotImplementedError

    def notify_new_decision_level(self) -> None:
        raise NotImplementedError

    def notify_backtrack(self, new_level: int) -> None:
        raise NotImplementedError


class IPASIRExternalDecide(IPASIRExternal):
    def _connect(self, solver: 'IPASIRSolver'):
        cb_decide_type = solver.lib.ipasir_ext_connect_external_propagator.argtypes[5]
        self._cb_decide = cb_decide_type(solver.wrap_callback(self.decide, error_return_value=0))

        super()._connect(solver)

    def decide(self) -> int:
        raise NotImplementedError


class IPASIRExternalPropagate(IPASIRExternal):
    def __init__(self):
        super().__init__()

        self._reasons: Dict[int, List[int]] = {}

    def _connect(self, solver: 'IPASIRSolver'):
        cb_propagate_type = solver.lib.ipasir_ext_connect_external_propagator.argtypes[6]
        cb_add_reason_clause_lit_type = solver.lib.ipasir_ext_connect_external_propagator.argtypes[7]

        def propagate_impl():
            result = self.propagate()
            if result is None:
                return 0

            lit, reason = result

            # assert lit != 0
            # assert -lit in reason

            self._reasons[lit] = reason
            return lit
        self._cb_propagate = cb_propagate_type(solver.wrap_callback(propagate_impl, error_return_value=0))

        def reason_impl(lit: int):
            reason = self._reasons[lit]
            if len(reason) == 0:
                return 0
            return reason.pop()
        self._cb_add_reason_clause_lit = cb_add_reason_clause_lit_type(solver.wrap_callback(reason_impl, error_return_value=0))

        super()._connect(solver)

    def propagate(self) -> Optional[Tuple[int, List[int]]]:
        raise NotImplementedError


class IPASIRLibrary:
    def __init__(self, filename: str):
        self.lib = lib = ctypes.cdll.LoadLibrary(filename)

        lib.ipasir_signature.argtypes = []
        lib.ipasir_signature.restype = ctypes.c_char_p

        lib.ipasir_init.argtypes = []
        lib.ipasir_init.restype = ctypes.c_void_p

        lib.ipasir_release.argtypes = [ctypes.c_void_p]
        lib.ipasir_release.restype = None

        lib.ipasir_add.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.ipasir_add.restype = None

        lib.ipasir_assume.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.ipasir_assume.restype = None

        lib.ipasir_solve.argtypes = [ctypes.c_void_p]
        lib.ipasir_solve.restype = ctypes.c_int

        lib.ipasir_val.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.ipasir_val.restype = ctypes.c_int32

        lib.ipasir_failed.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.ipasir_failed.restype = ctypes.c_int

        lib.ipasir_set_terminate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)]
        lib.ipasir_set_terminate.restype = None

        lib.ipasir_set_learn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)),
        ]
        lib.ipasir_set_learn.restype = None

        # lib.ipasir_ext_set_trail.argtypes = [
        #     ctypes.c_void_p,
        #     ctypes.c_void_p,
        #     ctypes.c_int,
        #     ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)),
        # ]
        # lib.ipasir_ext_set_trail.restype = None

        # lib.ipasir_ext_add_redundant.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        # lib.ipasir_ext_add_redundant.restype = None

        lib.ipasir_ext_connect_external_propagator.argtypes = [
            ctypes.c_void_p,  # solver
            ctypes.c_void_p,  # state
            ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool),  # notify_assignment
            ctypes.CFUNCTYPE(None, ctypes.c_void_p),  # notify_new_decision_level
            ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t),  # notify_backtrack
            ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p),  # cb_decide
            ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p),  # cb_propagate
            ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int),  # cb_add_reason_clause_lit
        ]
        lib.ipasir_ext_connect_external_propagator.restype = None
        lib.ipasir_ext_disconnect_external_propagator.argtypes = [
            ctypes.c_void_p,
        ]
        lib.ipasir_ext_disconnect_external_propagator.restype = None

        lib.ipasir_ext_add_observed_var.restype = None
        lib.ipasir_ext_add_observed_var.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.ipasir_ext_remove_observed_var.restype = None
        lib.ipasir_ext_remove_observed_var.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]

    def get_signature(self) -> bytes:
        return self.lib.ipasir_signature()

    def create_solver(self):
        return IPASIRSolver(self.lib)


class IPASIRSolver:
    def __init__(self, lib):
        self.lib = lib
        self.solver_p = lib.ipasir_init()
        self.variables: Set[int] = set()
        self._learn_callback = None
        self._terminate_callback = None
        self._trail_callback = None
        self._propagator: Optional[IPASIRExternal] = None
        self._caught_exception: Optional[BaseException] = None

    def check_closed(self):
        if self.solver_p is None:
            raise RuntimeError('Solver already closed')

    def add_clause(self, clause: ClauseType):
        self.check_closed()

        for lit in clause:
            assert lit != 0
            self.variables.add(abs(lit))
            self.lib.ipasir_add(self.solver_p, lit)
        self.lib.ipasir_add(self.solver_p, 0)

    def add_redundant_clause(self, clause: ClauseType):
        self.check_closed()

        for lit in clause:
            assert lit != 0
            self.variables.add(abs(lit))
            self.lib.ipasir_ext_add_redundant(self.solver_p, lit)
        self.lib.ipasir_ext_add_redundant(self.solver_p, 0)

    def add_dummy_terminate_callback_if_needed(self):
        if self._terminate_callback is not None:
            return

        if self._learn_callback is None and self._trail_callback is None:
            return

        self.set_terminate(lambda: False)

    def set_learn(self, callback: Callable[[ClauseType], None], max_clause_size: int = 2):
        callback_type = self.lib.ipasir_set_learn.argtypes[-1]
        if callback is None:
            raw_callback = callback_type(0)
        else:
            def raw_callback_impl(_, p):
                if self._caught_exception is not None:
                    return
                clause = []
                i = 0
                while True:
                    value = p[i]
                    if value == 0:
                        break
                    clause.append(value)
                    i += 1
                try:
                    callback(clause)
                except BaseException as e:
                    self._caught_exception = e
                    self.add_dummy_terminate_callback_if_needed()
            raw_callback = callback_type(raw_callback_impl)

        self.lib.ipasir_set_learn(self.solver_p, None, max_clause_size, raw_callback)
        self._learn_callback = raw_callback

        # self.add_dummy_terminate_callback_if_needed()

    def set_trail(self, min_decision_prominance: int, callback: Callable[[List[LiteralType]], None]):
        callback_type = self.lib.ipasir_ext_set_trail.argtypes[-1]
        if callback is None:
            raw_callback = callback_type(0)
        else:
            def raw_callback_impl(_, p):
                if self._caught_exception is not None:
                    return
                trail = []
                i = 0
                while True:
                    value = p[i]
                    if value == 0:
                        break
                    trail.append(value)
                    i += 1
                try:
                    callback(trail)
                except BaseException as e:
                    self._caught_exception = e
                    self.add_dummy_terminate_callback_if_needed()
            raw_callback = callback_type(raw_callback_impl)

        self.lib.ipasir_ext_set_trail(self.solver_p, None, min_decision_prominance, raw_callback)
        self._trail_callback = raw_callback

        # self.add_dummy_terminate_callback_if_needed()

    def wrap_callback(self, callback: Callable, error_return_value: Any = None) -> Callable:
        def impl(_, *args):
            if self._caught_exception is not None:
                return error_return_value
            try:
                return callback(*args)
            except BaseException as e:
                self._caught_exception = e
                # Incorrect but works
                self.add_dummy_terminate_callback_if_needed()
                return error_return_value
        return impl

    def set_propagator(self, propagator: Optional[IPASIRExternal]):
        if self._propagator is not None:
            self._propagator._disconnect()
        if propagator is not None:
            propagator._connect(self)
        self._propagator = propagator

        # self.add_dummy_terminate_callback_if_needed()

    def set_terminate(self, callback: Callable[[], bool]):
        callback_type = self.lib.ipasir_set_terminate.argtypes[-1]
        if callback is None:
            raw_callback = callback_type(0)
        else:
            def raw_callback_impl(_):
                if self._caught_exception is not None:
                    return True
                try:
                    return bool(callback())
                except BaseException as e:
                    self._caught_exception = e
                    return True
            raw_callback = callback_type(raw_callback_impl)

        self.lib.ipasir_set_terminate(self.solver_p, None, raw_callback)
        self._terminate_callback = raw_callback

        # self.add_dummy_terminate_callback_if_needed()

    def add_clauses(self, clauses: ClauseList):
        self.check_closed()
        ipasir_add = self.lib.ipasir_add
        solver_p = self.solver_p
        add_var = self.variables.add
        for clause in clauses:
            for lit in clause:
                assert lit != 0
                add_var(abs(lit))
                ipasir_add(solver_p, lit)
            ipasir_add(solver_p, 0)

    def add_observed(self, lit: int):
        self.lib.ipasir_ext_add_observed_var(self.solver_p, lit)

    def remove_observed(self, lit: int):
        self.lib.ipasir_ext_remove_observed_var(self.solver_p, lit)

    def assume(self, lit: LiteralType):
        self.check_closed()

        self.lib.ipasir_assume(self.solver_p, lit)

    def solve(self):
        self.check_closed()

        self._caught_exception = None
        res = self.lib.ipasir_solve(self.solver_p)

        if self._caught_exception is not None:
            raise self._caught_exception

        if res == 0:  # Terminated
            return None
        if res == 10:  # SAT
            return True
        if res == 20:  # UNSAT
            return False
        raise RuntimeError('Unknown solver state: ' + str(res))

    def get_model(self) -> List[LiteralType]:
        self.check_closed()

        model = []
        for var in self.variables:
            value = self.lib.ipasir_val(self.solver_p, var)
            if value == 0:
                value = var
            model.append(value)
        return model

    def unsat_used_assumption(self, lit: LiteralType):
        self.check_closed()

        return bool(self.lib.ipasir_failed(self.solver_p, lit))

    def __enter__(self):
        self.check_closed()
        return self

    def __exit__(self, *_):
        self.check_closed()

        self.lib.ipasir_release(self.solver_p)
        self.solver_p = None

    def __del__(self):
        if self.solver_p is not None:
            self.lib.ipasir_release(self.solver_p)
