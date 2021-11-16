from typing import *
from ctypes import *

from util import LiteralType, ClauseType, ClauseList

class IPASIRLibrary:
    def __init__(self, filename: str):
        self.lib = lib = cdll.LoadLibrary(filename)

        lib.ipasir_signature.argtypes = []
        lib.ipasir_signature.restype = c_char_p

        lib.ipasir_init.argtypes = []
        lib.ipasir_init.restype = c_void_p

        lib.ipasir_release.argtypes = [c_void_p]
        lib.ipasir_release.restype = None

        lib.ipasir_add.argtypes = [c_void_p, c_int32]
        lib.ipasir_add.restype = None

        lib.ipasir_assume.argtypes = [c_void_p, c_int32]
        lib.ipasir_assume.restype = None

        lib.ipasir_solve.argtypes = [c_void_p]
        lib.ipasir_solve.restype = c_int

        lib.ipasir_val.argtypes = [c_void_p, c_int32]
        lib.ipasir_val.restype = c_int32

        lib.ipasir_failed.argtypes = [c_void_p, c_int32]
        lib.ipasir_failed.restype = c_int

        lib.ipasir_set_terminate.argtypes = [c_void_p, c_void_p, CFUNCTYPE(c_int, c_void_p)]
        lib.ipasir_set_terminate.restype = None

        lib.ipasir_set_learn.argtypes = [c_void_p, c_void_p, c_int, CFUNCTYPE(None, c_void_p, POINTER(c_int32))]
        lib.ipasir_set_learn.restype = None

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

    def check_closed(self):
        if self.solver_p is None:
            raise RuntimeError('Solver already closed')

    def add_clause(self, clause: ClauseType):
        self.check_closed()
        
        for lit in clause:
            self.variables.add(abs(lit))
            self.lib.ipasir_add(self.solver_p, lit)
        self.lib.ipasir_add(self.solver_p, 0)

    def set_learn(self, callback: Callable[[ClauseType], None], max_clause_size: int=2):
        callback_type = self.lib.ipasir_set_learn.argtypes[-1]
        if callback is None: 
            raw_callback = callback_type(0)
        else:
            def raw_callback_impl(_, p):
                clause = []
                i = 0
                while True:
                    value = p[i]
                    if value == 0:
                        break
                    clause.append(value)
                    i += 1
                callback(clause)
            raw_callback = callback_type(raw_callback_impl)
            
        self.lib.ipasir_set_learn(self.solver_p, None, max_clause_size, raw_callback)
        self._learn_callback = raw_callback

    def set_terminate(self, callback: Callable[[], bool]):
        callback_type = self.lib.ipasir_set_terminate.argtypes[-1]
        if callback is None:
            raw_callback = callback_type(0)
        else:
            def raw_callback_impl(_):
                return bool(callback())
            raw_callback = callback_type(raw_callback_impl)
        
        self.lib.ipasir_set_terminate(self.solver_p, None, raw_callback)
        self._terminate_callback = raw_callback

    def add_clauses(self, clauses: ClauseList):
        self.check_closed()
        ipasir_add = self.lib.ipasir_add
        solver_p = self.solver_p
        add_var = self.variables.add
        for clause in clauses:
            for lit in clause:
                add_var(abs(lit))
                ipasir_add(solver_p, lit)
            ipasir_add(solver_p, 0)
    
    def assume(self, lit: LiteralType):
        self.check_closed()
        
        self.lib.ipasir_assume(self.solver_p, lit)
    
    def solve(self):
        self.check_closed()
        
        res = self.lib.ipasir_solve(self.solver_p)
        if res == 0: # Terminated
            return None
        if res == 10: # SAT
            return True
        if res == 20: # UNSAT
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