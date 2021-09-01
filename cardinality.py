from typing import *

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from util import *

# AMO - At Most One

def quadratic_amo(variables: List[VariableType], _: Optional[AllocatorType]=None) -> ClauseList:
    clauses = []
    for i, var_a in enumerate(variables):
        for var_b in variables[:i]:
            clauses.append([-var_a, -var_b])
    return clauses

def logarithmic_amo(variables: List[VariableType], allocator: AllocatorType) -> ClauseList:
    location_variables = [allocator() for _ in range(bin_length(len(variables)))]
    clauses = [list(variables)]
    for i, var in enumerate(variables):
        clauses += implies([var], set_number(i, location_variables))
    return clauses

def heule_amo(variables: List[VariableType], allocator: AllocatorType, recursive_cutoff: int=3) -> ClauseList:
    assert recursive_cutoff >= 3
    if len(variables) <= recursive_cutoff:
        return quadratic_amo(variables)
    else:
        auxilary = allocator()
        middle = len(variables) // 2
        return heule_amo(variables[:middle] + [ auxilary], allocator, recursive_cutoff) + \
               heule_amo(variables[middle:] + [-auxilary], allocator, recursive_cutoff) 


def quadratic_one(variables: List[VariableType], _: Optional[AllocatorType]=None) -> ClauseList:
    return quadratic_amo(variables) + [list(variables)]

def logarithmic_one(variables: List[VariableType], allocator: AllocatorType) -> ClauseList:
    return logarithmic_amo(variables, allocator) + [list(variables)]

def heule_one(variables: List[VariableType], allocator: AllocatorType, recursive_cutoff: int=3) -> ClauseList:
    return heule_amo(variables, allocator, recursive_cutoff) + [list(variables)]

def library_equals(inputs: List[VariableType], n: int, pool: IDPool, encoding=EncType.kmtotalizer) -> ClauseList:
    clauses = CardEnc.equals(inputs, n, vpool=pool, encoding=encoding).clauses
    if len(clauses) == 0:
        raise RuntimeError('Failed to generate clauses')
    return clauses

def library_atmost(inputs: List[VariableType], n: int, pool: IDPool) -> ClauseList:
    clauses = CardEnc.atmost(inputs, n, vpool=pool, encoding=EncType.kmtotalizer).clauses
    if len(clauses) == 0:
        raise RuntimeError('Failed to generate clauses')
    return clauses

def library_atleast(inputs: List[VariableType], n: int, pool: IDPool) -> ClauseList:
    clauses = CardEnc.atleast(inputs, n, vpool=pool, encoding=EncType.kmtotalizer).clauses
    if len(clauses) == 0:
        raise RuntimeError('Failed to generate clauses')
    return clauses