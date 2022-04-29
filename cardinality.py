from typing import List, Optional

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool

from util import AllocatorType, ClauseList, LiteralType, bin_length, implies, set_number

# AMO - At Most One


def quadratic_amo(literals: List[LiteralType], _: Optional[AllocatorType] = None) -> ClauseList:
    clauses = []
    for i, lit_a in enumerate(literals):
        for lit_b in literals[:i]:
            clauses.append([-lit_a, -lit_b])
    return clauses


def logarithmic_amo(literals: List[LiteralType], allocator: AllocatorType) -> ClauseList:
    location_literals = [allocator() for _ in range(bin_length(len(literals)))]
    clauses = [list(literals)]
    for i, lit in enumerate(literals):
        clauses += implies([lit], set_number(i, location_literals))
    return clauses


def heule_amo(literals: List[LiteralType], allocator: AllocatorType, recursive_cutoff: int = 3) -> ClauseList:
    assert recursive_cutoff >= 3
    if len(literals) <= recursive_cutoff:
        return quadratic_amo(literals)
    else:
        auxilary = allocator()
        middle = len(literals) // 2
        return heule_amo(literals[:middle] + [auxilary], allocator, recursive_cutoff) + \
            heule_amo(literals[middle:] + [-auxilary], allocator, recursive_cutoff)


def quadratic_one(literals: List[LiteralType], _: Optional[AllocatorType] = None) -> ClauseList:
    return quadratic_amo(literals) + [list(literals)]


def logarithmic_one(literals: List[LiteralType], allocator: AllocatorType) -> ClauseList:
    return logarithmic_amo(literals, allocator) + [list(literals)]


def heule_one(literals: List[LiteralType], allocator: AllocatorType, recursive_cutoff: int = 3) -> ClauseList:
    return heule_amo(literals, allocator, recursive_cutoff) + [list(literals)]


def library_equals(inputs: List[LiteralType], n: int, pool: IDPool, encoding=EncType.kmtotalizer) -> ClauseList:
    clauses = CardEnc.equals(inputs, n, vpool=pool, encoding=encoding).clauses
    if len(clauses) == 0:
        raise RuntimeError('Failed to generate clauses')
    return clauses


def library_atmost(inputs: List[LiteralType], n: int, pool: IDPool) -> ClauseList:
    clauses = CardEnc.atmost(inputs, n, vpool=pool, encoding=EncType.kmtotalizer).clauses
    if len(clauses) == 0:
        raise RuntimeError('Failed to generate clauses')
    return clauses


def library_atleast(inputs: List[LiteralType], n: int, pool: IDPool) -> ClauseList:
    clauses = CardEnc.atleast(inputs, n, vpool=pool, encoding=EncType.kmtotalizer).clauses
    if len(clauses) == 0:
        raise RuntimeError('Failed to generate clauses')
    return clauses
