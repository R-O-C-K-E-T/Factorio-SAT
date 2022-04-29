import collections
import math
import traceback
from os import path
from typing import Callable, Iterator, List, Optional, Tuple

LiteralType = int
ClauseType = List[LiteralType]
ClauseList = List[ClauseType]
AllocatorType = Callable[[], int]


def get_stack():  # Doesn't include caller
    result = []
    for entry in traceback.extract_stack()[:-2]:
        result.append((entry.filename, entry.lineno))
    return tuple(result)


class StackTracingList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traces = collections.defaultdict(lambda: 0)

    def __iadd__(self, other):
        self.traces[get_stack()] += len(other)
        return super().__iadd__(other)

    def append(self, value):
        self.traces[get_stack()] += 1
        super().append(value)

    def profile(self):
        longest_file = max(len(path.basename(file)) for stack in self.traces for file, _ in stack)
        longest_lineno = max(lineno for stack in self.traces for _, lineno in stack)
        deepest_stack = max(len(stack) for stack in self.traces)

        format = ('{:>' + str(longest_file) + '}:{:<' + str(len(str(longest_lineno))) + '}').format

        trace_length = len(format('', '')) * deepest_stack + 2 * (deepest_stack - 1)

        for stack, count in sorted(self.traces.items(), key=lambda v: v[1], reverse=True):
            trace = ', '.join(format(path.basename(file), lineno) for file, lineno in stack)
            print(trace + ' ' * (trace_length - len(trace)) + ' - ' + str(count))


def add_numbers(
        input_a: List[LiteralType],
        input_b: List[LiteralType],
        output: List[LiteralType],
        allocator: AllocatorType,
        carry_in: Optional[LiteralType] = None,
        allow_overflow=False) -> ClauseList:
    assert len(input_a) == len(input_b)
    assert len(output) in (len(input_a), len(input_a) + 1)

    clauses = []
    for in_a, in_b, out in zip(input_a, input_b, output):
        carry_out = allocator()
        if carry_in is None:
            clauses += [
                [-in_a, -in_b, carry_out],
                [in_a, -carry_out],
                [in_b, -carry_out],

                [in_a,  in_b, -out],
                [-in_a,  in_b,  out],
                [in_a, -in_b,  out],
                [-in_a, -in_b, -out],
            ]
        else:
            clauses += [
                [-in_a, -in_b,     carry_out],
                [-in_a, -carry_in, carry_out],
                [-in_b, -carry_in, carry_out],

                [in_a, in_b,     -carry_out],
                [in_a, carry_in, -carry_out],
                [in_b, carry_in, -carry_out],

                [in_a,  in_b,  carry_in, -out],
                [-in_a,  in_b,  carry_in,  out],
                [in_a, -in_b,  carry_in,  out],
                [-in_a, -in_b,  carry_in, -out],
                [in_a,  in_b, -carry_in,  out],
                [-in_a,  in_b, -carry_in, -out],
                [in_a, -in_b, -carry_in, -out],
                [-in_a, -in_b, -carry_in,  out],
            ]

        carry_in = carry_out

    if len(output) > len(input_a):
        clauses += literals_same(carry_in, output[-1])
    elif not allow_overflow:
        clauses += [[-carry_in]]
    return clauses


def sum_numbers(numbers: List[List[LiteralType]], output: List[LiteralType], allocator: AllocatorType, allow_overflow=False) -> ClauseList:
    assert len(numbers) > 1

    size = len(numbers[0])
    assert all(len(number) == size for number in numbers) and size == len(output)

    clauses = []

    number_in = numbers[0]
    for i, number in enumerate(numbers[1:]):
        if i == len(numbers) - 2:
            number_out = output
        else:
            number_out = [allocator() for _ in range(size)]

        clauses += add_numbers(number_in, number, number_out, allocator, allow_overflow=allow_overflow)

        number_in = number_out

    return clauses


def increment_number(input: List[LiteralType], output: List[LiteralType]):
    assert len(input) == len(output)
    assert len(input) > 0

    clauses = []
    for i, (in_lit, out_lit) in enumerate(zip(input, output)):
        clauses += implies(input[:i], literals_different(in_lit, out_lit))

        for lit in input[:i]:
            clauses += implies([-lit], literals_same(in_lit, out_lit))
    return clauses


def get_popcount(bits: List[LiteralType], output: List[LiteralType], allocator: AllocatorType) -> ClauseList:
    assert bin_length(len(bits) + 1) == len(output)
    assert len(bits) > 1

    clauses = []
    if len(bits) <= 3:
        carry_in = bits[2] if len(bits) == 3 else None
        clauses += add_numbers([bits[0]], [bits[1]], output, allocator, carry_in)
    else:
        carry_in = bits[-1] if len(bits) % 2 != 0 else None
        sub_size = len(bits) // 2

        output_a = [allocator() for _ in range(len(output) - 1)]
        output_b = [allocator() for _ in range(len(output) - 1)]

        clauses += get_popcount(bits[:sub_size], output_a, allocator)
        clauses += get_popcount(bits[sub_size:(2 * sub_size)], output_b, allocator)
        clauses += add_numbers(output_a, output_b, output, allocator, carry_in)

    return clauses


def read_number(bits: List[bool], signed=False):
    result = 0
    for i, bit in enumerate(bits):
        if bit:
            result |= 1 << i

    if signed:
        assert len(bits) > 1
        if bits[-1]:  # Two's complement
            result = result - (1 << len(bits))

    return result


def direction_to_vec(direction: int) -> Tuple[int, int]:
    return [(1, 0), (0, -1), (-1, 0), (0, 1)][direction]


def bin_length(value: int):
    return math.ceil(math.log2(value))


def set_literal(lit: LiteralType, value: bool) -> LiteralType:
    if value:
        return lit
    else:
        return -lit


def set_all_false(literals: List[LiteralType]) -> ClauseList:
    return [[-lit] for lit in literals]


def set_all_true(literals: List[LiteralType]) -> ClauseList:
    return [[lit] for lit in literals]


def set_number(value: int, literals: List[LiteralType]) -> ClauseList:
    assert value < (1 << len(literals))

    clauses = []
    for lit, bit in zip(literals, get_bits(value, len(literals))):
        clauses.append([set_literal(lit, bit)])
    return clauses


def set_numbers(value_a: int, value_b: int, literals_a: List[LiteralType], literals_b: List[LiteralType]) -> ClauseList:
    # One set of variables is set to value_a, the other is set to value_b
    assert len(literals_a) == len(literals_b)
    total_bits = len(literals_a)
    assert value_a < (1 << total_bits)
    assert value_b < (1 << total_bits)

    clauses = []
    differences = []
    for lit_a, lit_b, bit_a, bit_b in zip(literals_a, literals_b, get_bits(value_a, total_bits), get_bits(value_b, total_bits)):
        if bit_a == bit_b:
            clauses.append([set_literal(lit_a, bit_a)])
            clauses.append([set_literal(lit_b, bit_a)])
        else:
            clauses += literals_different(lit_a, lit_b)
            differences.append((lit_a, lit_b, bit_a))

    if len(differences) != 0:
        lit_a0, lit_b0, bit_a0 = differences[0]
        # clauses += literals_different(lit_a0, lit_b0)
        for lit_a1, lit_b1, bit_a1 in differences[1:]:
            if bit_a0 == bit_a1:  # Bits are correlated
                clauses += literals_same(lit_a0, lit_a1)
                # clauses += literals_different(lit_a0, lit_b1)
            else:  # Anti-correlated
                clauses += literals_different(lit_a0, lit_a1)
                # clauses += literals_same(lit_a0, lit_b1)

    return clauses


def set_numbers_equal(number_a: List[LiteralType], number_b: List[LiteralType], allow_different_lengths: bool = False) -> ClauseList:
    clauses = []

    if allow_different_lengths:
        clauses += set_all_false(number_a[len(number_b):])
        clauses += set_all_false(number_b[len(number_a):])
    else:
        assert len(number_a) == len(number_b)

    for lit_a, lit_b in zip(number_a, number_b):
        clauses += literals_same(lit_a, lit_b)
    return clauses


def set_not_number(value: int, literals: List[LiteralType]) -> ClauseType:
    return [-lit[0] for lit in set_number(value, literals)]


def set_maximum(value: int, literals: List[LiteralType]) -> ClauseList:
    if len(literals) == 0:
        assert value == 0
        return []

    tail = literals[1:]
    clauses = set_maximum(value >> 1, tail)
    if not (value & 1):
        clause = [-literals[0]]
        for bit, lit in zip(get_bits(value >> 1, len(tail)), tail):
            if bit:
                clause.append(-lit)
        clauses.append(clause)
    return clauses


def invert_number(input: List[LiteralType], output: List[LiteralType], allocator: AllocatorType):
    assert len(input) == len(output)

    clauses = []

    carry_in = None
    for i, (lit_a, lit_b) in enumerate(zip(input, output)):
        if i == len(input) - 1:
            carry_out = None
        else:
            carry_out = allocator()

        if carry_in is None:
            clauses += literals_same(lit_a, lit_b)
            if carry_out is not None:
                clauses += [
                    [-lit_a, -lit_b, carry_out],

                    [lit_a, -carry_out],
                    [lit_b, -carry_out],
                ]
        else:
            clauses += [
                [-lit_a, -lit_b, -carry_in],

                [-lit_a, +lit_b, +carry_in],
                [+lit_a, -lit_b, +carry_in],
                [+lit_a, +lit_b, -carry_in],
            ]

            if carry_out is not None:
                clauses += [
                    [-lit_a, -lit_b,    carry_out],
                    [-lit_a, -carry_in, carry_out],
                    [-lit_b, -carry_in, carry_out],

                    [lit_a, lit_b,    -carry_out],
                    [lit_a, carry_in, -carry_out],
                    [lit_b, carry_in, -carry_out],
                ]
        carry_in = carry_out

    clauses.append(input[:-1] + [-input[-1]])

    return clauses


def is_power_of_two(value):
    return not (value & (value - 1))


def get_bits(value: int, total_bits: int) -> Iterator[bool]:
    # Result is little-endian
    for bit in range(total_bits):
        yield bool(value & (1 << bit))


def implies(condition: List[LiteralType], consequences: ClauseList) -> ClauseList:
    # If all the input variables are the correct value then the consequences must be satisfied
    inverse_condition = [-lit for lit in condition]
    return [inverse_condition + consequence for consequence in consequences]


def literals_different(lit_a: LiteralType, lit_b: LiteralType) -> ClauseList:
    return [[lit_a, lit_b], [-lit_a, -lit_b]]


def literals_same(lit_a: LiteralType, lit_b: LiteralType) -> ClauseList:
    return [[-lit_a, lit_b], [lit_a, -lit_b]]


def invert_components(clause: ClauseType) -> ClauseType:
    # Converts c0 OR c1 OR c2 OR ... to NOT (c0 AND c1 AND c2 AND ...)
    return [-lit for lit in clause]


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def make_allocator(initial: int) -> AllocatorType:
    value = initial

    def allocator():
        nonlocal value
        value += 1
        return value
    return allocator


def make_fixed_allocator(literals: List[LiteralType]) -> AllocatorType:
    iterator = iter(literals)
    return lambda: next(iterator)


def break_symmetry(left: List[LiteralType], right: List[LiteralType], allocator: AllocatorType) -> ClauseList:
    assert len(left) == len(right)
    differences = [allocator() for _ in left]

    clauses = []
    for left_lit, right_lit, diff, prev_diff in zip(left, right, differences, [None] + differences):
        if prev_diff is None:
            clauses += [
                [left_lit, -right_lit,  diff],
                [-left_lit, -right_lit, -diff],
                [left_lit,  right_lit, -diff],
                [-left_lit,  right_lit],
            ]
        else:
            clauses += [
                [-prev_diff, diff],
                [left_lit, -right_lit,  diff],
                [prev_diff, -left_lit, -right_lit, -diff],
                [prev_diff,  left_lit,  right_lit, -diff],
                [prev_diff, -left_lit,  right_lit],
            ]
    return clauses


__all__ = [
    'AllocatorType',
    'ClauseList',
    'ClauseType',
    'LiteralType',
    'bin_length',
    'break_symmetry',
    'direction_to_vec',
    'get_popcount',
    'implies',
    'invert_components',
    'is_power_of_two',
    'literals_different',
    'literals_same',
    'read_number',
    'set_all_false',
    'set_all_true',
    'set_literal',
    'set_maximum',
    'set_not_number',
    'set_number',
    'set_numbers',
    'set_numbers_equal',
]
