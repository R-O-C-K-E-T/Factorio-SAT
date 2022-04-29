class BaseTile:
    def __init__(self, input_direction=None, output_direction=None):
        self.input_direction = input_direction
        self.output_direction = output_direction


class Belt(BaseTile):
    def __init__(self, input_direction: int, output_direction: int):
        assert (input_direction - output_direction) % 4 != 2
        super().__init__(input_direction, output_direction)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Belt):
            return False

        return self.input_direction == other.input_direction and self.output_direction == other.output_direction

    def __hash__(self):
        return hash((self.input_direction, self.output_direction))

    def __str__(self):
        return 'Belt({}, {})'.format(self.input_direction, self.output_direction)
    __repr__ = __str__


class UndergroundBelt(BaseTile):
    def __init__(self, direction: int, is_input: bool):
        self.direction = direction
        self.is_input = is_input

        if is_input:
            super().__init__(direction, None)
        else:
            super().__init__(None, direction)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, UndergroundBelt):
            return False

        return self.direction == other.direction and self.is_input == other.is_input

    def __hash__(self):
        return hash((self.direction, self.is_input))

    def __str__(self):
        return 'UndergroundBelt({}, {})'.format(self.direction, self.is_input)
    __repr__ = __str__


class Splitter(BaseTile):
    def __init__(self, direction: int, is_head: bool):
        self.direction = direction
        self.is_head = is_head

        super().__init__(direction, direction)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Splitter):
            return False

        return self.direction == other.direction and self.is_head == other.is_head

    def __hash__(self):
        return hash((self.direction, self.is_head))

    def __str__(self):
        return 'Splitter({}, {})'.format(self.direction, self.is_head)
    __repr__ = __str__


class Inserter(BaseTile):
    def __init__(self, direction: int, type: int):
        self.direction = direction
        self.type = type  # 0 -> Normal, 1 -> Long

        super().__init__()

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Inserter):
            return False

        return self.direction == other.direction and self.type == other.type

    def __hash__(self):
        return hash((self.direction, self.type))

    def __str__(self):
        return 'Inserter({}, {})'.format(self.direction, self.type)
    __repr__ = __str__


class AssemblingMachine(BaseTile):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        super().__init__()

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, AssemblingMachine):
            return False

        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return 'AssemblingMachine({}, {})'.format(self.x, self.y)
    __repr__ = __str__


BELT_TILES = [Belt(direction, (direction + curve) % 4) for direction in range(4) for curve in range(-1, 2)]
UNDERGROUND_TILES = [UndergroundBelt(direction, type) for direction in range(4) for type in range(2)]
SPLITTER_TILES = [Splitter(direction, i) for direction in range(4) for i in range(2)]
ALL_TILES = [None] + BELT_TILES + UNDERGROUND_TILES + SPLITTER_TILES
