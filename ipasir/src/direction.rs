use crate::coord_2d::Coord2D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Right,
    Up,
    Left,
    Down,
}

impl TryFrom<usize> for Direction {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Right),
            1 => Ok(Self::Up),
            2 => Ok(Self::Left),
            3 => Ok(Self::Down),
            _ => Err(()),
        }
    }
}

impl Into<usize> for Direction {
    fn into(self) -> usize {
        match self {
            Self::Right => 0,
            Self::Up => 1,
            Self::Left => 2,
            Self::Down => 3,
        }
    }
}

impl Direction {
    pub fn iterator() -> impl Iterator<Item = Direction> {
        [Self::Right, Self::Up, Self::Left, Self::Down]
            .iter()
            .copied()
    }

    pub fn delta(&self) -> Coord2D<isize> {
        match self {
            Self::Right => Coord2D { y: 0, x: 1 },
            Self::Up => Coord2D { y: -1, x: 0 },
            Self::Left => Coord2D { y: 0, x: -1 },
            Self::Down => Coord2D { y: 1, x: 0 },
        }
    }

    pub fn next(&self) -> Direction {
        match self {
            Direction::Right => Direction::Up,
            Direction::Up => Direction::Left,
            Direction::Left => Direction::Down,
            Direction::Down => Direction::Right,
        }
    }

    pub fn prev(&self) -> Direction {
        match self {
            Direction::Right => Direction::Down,
            Direction::Up => Direction::Right,
            Direction::Left => Direction::Up,
            Direction::Down => Direction::Left,
        }
    }
}
