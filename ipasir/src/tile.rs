use crate::direction::Direction;
use pyo3::exceptions::PyTypeError;
use pyo3::FromPyObject;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tile {
    Empty,
    Belt {
        input: Direction,
        output: Direction,
    },
    Underground {
        direction: Direction,
        is_input: bool,
    },
    Splitter {
        direction: Direction,
        is_head: bool,
    },
}

impl<'source> FromPyObject<'source> for Direction {
    fn extract(ob: &'source pyo3::PyAny) -> pyo3::PyResult<Self> {
        let index = ob.call_method0("__index__")?;
        let index: usize = index.extract()?;
        match index {
            0 => Ok(Direction::Right),
            1 => Ok(Direction::Up),
            2 => Ok(Direction::Left),
            3 => Ok(Direction::Down),
            _ => Err(PyTypeError::new_err(format!(
                "Invalid direction (index={})",
                index
            ))),
        }
    }
}

impl<'source> FromPyObject<'source> for Tile {
    fn extract(ob: &'source pyo3::PyAny) -> pyo3::PyResult<Self> {
        // TODO Import the types and check them against this type instead of string matching
        let type_name = ob.get_type().name()?;

        match type_name {
            "EmptyTile" => Ok(Tile::Empty),
            "Belt" => Ok(Tile::Belt {
                input: ob.getattr("input_direction")?.extract()?,
                output: ob.getattr("output_direction")?.extract()?,
            }),
            "UndergroundBelt" => Ok(Tile::Underground {
                direction: ob.getattr("direction")?.extract()?,
                is_input: ob.getattr("is_input")?.extract()?,
            }),
            "Splitter" => Ok(Tile::Splitter {
                direction: ob.getattr("direction")?.extract()?,
                is_head: ob.getattr("is_head")?.extract()?,
            }),
            _ => Err(PyTypeError::new_err(format!(
                "Unknown tile type (name={})",
                type_name
            ))),
        }
    }
}

impl Tile {
    pub fn input_direction(&self) -> Option<Direction> {
        match *self {
            Tile::Empty => None,
            Tile::Belt { input, .. } => Some(input),
            Tile::Underground {
                is_input: false, ..
            } => None,
            Tile::Underground {
                is_input: true,
                direction,
            } => Some(direction),
            Tile::Splitter { direction, .. } => Some(direction),
        }
    }

    pub fn output_direction(&self) -> Option<Direction> {
        match self {
            Tile::Empty => None,
            Tile::Belt { output, .. } => Some(*output),
            Tile::Underground {
                is_input: false,
                direction,
            } => Some(*direction),
            Tile::Underground { is_input: true, .. } => None,
            Tile::Splitter { direction, .. } => Some(*direction),
        }
    }
}
