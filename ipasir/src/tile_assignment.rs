use crate::array_2d::Array2D;
use crate::coord_2d::Coord2D;
use crate::direction::Direction;
use crate::sat::{LiteralAssignment, TileLiterals};
use crate::tile::Tile;

pub struct TileAssignment<'a> {
    literal_assignment: &'a LiteralAssignment,
    tile_templates: &'a Array2D<TileLiterals>,
    tile_assignment: &'a Array2D<Option<Tile>>,
}

impl<'a> TileAssignment<'a> {
    pub fn new(
        literal_assignment: &'a LiteralAssignment,
        tile_templates: &'a Array2D<TileLiterals>,
        tile_assignment: &'a Array2D<Option<Tile>>,
    ) -> TileAssignment<'a> {
        TileAssignment {
            literal_assignment,
            tile_templates,
            tile_assignment,
        }
    }

    pub fn next(&self, location: &PathLocation) -> Option<PathLocation> {
        if let Some(direction) = location.underground_direction {
            let next_coord = location.coord + direction.delta();
            let next_tile = self.tile_assignment.at(next_coord)?;

            if let Some(Tile::Underground {
                direction: tile_direction,
                is_input,
            }) = next_tile
            {
                if *tile_direction == direction && !is_input {
                    return Some(PathLocation {
                        coord: next_coord,
                        underground_direction: None,
                    });
                }
            }
            return Some(PathLocation {
                coord: next_coord,
                underground_direction: Some(direction),
            });
        }

        let current_tile = self.tile_assignment[location.coord].as_ref()?;

        if let Tile::Underground {
            direction,
            is_input,
        } = current_tile
        {
            if *is_input {
                return Some(PathLocation {
                    coord: location.coord + direction.delta(),
                    underground_direction: Some(*direction),
                });
            }
        }

        let direction = current_tile.output_direction()?;
        let next_coord = location.coord + direction.delta();

        if !self.tile_assignment.valid_index(next_coord) {
            return None;
        }

        Some(PathLocation {
            coord: next_coord,
            underground_direction: None,
        })
    }

    pub fn prev(&self, location: &PathLocation) -> Option<PathLocation> {
        if let Some(direction) = location.underground_direction {
            let prev_coord = location.coord - direction.delta();
            let prev_tile = self.tile_assignment.at(prev_coord)?;

            if let Some(Tile::Underground {
                direction: tile_direction,
                is_input,
            }) = prev_tile
            {
                if *tile_direction == direction && *is_input {
                    return Some(PathLocation {
                        coord: prev_coord,
                        underground_direction: None,
                    });
                }
            }
            return Some(PathLocation {
                coord: prev_coord,
                underground_direction: Some(direction),
            });
        }

        let current_tile = self.tile_assignment[location.coord].as_ref()?;

        if let Tile::Underground {
            direction,
            is_input,
        } = current_tile
        {
            if !is_input {
                return Some(PathLocation {
                    coord: location.coord - direction.delta(),
                    underground_direction: Some(*direction),
                });
            }
        }

        let direction = current_tile.input_direction()?;
        let prev_coord = location.coord - direction.delta();

        if !self.tile_assignment.valid_index(prev_coord) {
            return None;
        }

        Some(PathLocation {
            coord: prev_coord,
            underground_direction: None,
        })
    }

    pub fn get_underground(&self, coord: Coord2D<isize>, direction: Direction) -> Option<bool> {
        self.tile_templates[coord].get_underground(self.literal_assignment, direction)
    }

    pub fn get_tile(&self, coord: Coord2D<isize>) -> Option<&Tile> {
        self.tile_assignment[coord].as_ref()
    }

    pub fn get_literals(&self, coord: Coord2D<isize>) -> &TileLiterals {
        &self.tile_templates[coord]
    }

    pub fn conflict_path(&self, path: &[PathLocation]) -> Box<[i32]> {
        let mut clause: Vec<i32> = Vec::new();

        for location in path {
            let tile_template = &self.tile_templates[location.coord];

            if let Some(direction) = location.underground_direction {
                tile_template.conflict_underground(&mut clause, direction);
            } else if let Some(tile) = self.tile_assignment[location.coord].as_ref() {
                tile_template.conflict_tile(&mut clause, tile)
            } else {
                panic!("Shouldn't be empty right??");
            }
        }

        clause.into()
    }

    pub fn valid_index(&self, coord: Coord2D<isize>) -> bool {
        self.tile_assignment.valid_index(coord)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathLocation {
    pub coord: Coord2D<isize>,
    pub underground_direction: Option<Direction>,
}

impl PathLocation {
    pub fn new(coord: Coord2D<isize>, underground_direction: Option<Direction>) -> PathLocation {
        PathLocation {
            coord,
            underground_direction,
        }
    }
}
