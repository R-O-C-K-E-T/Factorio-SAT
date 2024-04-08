use arrayvec::ArrayVec;
use cadical::{Callbacks, Conflict};
use pyo3::FromPyObject;

use crate::{
    array_2d::Array2D,
    coord_2d::Coord2D,
    tile_assignment::{PathLocation, TileAssignment},
    Direction, Tile,
};

#[derive(Debug, Clone)]
pub struct TileLiterals {
    input_direction: [i32; 4],
    output_direction: [i32; 4],
    is_splitter: i32,
    is_splitter_head: i32,
    underground: [i32; 4],
}

impl TileLiterals {
    pub fn conflict_underground(&self, clause: &mut impl Extend<i32>, direction: Direction) {
        let idx: usize = direction.into();
        clause.extend(Some(-self.underground[idx]));
    }

    pub fn conflict_tile(&self, clause: &mut impl Extend<i32>, tile: &Tile) {
        match tile {
            Tile::Underground {
                direction,
                is_input,
            } => {
                let idx: usize = direction.clone().into();
                if *is_input {
                    clause.extend([
                        -self.input_direction[idx],
                        self.output_direction[0],
                        self.output_direction[1],
                        self.output_direction[2],
                        self.output_direction[3],
                    ])
                } else {
                    clause.extend([
                        -self.output_direction[idx],
                        self.input_direction[0],
                        self.input_direction[1],
                        self.input_direction[2],
                        self.input_direction[3],
                    ])
                }
            }
            Tile::Belt { input, output } => {
                let input_idx: usize = input.clone().into();
                let output_idx: usize = output.clone().into();
                clause.extend([
                    -self.input_direction[input_idx],
                    -self.output_direction[output_idx],
                ])
            }
            Tile::Splitter {
                direction: _,
                is_head: _,
            } => {
                todo!("Splitters aren't real")
            }
            Tile::Empty => {
                // Yikes
                clause.extend([
                    self.input_direction[0],
                    self.input_direction[1],
                    self.input_direction[2],
                    self.input_direction[3],
                    self.output_direction[0],
                    self.output_direction[1],
                    self.output_direction[2],
                    self.output_direction[3],
                ])
            }
        };
    }

    pub fn get_literals(&self) -> impl Iterator<Item = i32> {
        let mut literals: Vec<i32> = Vec::with_capacity(4 + 4 + 1 + 1 + 4);
        literals.extend(self.input_direction);
        literals.extend(self.output_direction);
        literals.push(self.is_splitter);
        literals.push(self.is_splitter_head);
        literals.extend(self.underground);
        literals.into_iter()
    }

    pub fn max_variable(&self) -> i32 {
        self.get_literals().map(|v| v.abs()).max().unwrap()
    }

    pub fn get_tile(&self, assignment: &LiteralAssignment) -> Option<Tile> {
        let mut input_direction: Option<Direction> = None;
        let mut output_direction: Option<Direction> = None;
        for direction in Direction::iterator() {
            let idx: usize = direction.into();
            let has_input = assignment.get(self.input_direction[idx])?;

            if has_input {
                if input_direction != None {
                    println!("input_direction: {input_direction:?}, direction: {direction:?}");
                    return None;
                }
                // assert_eq!(input_direction, None, "Cannot have multiple input directions");
                input_direction = Some(direction);
            }

            let has_output = assignment.get(self.output_direction[idx])?;

            if has_output {
                if output_direction != None {
                    println!("output_direction: {output_direction:?}, direction: {direction:?}");
                    return None;
                }
                output_direction = Some(direction);
            }
        }

        let input_direction = input_direction;
        let output_direction = output_direction;

        let is_splitter = assignment.get(self.is_splitter)?;

        if is_splitter {
            let is_head = assignment.get(self.is_splitter_head)?;

            let direction = input_direction
                .or(output_direction)
                .expect("Shouldn't be possible to create a splitter without an input or output");

            return Some(Tile::Splitter { direction, is_head });
        }

        Some(match (input_direction, output_direction) {
            (None, None) => Tile::Empty,
            (Some(direction), None) => Tile::Underground {
                direction,
                is_input: true,
            },
            (None, Some(direction)) => Tile::Underground {
                direction,
                is_input: false,
            },
            (Some(input), Some(output)) => Tile::Belt { input, output },
        })
    }

    pub fn get_underground(
        &self,
        assignment: &LiteralAssignment,
        direction: Direction,
    ) -> Option<bool> {
        let idx: usize = direction.into();
        assignment.get(self.underground[idx])
    }
}

impl<'source> FromPyObject<'source> for TileLiterals {
    fn extract(ob: &'source pyo3::PyAny) -> pyo3::PyResult<Self> {
        Ok(TileLiterals {
            input_direction: ob.getattr("input_direction")?.extract()?,
            output_direction: ob.getattr("output_direction")?.extract()?,
            is_splitter: ob.getattr("is_splitter")?.extract()?,
            is_splitter_head: ob.getattr("is_splitter_head")?.extract()?,
            underground: ob.getattr("underground")?.extract()?,
        })
    }
}

pub struct LiteralAssignment {
    assignment: Box<[Option<bool>]>,
}

impl LiteralAssignment {
    pub fn new(max_variable: usize) -> LiteralAssignment {
        LiteralAssignment {
            assignment: vec![None; max_variable + 1].into(),
        }
    }

    pub fn get(&self, lit: i32) -> Option<bool> {
        if lit < 0 {
            self.assignment[-lit as usize].map(|v| !v)
        } else {
            self.assignment[lit as usize]
        }
    }

    pub fn set(&mut self, lit: i32, value: Option<bool>) {
        if lit < 0 {
            self.assignment[-lit as usize] = value.map(|v| !v);
        } else {
            self.assignment[lit as usize] = value;
        }
    }
}

pub trait TileListener {
    fn notify_new_tile(
        &self,
        tile_assignment: TileAssignment,
        path_location: PathLocation,
    ) -> Option<Conflict>;
}

type LiteralLookupElementType = ArrayVec<Coord2D<usize>, 2>;

pub struct TileAssignmentTracker<T: TileListener> {
    tile_assignment: Array2D<Option<Tile>>,
    tile_assignment_levels: Vec<Option<Coord2D<usize>>>,

    tile_templates: Array2D<TileLiterals>,

    literal_lookup: Box<[LiteralLookupElementType]>,

    tile_listener: T,
}

impl<T: TileListener> TileAssignmentTracker<T> {
    pub fn new(
        tile_templates: Array2D<TileLiterals>,
        tile_listener: T,
    ) -> TileAssignmentTracker<T> {
        let max_variable = tile_templates
            .flat_iter()
            .map(|tile| tile.max_variable())
            .max()
            .unwrap();

        let mut literal_lookup = vec![LiteralLookupElementType::new(); max_variable as usize + 1];
        // let mut literal_lookup = vec![None; max_variable as usize + 1];

        for (idx, tile) in tile_templates.enumerate() {
            // dbg!(idx, tile);
            for literal in tile.get_literals() {
                let lookup = &mut literal_lookup[literal.abs() as usize];

                lookup.push(idx);
            }
        }

        TileAssignmentTracker {
            tile_assignment: Array2D::new(tile_templates.width(), tile_templates.height(), None),
            tile_assignment_levels: Vec::new(),
            tile_templates,
            literal_lookup: literal_lookup.into(),
            tile_listener,
        }
    }
}

impl<T: TileListener> AssignmentListener for TileAssignmentTracker<T> {
    fn notify_assignment(
        &mut self,
        assignment: &LiteralAssignment,
        new_lit: i32,
    ) -> Option<Conflict> {
        let mut conflict: Option<Conflict> = None;

        for tile_idx in self.literal_lookup[new_lit.abs() as usize].iter() {
            let tile_literals = &self.tile_templates[tile_idx];

            if let Some(new_tile) = tile_literals.get_tile(assignment) {
                let slot = &mut self.tile_assignment[tile_idx];

                match slot {
                    Some(existing_tile) => {
                        assert_eq!(*existing_tile, new_tile);
                    }
                    None => {
                        *slot = Some(new_tile);
                        self.tile_assignment_levels.push(Some(*tile_idx));

                        if let Some(new_conflict) = self.tile_listener.notify_new_tile(
                            TileAssignment::new(
                                &assignment,
                                &self.tile_templates,
                                &self.tile_assignment,
                            ),
                            PathLocation::new(
                                Coord2D {
                                    y: tile_idx.y as isize,
                                    x: tile_idx.x as isize,
                                },
                                None,
                            ),
                        ) {
                            conflict = Some(new_conflict);
                        }
                    }
                }
            }

            for (direction, lit) in Direction::iterator().zip(tile_literals.underground) {
                if lit == new_lit {
                    if let Some(new_conflict) = self.tile_listener.notify_new_tile(
                        TileAssignment::new(
                            &assignment,
                            &self.tile_templates,
                            &self.tile_assignment,
                        ),
                        PathLocation::new(
                            Coord2D {
                                y: tile_idx.y as isize,
                                x: tile_idx.x as isize,
                            },
                            Some(direction),
                        ),
                    ) {
                        conflict = Some(new_conflict)
                    }
                }
            }
        }

        conflict
    }

    fn notify_new_decision_level(&mut self) {
        self.tile_assignment_levels.push(None);
    }

    fn notify_backtrack_single_level(&mut self) {
        loop {
            let Some(idx) = self.tile_assignment_levels.pop().expect(
                "Level must be greater than 0, therefore there should exist a None sentinel value",
            ) else {
                break;
            };

            self.tile_assignment[idx] = None;
        }
    }
}

pub trait AssignmentListener {
    fn notify_assignment(
        &mut self,
        assignment: &LiteralAssignment,
        new_lit: i32,
    ) -> Option<Conflict>;
    fn notify_new_decision_level(&mut self);
    fn notify_backtrack_single_level(&mut self);
}

pub struct AssignmentTracker<Listener: AssignmentListener> {
    level: usize,
    stack: Vec<i32>,
    assignment: LiteralAssignment,
    to_propagate: Vec<Conflict>,
    listener: Listener,
}

impl<'a, Listener: AssignmentListener> AssignmentTracker<Listener> {
    pub fn new(max_variable: usize, listener: Listener) -> AssignmentTracker<Listener> {
        AssignmentTracker {
            level: 0,
            stack: Vec::new(),
            assignment: LiteralAssignment::new(max_variable),
            to_propagate: Vec::new(),
            listener,
        }
    }
}

impl<Listener: AssignmentListener> Callbacks for AssignmentTracker<Listener> {
    #[inline(always)]
    fn notify_new_decision_level(&mut self) {
        self.level += 1;
        self.stack.push(0);
        self.listener.notify_new_decision_level();
    }

    #[inline(always)]
    fn notify_assignment(&mut self, lit: i32, is_fixed: bool) {
        if is_fixed {
            self.stack.insert(0, lit);
        } else {
            self.stack.push(lit);
        }
        self.assignment.set(lit, Some(true));

        if is_fixed && self.level != 0 {
            for _ in 0..self.level {
                self.listener.notify_backtrack_single_level();
            }

            let mut partial_assignment = LiteralAssignment::new(self.assignment.assignment.len());
            let mut partial_level = 0;

            for &lit in self.stack.iter() {
                if lit == 0 {
                    partial_level += 1;
                    self.listener.notify_new_decision_level();
                } else {
                    partial_assignment.set(lit, Some(true));
                    if partial_level > 0 {
                        if let Some(conflict) =
                            self.listener.notify_assignment(&partial_assignment, lit)
                        {
                            self.to_propagate.push(conflict);
                        }
                    }
                }
            }
        } else {
            if let Some(conflict) = self.listener.notify_assignment(&self.assignment, lit) {
                self.to_propagate.push(conflict);
            }
        }
    }

    #[inline(always)]
    fn notify_backtrack(&mut self, new_level: usize) {
        self.to_propagate.clear();

        while self.level > new_level {
            let mut lit = self.stack.pop().expect("Level must be greater then zero therefore there must be at least one element in the stack that is the level divider (0)");

            while lit != 0 {
                self.assignment.set(lit, None);
                lit = self.stack.pop().expect("Previous element was not the level divider (0) value, so there must be at least one more elem in the vec that is the level divider");
            }

            self.level -= 1;

            self.listener.notify_backtrack_single_level();
        }
    }

    #[inline(always)]
    fn propagate(&mut self) -> Option<Conflict> {
        self.to_propagate.pop()
    }
}
