mod array_2d;
mod coord_2d;
mod direction;
mod sat;
mod tile;
mod tile_assignment;
mod tile_listener;

use crate::direction::Direction;
use crate::sat::TileLiterals;
use crate::tile::Tile;

use array_2d::Array2D;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use cadical::Solver;
use sat::{AssignmentTracker, TileAssignmentTracker};
use tile_listener::TileListenerImpl;

#[pyclass]
struct FactorioSolver {
    solver: Solver<AssignmentTracker<TileAssignmentTracker<TileListenerImpl>>>,
}

#[pymethods]
impl FactorioSolver {
    #[new]
    fn new(tile_templates: Vec<Vec<TileLiterals>>) -> PyResult<Self> {
        let tile_templates = Box::from_iter(
            tile_templates
                .iter()
                .map(|row| row.clone().into_boxed_slice()),
        );

        let Ok(tile_templates) = Array2D::try_from(tile_templates.as_ref()) else {
            return Err(PyRuntimeError::new_err(
                "Invalid 2D Array for tile_templates",
            ));
        };

        let Some(max_variable) = tile_templates
            .flat_iter()
            .map(|tile| tile.max_variable())
            .max()
        else {
            return Err(PyRuntimeError::new_err("Cannot solve empty array"));
        };

        let mut solver = Solver::new();

        // TODO Do this smarter and rustier
        let literals: Vec<i32> = tile_templates
            .flat_iter()
            .flat_map(|tile| tile.get_literals())
            .collect();
        solver.set_callbacks(Some(AssignmentTracker::new(
            max_variable as usize,
            TileAssignmentTracker::new(tile_templates, TileListenerImpl::new()),
        )));

        for lit in literals {
            solver.add_observed(lit);
        }

        Ok(FactorioSolver { solver })
    }

    pub fn add_clause(&mut self, clause: Vec<i32>) {
        self.solver.add_clause(clause);
    }

    pub fn solve(&mut self) -> Option<bool> {
        self.solver.solve()
    }

    pub fn get_model(&self) -> Vec<i32> {
        let mut result = Vec::new();

        for var in 1..=self.solver.max_variable() {
            result.push(match self.solver.value(var) {
                Some(true) => var,
                Some(false) => -var,
                None => continue,
            });
        }

        result
    }
}

#[pymodule]
fn ipasir(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FactorioSolver>()?;
    Ok(())
}
