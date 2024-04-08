use cadical::Conflict;

use crate::coord_2d::Coord2D;
use crate::direction::Direction;
use crate::sat::TileListener;
use crate::tile::Tile;
use crate::tile_assignment::{PathLocation, TileAssignment};

pub struct TileListenerImpl {}

impl TileListenerImpl {
    pub fn new() -> TileListenerImpl {
        TileListenerImpl {}
    }
}

impl TileListener for TileListenerImpl {
    fn notify_new_tile(
        &self,
        tile_assignment: TileAssignment,
        location: PathLocation,
    ) -> Option<Conflict> {
        if let Some(loop_thing) = find_loop(&tile_assignment, &location) {
            // dbg!(Vec::from(loop_thing));
            println!("loop start");
            for location in loop_thing.iter() {
                let content = tile_assignment.get_tile(location.coord);
                let underground = location
                    .underground_direction
                    .map(|direction| tile_assignment.get_underground(location.coord, direction));
                let literals = tile_assignment.get_literals(location.coord);
                println!("Location: {location:?}, Tile: {content:?}, Underground: {underground:?}, Literals: {literals:?}");
            }
            let conflict = tile_assignment.conflict_path(&loop_thing);
            println!("conflict {conflict:?}");
            println!("loop end");

            return Some(Conflict {
                pivot: -conflict[0],
                clause: conflict,
            });
        }

        // println!("find start: {location:?}");
        let mut path_start = location;
        loop {
            let Some(prev_location) = tile_assignment.prev(&path_start) else {
                break;
            };
            // let tile = tile_assignment.get_tile(prev_location.coord);
            // println!("prev_location: {prev_location:?}, tile: {tile:?}");
            path_start = prev_location;
        }
        // println!("path_start: {path_start:?}");

        let mut path: Vec<PathLocation> = vec![path_start];
        loop {
            let Some(next_location) = tile_assignment.next(&path.last().unwrap()) else {
                break;
            };
            path.push(next_location);
        }
        let path = path.into_boxed_slice();

        let shortcuts = find_shortcuts(&tile_assignment, &path);

        if let Some(shortcut) = shortcuts.iter().min_by_key(|shortcut| shortcut.len()) {
            let conflict = tile_assignment.conflict_path(&shortcut);
            println!("shortcut conflict {conflict:?}");
            // return Box::new([
            //     (-conflict[0], conflict)
            // ]);
            return Some(Conflict {
                pivot: -conflict[0],
                clause: conflict,
            });
        }
        // if !shortcuts.is_empty() {

        //     dbg!(shortcuts.len());
        //     let shortcut = shortcuts.iter().min_by_key(|shortcut| shortcut.len()).unwrap();
        //     dbg!(shortcut.len());
        //     for location in shortcut.iter() {
        //         let tile = tile_assignment.get_tile(location.coord);
        //         println!("loc: {location:?}, tile: {tile:?}");
        //     }

        // }

        None
    }
}

fn find_loop<'a>(
    assignment: &'a TileAssignment<'a>,
    start: &PathLocation,
) -> Option<Box<[PathLocation]>> {
    let mut visited = vec![start.clone()];

    loop {
        let next_location = assignment.next(visited.last().unwrap())?;
        if next_location == *start {
            return Some(visited.into_boxed_slice());
        }

        visited.push(next_location);
    }
}

fn find_shortcuts<'a>(
    assignment: &'a TileAssignment<'a>,
    path: &[PathLocation],
) -> Box<[Box<[PathLocation]>]> {
    let mut shortcuts: Vec<Box<[PathLocation]>> = Vec::new();

    let get_distance = |location: PathLocation| {
        for (i, path_location) in path.iter().enumerate() {
            if location == *path_location {
                return Some(i);
            }
        }

        None
    };

    for (distance, location) in path.iter().enumerate() {
        if location.underground_direction != None {
            continue;
        }

        let mut base_require_empty: Vec<Coord2D<isize>> = Vec::new();
        let min_cut_distance: usize;
        let alt_connection_directions: Vec<Direction>;

        match assignment.get_tile(location.coord) {
            Some(Tile::Belt { input, output }) => {
                alt_connection_directions = [*input, input.next(), input.prev()]
                    .into_iter()
                    .filter(|direction| *direction != *output)
                    .collect();
                min_cut_distance = distance;
            }
            Some(Tile::Underground {
                direction,
                is_input,
            }) => {
                if *is_input {
                    alt_connection_directions =
                        vec![*direction, direction.next(), direction.prev()];
                    min_cut_distance = distance;
                } else {
                    let previous_loc = location.coord - direction.delta();
                    if !assignment.valid_index(previous_loc) {
                        continue;
                    }

                    alt_connection_directions = vec![direction.next(), direction.prev()];

                    let previous_distance = get_distance(PathLocation {
                        coord: previous_loc,
                        underground_direction: None,
                    });
                    if previous_distance
                        .map_or(false, |previous_distance| previous_distance > distance)
                    {
                        min_cut_distance = previous_distance.unwrap();
                    } else if assignment.get_tile(previous_loc) == Some(&Tile::Empty) {
                        min_cut_distance = distance;
                        base_require_empty.push(previous_loc);
                    } else {
                        continue;
                    }
                }
            }
            _ => continue,
        }

        for transverse in alt_connection_directions {
            let mut trans_location = location.coord + transverse.delta();

            if !assignment.valid_index(trans_location) {
                continue;
            }

            let mut require_empty = base_require_empty.clone();

            let mut next_tile = assignment.get_tile(trans_location);
            let mut shortcut_len = 1;

            while next_tile == Some(&Tile::Empty) {
                require_empty.push(trans_location);
                shortcut_len += 1;
                trans_location = trans_location + transverse.delta();

                if assignment.valid_index(trans_location) {
                    next_tile = assignment.get_tile(trans_location);
                } else {
                    next_tile = None;
                }
            }

            let Some(next_tile) = next_tile else {
                continue;
            };

            let shortcut_len = shortcut_len;
            let trans_location = trans_location;
            let Some(trans_distance) = get_distance(PathLocation {
                coord: trans_location,
                underground_direction: None,
            }) else {
                continue;
            };

            if let Tile::Underground {
                direction,
                is_input,
            } = next_tile
            {
                if *is_input {
                    let underground_next = trans_location + direction.delta();
                    if !assignment.valid_index(underground_next) {
                        continue;
                    }

                    if assignment.get_tile(underground_next) == Some(&Tile::Empty) {
                        require_empty.push(underground_next);
                    } else {
                        let Some(underground_next_dist) = get_distance(PathLocation {
                            coord: underground_next,
                            underground_direction: None,
                        }) else {
                            continue;
                        };

                        if !(min_cut_distance < underground_next_dist
                            && underground_next_dist < trans_distance)
                        {
                            continue;
                        }
                    }
                }
            }

            if trans_distance > min_cut_distance + shortcut_len {
                let mut conflict: Vec<PathLocation> = path[distance..trans_distance + 1].into();

                conflict.extend(require_empty.iter().map(|coord| PathLocation {
                    coord: *coord,
                    underground_direction: None,
                }));

                shortcuts.push(conflict.into_boxed_slice());
            }
        }

        // TODO Underground??
    }

    shortcuts.into_boxed_slice()
}
