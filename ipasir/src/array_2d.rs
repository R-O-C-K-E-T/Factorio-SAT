use std::ops::{Index, IndexMut};

use crate::coord_2d::Coord2D;

pub struct Array2D<T> {
    storage: Box<[T]>,
    width: usize,
    height: usize,
}

impl<T> Array2D<T> {
    pub fn new(width: usize, height: usize, elem: T) -> Array2D<T>
    where
        T: Clone,
    {
        Array2D {
            storage: vec![elem; width * height].into(),
            width,
            height,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn flat_iter(&self) -> impl Iterator<Item = &T> {
        self.storage.iter()
    }

    pub fn enumerate(&self) -> impl Iterator<Item = (Coord2D<usize>, &T)> {
        (0..self.width)
            .flat_map(|col| (0..self.height).map(move |row| Coord2D { x: col, y: row }))
            .zip(self.flat_iter())
    }

    pub fn valid_index(&self, coord: Coord2D<isize>) -> bool {
        return coord.x >= 0
            && coord.y >= 0
            && coord.x < self.width as isize
            && coord.y < self.height as isize;
    }

    pub fn at(&self, coord: Coord2D<isize>) -> Option<&T> {
        if !self.valid_index(coord) {
            return None;
        }

        Some(&self[coord])
    }
}

impl<T> Index<[usize; 2]> for Array2D<T> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.storage[index[0] * self.width + index[1]]
    }
}

impl<T> IndexMut<[usize; 2]> for Array2D<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.storage[index[0] * self.width + index[1]]
    }
}

impl<T> Index<&Coord2D<isize>> for Array2D<T> {
    type Output = T;

    fn index(&self, index: &Coord2D<isize>) -> &Self::Output {
        &self.storage[index.y as usize * self.width + index.x as usize]
    }
}

impl<T> IndexMut<&Coord2D<isize>> for Array2D<T> {
    fn index_mut(&mut self, index: &Coord2D<isize>) -> &mut Self::Output {
        &mut self.storage[index.y as usize * self.width + index.x as usize]
    }
}

impl<T> Index<Coord2D<isize>> for Array2D<T> {
    type Output = T;

    fn index(&self, index: Coord2D<isize>) -> &Self::Output {
        &self.storage[index.y as usize * self.width + index.x as usize]
    }
}

impl<T> IndexMut<Coord2D<isize>> for Array2D<T> {
    fn index_mut(&mut self, index: Coord2D<isize>) -> &mut Self::Output {
        &mut self.storage[index.y as usize * self.width + index.x as usize]
    }
}

impl<T> Index<&Coord2D<usize>> for Array2D<T> {
    type Output = T;

    fn index(&self, index: &Coord2D<usize>) -> &Self::Output {
        &self.storage[index.y * self.width + index.x]
    }
}

impl<T> IndexMut<&Coord2D<usize>> for Array2D<T> {
    fn index_mut(&mut self, index: &Coord2D<usize>) -> &mut Self::Output {
        &mut self.storage[index.y * self.width + index.x]
    }
}

impl<T> Index<Coord2D<usize>> for Array2D<T> {
    type Output = T;

    fn index(&self, index: Coord2D<usize>) -> &Self::Output {
        &self.storage[index.y * self.width + index.x]
    }
}

impl<T> IndexMut<Coord2D<usize>> for Array2D<T> {
    fn index_mut(&mut self, index: Coord2D<usize>) -> &mut Self::Output {
        &mut self.storage[index.y * self.width + index.x]
    }
}

impl<T> TryFrom<&[Box<[T]>]> for Array2D<T>
where
    T: Clone,
{
    type Error = ();

    fn try_from(value: &[Box<[T]>]) -> Result<Self, Self::Error> {
        if value.is_empty() {
            return Ok(Array2D {
                width: 0,
                height: 0,
                storage: Box::new([]),
            });
        }

        let height = value.len();
        let width = value[0].len();
        let mut flat: Vec<T> = Vec::with_capacity(width * height);

        for row in value.into_iter() {
            flat.extend_from_slice(row);
            if row.len() != width {
                return Err(());
            }
        }

        Ok(Array2D {
            width,
            height,
            storage: flat.into(),
        })
    }
}
