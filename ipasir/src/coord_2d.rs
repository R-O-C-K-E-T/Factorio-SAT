use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coord2D<T> {
    pub x: T,
    pub y: T,
}

impl<T> Add for Coord2D<T>
where
    T: Add,
{
    type Output = Coord2D<<T as Add>::Output>;

    fn add(self, rhs: Self) -> Self::Output {
        Coord2D {
            y: self.y + rhs.y,
            x: self.x + rhs.x,
        }
    }
}

impl<T> Sub for Coord2D<T>
where
    T: Sub,
{
    type Output = Coord2D<<T as Sub>::Output>;

    fn sub(self, rhs: Self) -> Self::Output {
        Coord2D {
            y: self.y - rhs.y,
            x: self.x - rhs.x,
        }
    }
}

impl<T> AddAssign for Coord2D<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.y += rhs.y;
        self.x += rhs.x;
    }
}

impl<T> SubAssign for Coord2D<T>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.y -= rhs.y;
        self.x -= rhs.x;
    }
}

impl<T> Mul<T> for Coord2D<T>
where
    T: Mul,
    T: Copy,
{
    type Output = Coord2D<<T as Mul>::Output>;

    fn mul(self, rhs: T) -> Self::Output {
        Coord2D {
            y: self.y * rhs,
            x: self.x * rhs,
        }
    }
}

impl<T> MulAssign<T> for Coord2D<T>
where
    T: MulAssign,
    T: Copy,
{
    fn mul_assign(&mut self, rhs: T) {
        self.y *= rhs;
        self.x *= rhs;
    }
}
