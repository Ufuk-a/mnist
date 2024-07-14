use std::ops::{Index, IndexMut, Range, Add, Sub, Mul, Div};
use std::vec;
use super::vector::{Vector, VectorError};


#[derive(Debug, Clone)]
pub struct Matrix {
    pub elements: Vec<Vector>,
}

#[derive(Debug, Clone)]
pub enum MatrixError {
    DimensionMismatch(&'static str),
    DivisionByZero(&'static str),
    VectorError(VectorError),
}

impl From<VectorError> for MatrixError {
    fn from(error: VectorError) -> Self {
        MatrixError::VectorError(error)
    }
}

impl Matrix {
    pub fn new(capacity_x: usize, capacity_y: usize, value: f64) -> Self {
        let elements = vec![Vector::new(capacity_x, value); capacity_y]; 
        Matrix{elements}
    }

    pub fn random(dim_x: usize, dim_y: usize, range: Range<f64>) -> Self {
        let elements = (0..dim_y).map(|_| Vector::random(dim_x, range.clone())).collect();
        Matrix{elements}
    }

    pub fn dim_x(&self) -> usize {
        if let Some(first_row) = self.elements.first() {
            first_row.dim()
        } else {
            0
        }
    }

    pub fn dim_y(&self) -> usize {
        self.elements.len()
    }

    pub fn iter(&self) -> std::slice::Iter<Vector> {
        self.elements.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Vector> {
        self.elements.iter_mut()
    }

    pub fn push(&mut self, element: Vector) {
        self.elements.push(element);
    }

    pub fn pop(&mut self) -> Option<Vector> {
        self.elements.pop()
    }

    pub fn transpose(&self) -> Matrix {
        let mut elements = vec![Vector::with_capacity(self.dim_y()); self.dim_x()];

        for i in 0..self.dim_x() {
            for j in 0..self.dim_y() {
                elements[i].push(self[j][i]);
            }
        }

        Matrix{elements}
    }
}

impl Index<usize> for Matrix {
    type Output = Vector;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}


impl Add<&Matrix> for &Matrix {
    type Output = Result<Matrix, MatrixError>;

    fn add(self, rhs: &Matrix) -> Self::Output {
        if self.dim_x() != rhs.dim_x() || self.dim_y() != rhs.dim_y() {
            return Err(MatrixError::DimensionMismatch("The matrices must have the same dimensions."));
        }

        let elements: Vec<Vector> = self.elements.iter().zip(&rhs.elements).map(|(a, b)| a + b).collect::<Result<Vec<Vector>, VectorError>>()?;

        Ok(Matrix{elements})
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Result<Matrix, MatrixError>;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        if self.dim_x() != rhs.dim_x() || self.dim_y() != rhs.dim_y() {
            return Err(MatrixError::DimensionMismatch("The matrices must have the same dimensions."));
        }

        let elements: Vec<Vector> = self.elements.iter().zip(&rhs.elements).map(|(a, b)| a - b).collect::<Result<Vec<Vector>, VectorError>>()?;

        Ok(Matrix{elements})
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Result<Matrix, MatrixError>;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        if self.dim_x() != rhs.dim_y() {
            return Err(MatrixError::DimensionMismatch("The number of columns in the first matrix must be equal to the number of rows in the second matrix."));
        }

        let mut elements = vec![Vector::with_capacity(rhs.dim_x()); self.dim_y()];

        for i in 0..self.dim_y() {
            for j in 0..rhs.dim_x() {
                let sum = (0..self.dim_x()).map(|k| self[i][k] * rhs[k][j]).sum();
                elements[i].push(sum);
            }
        }

        Ok(Matrix{elements})
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Self::Output {
        let elements = self.elements.iter().map(|v| v * rhs).collect();
        Matrix{elements}
    }
}

impl Mul<&Vector> for &Matrix {
    type Output = Result<Vector, MatrixError>;

    fn mul(self, rhs: &Vector) -> Self::Output {
        if self.dim_x() != rhs.dim() {
            return Err(MatrixError::DimensionMismatch("The number of columns in the matrix must be equal to the number of elements in the vector."));
        }

        let result: Vec<f64> = self.elements.iter().map(|row| Vector::dot(row, rhs).unwrap()).collect();

        Ok(Vector::from_vec(result))
    }
}

impl Mul<&Vector> for Matrix {
    type Output = Result<Vector, MatrixError>;

    fn mul(self, rhs: &Vector) -> Self::Output {
        &self * rhs
    }
}

impl Div<f64> for &Matrix {
    type Output = Result<Matrix, MatrixError>;

    fn div(self, rhs: f64) -> Self::Output {
        if rhs == 0.0 {
            return Err(MatrixError::DivisionByZero("Cannot divide by zero"));
        }

        let elements = self.elements.iter().map(|v| v / rhs).collect::<Result<Vec<Vector>, VectorError>>()?;
        Ok(Matrix{elements})
    }
}

impl Add for Matrix {
    type Output = Result<Matrix, MatrixError>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Sub for Matrix {
    type Output = Result<Matrix, MatrixError>;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Mul for Matrix {
    type Output = Result<Matrix, MatrixError>;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Self::Output {
        &self * rhs
    }
}

impl Mul<Vector> for Matrix {
    type Output = Result<Vector, MatrixError>;

    fn mul(self, rhs: Vector) -> Self::Output {
        &self * &rhs
    }
}

impl Div<f64> for Matrix {
    type Output = Result<Matrix, MatrixError>;

    fn div(self, rhs: f64) -> Self::Output {
        &self / rhs
    }
}

impl Mul<&Matrix> for &Vector {
    type Output = Result<Vector, MatrixError>;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        if self.dim() != rhs.dim_y() {
            return Err(MatrixError::DimensionMismatch("The number of elements in the vector must be equal to the number of rows in the matrix."));
        }

        let result: Vec<f64> = (0..rhs.dim_x()).map(|i| self.iter().zip(rhs.iter()).map(|(v, row)| v * row[i]).sum()).collect();

        Ok(Vector::from_vec(result))
    }
}

impl IntoIterator for Matrix {
    type Item = Vector;
    type IntoIter = vec::IntoIter<Vector>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl AsRef<Matrix> for Matrix {
    fn as_ref(&self) -> &Matrix {
        self
    }
}



impl FromIterator<Vector> for Matrix {
    fn from_iter<T: IntoIterator<Item = Vector>>(iter: T) -> Self {
        let elements: Vec<Vector> = iter.into_iter().collect();
        Matrix{elements}
    }
}