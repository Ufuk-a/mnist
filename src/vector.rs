use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Range, Sub};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Vector {
    pub elements: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum VectorError {
    DimensionMismatch(&'static str),
    DivisionByZero(&'static str)
}

impl Vector {
    pub fn new(capacity: usize, value: f64) -> Self {
        Vector{elements: (0..capacity).map(|_| value).collect()}
    }

    pub fn with_capacity(capacity: usize) -> Self { 
        Vector{elements: Vec::with_capacity(capacity)}
    }

    pub fn from_vec(elements: Vec<f64>) -> Self {
        Vector{elements}
    }

    pub fn random(dim: usize, range: Range<f64>) -> Self {
        let mut rng = rand::thread_rng();
        let result: Vec<f64> = (0..dim).map(|_| rng.gen_range(range.clone())).collect();
        Vector{elements: result}
    }

    pub fn iter(&self) -> std::slice::Iter<f64> {
        self.elements.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<f64> {
        self.elements.iter_mut()
    }

    pub fn push(&mut self, element: f64) {
        self.elements.push(element);
    }

    pub fn pop(&mut self) -> Option<f64> {
        self.elements.pop()
    }

    pub fn dim(&self) -> usize {
        self.elements.len()
    }

    pub fn len(&self) -> f64 {
        self.elements.iter().map(|a| a * a).sum::<f64>().sqrt()
    }
 
    pub fn normalize(&self) -> Result<Vector, VectorError> {
        if self.len() != 0.0 {
            Ok((self / self.len())?)
        } else {
            Err(VectorError::DivisionByZero("The vector must have a non-zero length"))
        }
    }

    pub fn dot(lhs: &Vector, rhs: &Vector) -> Result<f64, VectorError> {
        if lhs.dim() == rhs.dim() {
            let result = lhs.elements.iter().zip(&rhs.elements).map(|(a, b)| a * b).sum();
            Ok(result)
        } else {
            Err(VectorError::DimensionMismatch("The vectors must have the same dimensionality."))
        }
    }

    pub fn cross(&self, other: &Vector) -> Result<Vector, VectorError> {
        match (self.dim(), other.dim()) {
            (2, 2) => {
                let a = &self.elements;
                let b = &other.elements;
                let z = a[0] * b[1] - a[1] * b[0];
                Ok(Vector { elements: vec![0.0, 0.0, z] }) 
            },
            (3, 3) => {
                let a = &self.elements;
                let b = &other.elements;
                let cross_prod = Vector {
                    elements: vec![
                        a[1] * b[2] - a[2] * b[1], 
                        a[2] * b[0] - a[0] * b[2], 
                        a[0] * b[1] - a[1] * b[0],
                    ],
                };
                Ok(cross_prod)
            },
            _ => Err(VectorError::DimensionMismatch("Vectors must both be two-dimensional or both be three-dimensional."))
        }
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}

impl Neg for &Vector {
    type Output = Vector;

    fn neg(self) -> Self::Output {
        let result: Vec<f64> = self.elements.iter().map(|a| -a).collect();
        Vector{elements: result}
    }
}

impl Add<&Vector> for &Vector {
    type Output = Result<Vector, VectorError>;

    fn add(self, rhs: &Vector) -> Self::Output {
        if self.dim() != rhs.dim() {
            return Err(VectorError::DimensionMismatch("The vectors must have the same dimensionality."));
        }

        let result: Vec<f64> = self.elements.iter().zip(&rhs.elements).map(|(a,b)| a + b).collect();
        Ok(Vector{elements: result})
    }
}

impl Sub<&Vector> for &Vector {
    type Output = Result<Vector, VectorError>;

    fn sub(self, rhs: &Vector) -> Self::Output {
        if self.dim() != rhs.dim() {
            return Err(VectorError::DimensionMismatch("The vectors must have the same dimensionality."));
        }

        let result: Vec<f64> = self.elements.iter().zip(&rhs.elements).map(|(a,b)| a - b).collect();
        Ok(Vector{elements: result})
    }
}

impl Mul for &Vector {
    type Output = Result<Vector, VectorError>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.dim() != rhs.dim() {
            return Err(VectorError::DimensionMismatch("The vectors must have the same dimensionality."));
        }
        let elements = self.elements
            .iter()
            .zip(&rhs.elements)
            .map(|(&a, &b)| a * b)
            .collect();
        Ok(Vector{elements})
    }
}

impl Mul<f64> for &Vector {
    type Output = Vector;

    fn mul(self, rhs: f64) -> Self::Output {
        let elements = self.elements.iter().map(|&x| x * rhs).collect();
        Vector{elements}
    }
}

impl Mul<&Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: &Vector) -> Self::Output {
        rhs * self
    }
}

impl Div<f64> for &Vector {
    type Output = Result<Vector, VectorError>;

    fn div(self, rhs: f64) -> Self::Output {
        if rhs == 0.0 {
            Err(VectorError::DivisionByZero("Cannot divide by zero"))
        } else {
            let elements = self.elements.iter().map(|&x| x / rhs).collect();
            Ok(Vector { elements })
        }
    }
}

impl Div<&Vector> for f64 {
    type Output = Result<Vector, VectorError>;

    fn div(self, rhs: &Vector) -> Self::Output {
        rhs / self
    }
}

impl IntoIterator for Vector {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}