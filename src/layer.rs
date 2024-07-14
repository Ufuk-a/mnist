use std::ops::{Range, Index, IndexMut};
use std::vec;
use std::slice::Iter;
use crate::vector::Vector;
use crate::matrix::{Matrix, MatrixError};
use crate::neuron::Neuron;
use crate::activation::{ActivationFn, VectorActivationFn, linear, tanh, sigmoid, softmax, relu};


#[derive(Debug, Clone)]
pub enum LayerActivation {
    ElementWise(ActivationFn),
    Vector(VectorActivationFn),
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation: LayerActivation,
}

impl Layer {
    pub fn new(num: usize, prev_layer_num: usize, weight: f64, bias: f64, activation: LayerActivation) -> Self {
        Layer{neurons: (0..num).map(|_| Neuron::new(prev_layer_num, weight, bias)).collect(), activation}
    }

    pub fn random(num: usize, prev_layer_num: usize, w_range: Range<f64>, b_range: Range<f64>, activation: LayerActivation) -> Self {
        Layer{neurons: (0..num).map(|_| Neuron::random(prev_layer_num, w_range.clone(), b_range.clone())).collect(), activation}
    }

    pub fn dim(&self) -> usize {
        self.neurons.len()
    }

    pub fn iter(&self) -> std::slice::Iter<Neuron> {
        self.neurons.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Neuron> {
        self.neurons.iter_mut()
    }

    pub fn push(&mut self, neuron: Neuron) {
        self.neurons.push(neuron);
    }

    pub fn pop(&mut self) -> Option<Neuron> {
        self.neurons.pop()
    }

    pub fn to_matrix(&self) -> Matrix {
        let elements: Vec<Vector> = self.iter().map(|neuron|neuron.weights.clone()).collect();
        Matrix{elements} 
    }

    pub fn activate(&self, input: &Vector) -> Result<Vector, MatrixError> {
        let output = (self.to_matrix() * input)?;
        match &self.activation {
            LayerActivation::ElementWise(f) => Ok(Vector::from_vec(output.iter().map(|&x| f(x)).collect())),
            LayerActivation::Vector(f) => Ok(f(&output)),
        }
    }

    pub fn get_fn(&self) -> String {
        match self.activation {
            LayerActivation::ElementWise(f) => {
                if f as usize == relu as usize {
                    "ReLU".to_string()
                } else if f as usize == linear as usize {
                    "Linear".to_string()
                } else if f as usize == sigmoid as usize {
                    "Sigmoid".to_string()
                } else if f as usize == tanh as usize {
                    "Tanh".to_string()
                } else if f as usize == linear as usize {
                    "Linear".to_string()
                } else {
                    "Custom ElementWise".to_string()
                }
            },
            LayerActivation::Vector(f) => {
                if f as usize == softmax as usize {
                    "Softmax".to_string()
                } else {
                    "Custom Vector".to_string()
                }
            }
        }
    }
}

impl Index<usize> for Layer {
    type Output = Neuron;

    fn index(&self, index: usize) -> &Self::Output {
        &self.neurons[index]
    }
}

impl IndexMut<usize> for Layer {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.neurons[index]
    }
}

impl IntoIterator for Layer {
    type Item = Neuron;
    type IntoIter = vec::IntoIter<Neuron>;

    fn into_iter(self) -> Self::IntoIter {
        self.neurons.into_iter()
    }
}

impl<'a> IntoIterator for &'a Layer {
    type Item = &'a Neuron;
    type IntoIter = Iter<'a, Neuron>;

    fn into_iter(self) -> Self::IntoIter {
        self.neurons.iter()
    }
}