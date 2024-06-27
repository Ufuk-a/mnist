use std::ops::{Range, Index, IndexMut};
use std::vec;

use crate::vector::Vector;
use crate::matrix::Matrix;
use crate::neuron::Neuron;

#[derive(Debug, Clone)]
pub struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(num: usize, prev_layer_num: usize, weight: f64, bias: f64) -> Self {
        Layer{neurons: (0..num).map(|_| Neuron::new(prev_layer_num, weight, bias)).collect()}
    }

    pub fn random(num: usize, prev_layer_num: usize, w_range: Range<f64>, b_range: Range<f64>) -> Self {
        Layer{neurons: (0..num).map(|_| Neuron::random(prev_layer_num, w_range.clone(), b_range.clone())).collect()}
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
        Matrix {elements} 
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