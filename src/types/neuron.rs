use std::ops::Range;

use rand::Rng;

use super::{Vector, Matrix};

#[derive(Debug, PartialEq, Clone)]
pub struct Neuron {
    pub weights: Vector,
    pub bias: f64,
}

impl Neuron {
    pub fn new(num_weights: usize, bias: f64) -> Self {
        Neuron{weights: Vector::new(num_weights), bias}
    }

    pub fn from(weights: Vector, bias: f64) -> Self {
        Neuron{weights, bias}
    }

    pub fn random(num_weights: usize, range: Range<f64>) -> Self {
        let mut rng = rand::thread_rng();
        Neuron{weights: Vector::random(num_weights, range), bias: rng.gen_range(-0.1..0.1)}
    }

    pub fn to_matrix(neurons: Vec<Neuron>) -> Matrix {
        
    }
}
