use crate::vector::Vector;

use std::ops::Range;
use rand::Rng;


#[derive(Debug, Clone)]
pub struct Neuron {
    pub weights: Vector,
    pub bias: f64,
}

impl Neuron {
    pub fn new(num_weights: usize, weight: f64 , bias: f64) -> Self {
        Neuron{weights: Vector::new(num_weights, weight), bias}
    }

    pub fn from(weights: Vector, bias: f64) -> Self {
        Neuron{weights, bias}
    }

    pub fn random(num_weights: usize, w_range: Range<f64>, b_range: Range<f64>) -> Self {
        let mut rng = rand::thread_rng();
        Neuron{weights: Vector::random(num_weights, w_range), bias: rng.gen_range(b_range)}
    }
}