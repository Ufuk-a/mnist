use crate::vector::Vector;

pub type ActivationFn = fn(f64) -> f64;
pub type VectorActivationFn = fn(&Vector) -> Vector;

pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn linear(x: f64) -> f64 {
    x
}

pub fn softmax(x: &Vector) -> Vector {
    let max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f64> = x.iter().map(|&xi| (xi - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    Vector::from_vec(exps.into_iter().map(|xi| xi / sum).collect())
}

pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    let sigmoid_x = sigmoid(x);
    sigmoid_x * (1.0 - sigmoid_x)
}

pub fn tanh_derivative(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}

pub fn linear_derivative(_x: f64) -> f64 {
    1.0
}

pub fn softmax_derivative(output: &Vector) -> Vector {
    let n = output.dim();
    let mut jacobian = Vector::with_capacity(n);
    
    for i in 0..n {
        jacobian.push(output[i] * (1.0 - output[i]));
    }
    
    jacobian
}