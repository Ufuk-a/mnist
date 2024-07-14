use std::fs::File;
use std::ops::{Range, Index, IndexMut};
use std::iter;
use std::io::{Write, Error, BufReader, BufRead};

use crate::neuron::Neuron;
use crate::vector::{Vector, VectorError};
use crate::matrix::MatrixError;
use crate::layer::Layer;
use crate::activation::linear;
use crate::layer::LayerActivation;
use crate::activation::*;
use crate::activation;

#[derive(Debug)]
pub enum NetworkError {
    MatrixError(MatrixError),
    VectorError(VectorError),
    InvalidStructure(&'static str),
    ActivationMismatch(&'static str),
    DimensionMismatch(&'static str),
}

impl From<MatrixError> for NetworkError {
    fn from(error: MatrixError) -> Self {
        NetworkError::MatrixError(error)
    }
}

impl From<VectorError> for NetworkError {
    fn from(error: VectorError) -> Self {
        NetworkError::VectorError(error)
    }
}



pub struct Network {
    pub layers: Vec<Layer>
}

impl Network {
    pub fn new(structure: &[usize], weight: f64, bias: f64, activation_config: Vec<(LayerActivation, usize)>) -> Result<Self, NetworkError> {
        if structure.len() < 2 {
            return Err(NetworkError::InvalidStructure("Network must have at least two layers"));
        }

        let activations = Self::expand_activation_config(&activation_config, structure.len() - 1)?;
        if structure.len() - 1 != activations.len() {
            return Err(NetworkError::ActivationMismatch("Number of activation functions must match number of layers - 1"));
        }
        
        let mut layers = Vec::new();
        layers.push(Layer::new(structure[0], 1, 1.0, 0.0, LayerActivation::ElementWise(linear))); 
        layers.extend(structure.windows(2).zip(activations.iter()).map(|(window, activation)| Layer::new(window[1], window[0], weight, bias, activation.clone())));
        Ok(Network{layers})
    }

    pub fn random(structure: &[usize], w_range: Range<f64>, b_range: Range<f64>, activation_config: Vec<(LayerActivation, usize)>) -> Result<Self, NetworkError> {
        if structure.len() < 2 {
            return Err(NetworkError::InvalidStructure("Network must have at least two layers"));
        }

        let activations = Self::expand_activation_config(&activation_config, structure.len() - 1)?;
        if structure.len() - 1 != activations.len() {
            return Err(NetworkError::ActivationMismatch("Number of activation functions must match number of layers - 1"));
        }
        
        let mut layers = Vec::new();
        layers.push(Layer::new(structure[0], 1, 1.0, 0.0, LayerActivation::ElementWise(linear))); 
        layers.extend(structure.windows(2).zip(activations.iter()).map(|(window, activation)| Layer::random(window[1], window[0], w_range.clone(), b_range.clone(), activation.clone())));
        Ok(Network{layers})
    }

    fn expand_activation_config(config: &[(LayerActivation, usize)], total_layers: usize) -> Result<Vec<LayerActivation>, NetworkError> {
        let mut expanded = Vec::with_capacity(total_layers);
        for (activation, repeat) in config {
            expanded.extend(iter::repeat(activation.clone()).take(*repeat));
        }
        if expanded.len() != total_layers {
            return Err(NetworkError::ActivationMismatch("Number of activation functions must exactly match the number of layers (excluding input layer)"));
        }
        Ok(expanded)
    }

    pub fn iter(&self) -> std::slice::Iter<Layer> {
        self.layers.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Layer> {
        self.layers.iter_mut()
    }

    pub fn layer_num(&self) -> usize {
        self.layers.len()
    }

    pub fn push(&mut self, element: Layer) {
        self.layers.push(element);
    }

    pub fn pop(&mut self) -> Option<Layer> {
        self.layers.pop()
    }

    pub fn feed_forward(&self, input: Vector) -> Result<Vector, NetworkError> {
        let mut current_output = input;

        for layer in self.layers.iter().skip(1) {
            match layer.activate(&current_output) {
                Ok(output) => {
                    current_output = output;
                },
                Err(error) => {
                    return Err(NetworkError::from(error));
                }
            }
        }
    
        Ok(current_output)
    }

    pub fn serialize(&self, file_name: &str) -> Result<(), Error> {
        let mut buffer = Vec::new();
        writeln!(buffer, "mlp")?;
        
        for layer in self.iter() {
            write!(buffer, "{} ", layer.dim())?;
        }
        writeln!(buffer)?;
        
        for layer in self.iter() {
            for neuron in layer.iter() {
                writeln!(buffer, "{}", neuron.bias)?;
                writeln!(buffer, "{:?}", neuron.weights.elements)?;
            }
            writeln!(buffer, "{}", layer.get_fn())?;
        }
        
        let mut file = File::create(file_name)?;
        file.write_all(&buffer)?;
        Ok(())
    }

    pub fn deserialize(file_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(file_name)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let header = lines.next().ok_or("File is empty")??;
        if header != "mlp" {
            return Err("Invalid file format: missing 'mlp' header".into());
        }

        let dimensions: Vec<usize> = lines
            .next()
            .ok_or("Missing layer dimensions")?
            ?
            .split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?;

        let mut layers = Vec::new();
        let mut layer_iter = dimensions.windows(2);

        while let Some(&[_inputs, outputs]) = layer_iter.next() {
            let mut neurons = Vec::new();

            for _ in 0..outputs {
                // Read bias
                let bias: f64 = lines.next().ok_or("Unexpected end of file")??.parse()?;

                // Read weights
                let weights_str = lines.next().ok_or("Unexpected end of file")??;
                let weights: Vec<f64> = weights_str
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()?;

                let neuron = Neuron::from(Vector::from_vec(weights), bias);
                neurons.push(neuron);
            }

            // Read activation function
            let activation_str = lines.next().ok_or("Missing activation function")??;
            let activation = match activation_str.as_str() {
                "ReLU" => LayerActivation::ElementWise(activation::relu),
                "Linear" => LayerActivation::ElementWise(activation::linear),
                "Sigmoid" => LayerActivation::ElementWise(activation::sigmoid),
                "Tanh" => LayerActivation::ElementWise(activation::tanh),
                "Softmax" => LayerActivation::Vector(activation::softmax),
                _ => return Err(format!("Unknown activation function: {}", activation_str).into()),
            };

            layers.push(Layer { neurons, activation });
        }

        Ok(Network { layers })
    }

    fn mse_loss(output: &Vector, target: &Vector) -> Result<f64, NetworkError> {
        if output.dim() != target.dim() {
            return Err(NetworkError::DimensionMismatch("Output and target dimensions do not match"));
        }
        
        let squared_errors: Result<Vector, NetworkError> = (output - target)?.iter().map(|&e| Ok(e * e)).collect();
    
        Ok(squared_errors?.iter().sum::<f64>() / output.dim() as f64)
    }

    fn mse_loss_derivative(output: &Vector, target: &Vector) -> Result<Vector, NetworkError> {
        if output.dim() != target.dim() {
            return Err(NetworkError::DimensionMismatch("Output and target dimensions do not match"));
        }
        
        Ok(Vector::from_iter(
            output.iter().zip(target.iter()).map(|(&o, &t)| 2.0 * (o - t) / output.dim() as f64)
        ))
    }

    pub fn backpropagate(&mut self, input: &Vector, target: &Vector, learning_rate: f64) -> Result<(), NetworkError> {
        let mut layer_outputs = Vec::with_capacity(self.layers.len());
        let mut activations = Vec::with_capacity(self.layers.len());
        
        let mut current_input = input.clone();
        for layer in self.layers.iter().skip(1) {
            let output = layer.activate(&current_input)?;
            layer_outputs.push(current_input);
            activations.push(output.clone());
            current_input = output;
        }
        
        let mut delta = Self::mse_loss_derivative(&activations.last().unwrap(), target)?;
        
        for (layer_index, layer) in self.layers.iter_mut().skip(1).enumerate().rev() {
            let layer_output = &layer_outputs[layer_index];
            let activation_derivative = match &layer.activation {
                LayerActivation::ElementWise(f) => {
                    if *f as usize == relu as usize {
                        Vector::from_iter(layer_output.iter().map(|&x| relu_derivative(x)))
                    } else if *f as usize == sigmoid as usize {
                        Vector::from_iter(layer_output.iter().map(|&x| sigmoid_derivative(x)))
                    } else if *f as usize == tanh as usize {
                        Vector::from_iter(layer_output.iter().map(|&x| tanh_derivative(x)))
                    } else if *f as usize == linear as usize {
                        Vector::from_iter(layer_output.iter().map(|&x| linear_derivative(x)))
                    } else {
                        return Err(NetworkError::ActivationMismatch("Unknown activation function"));
                    }
                },
                LayerActivation::Vector(f) => {
                    if *f as usize == softmax as usize {
                        softmax_derivative(layer_output)
                    } else {
                        return Err(NetworkError::ActivationMismatch("Unknown vector activation function"));
                    }
                }
            };
            
            delta = match &layer.activation {
                LayerActivation::ElementWise(_) => {
                    Vector::from_iter(delta.iter().zip(activation_derivative.iter()).map(|(&d, &ad)| d * ad))
                },
                LayerActivation::Vector(_) => {
                    Vector::from_iter(delta.iter().zip(activation_derivative.iter()).map(|(&d, &ad)| d * ad))
                }
            };
            
            for (neuron, d) in layer.neurons.iter_mut().zip(delta.iter()) {
                neuron.weights = (&neuron.weights - &(layer_output * (learning_rate * d)))?;
                neuron.bias -= learning_rate * d;
            }
            
            if layer_index > 0 {
                delta = (&layer.to_matrix().transpose() * &delta)?;
            }
        }
        
        Ok(())
    }

    pub fn train(&mut self, inputs: &[Vector], targets: &[Vector], epochs: usize, learning_rate: f64) -> Result<(), NetworkError> {
        if inputs.len() != targets.len() {
            return Err(NetworkError::DimensionMismatch("Number of inputs and targets do not match"));
        }
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let output = self.feed_forward(input.clone())?;
                let loss = Self::mse_loss(&output, target)?;
                total_loss += loss;
                
                self.backpropagate(input, target, learning_rate)?;
            }
            
            let avg_loss = total_loss / inputs.len() as f64;
            println!("Epoch {}: Average Loss = {}", epoch + 1, avg_loss);
        }
        
        Ok(())
    }

}

impl IntoIterator for Network {
    type Item = Layer;
    type IntoIter = std::vec::IntoIter<Layer>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.into_iter()
    }
}

impl<'a> IntoIterator for &'a Network {
    type Item = &'a Layer;
    type IntoIter = std::slice::Iter<'a, Layer>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.iter()
    }
}

impl Index<usize> for Network {
    type Output = Layer;

    fn index(&self, index: usize) -> &Self::Output {
        &self.layers[index]
    }
}

impl IndexMut<usize> for Network {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.layers[index]
    }
}

