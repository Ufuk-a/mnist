use std::ops::{Range, Index, IndexMut};

use crate::vector::Vector;
use crate::matrix::MatrixError;
use crate::layer::Layer;

pub struct Network {
    pub layers: Vec<Layer>
}

impl Network {
    pub fn new(structure:&mut Vec<usize>, weight: f64, bias: f64) -> Self {
        structure.insert(0, 1);
        let layers = structure.windows(2).map(|window| Layer::new(window[1], window[0], weight, bias)).collect();
        Network{layers}
    }

    pub fn random(structure:&mut Vec<usize>, w_range: Range<f64>, b_range: Range<f64>) -> Self {
        let mut layers: Vec<Layer> = Vec::new();

        layers.push(Layer::new(structure[0], 1, 1.0, 0.0));
        
        layers.extend(structure.windows(2).map(|window| Layer::random(window[1], window[0], w_range.clone(), b_range.clone())));

        Network{layers}
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

    pub fn feed_forward(&self, input: Vector) -> Result<Vector, MatrixError> {
        let mut result = input;
        let mut is_input = true;
        for layer in self {
            if is_input {
                is_input = false;
                continue;
            }
            
            println!("{:?}", result);
            let w_matrix = layer.to_matrix();
            result = (w_matrix * result)?;  
        }
        Ok(result)
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