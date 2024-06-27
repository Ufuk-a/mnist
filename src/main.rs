mod vector;
mod matrix;
mod neuron;
mod layer;
mod network;

use network::Network;
use vector::Vector;
use matrix::MatrixError;

fn main() -> Result<(), MatrixError> {
    let mut structure = vec![10, 5, 5, 2];
    let w_range = -0.1..0.1;
    let b_range = -0.1..0.1;

    let mlp = Network::random(&mut structure, w_range, b_range);

    let input = Vector::from_vec(vec![2.0, 3.0, 5.0, 1.0, 4.0, 7.0, 4.0, 1.0, 4.0, 4.0]);
    let output = mlp.feed_forward(input)?;
    println!("{:?}", output);

    Ok(())
}