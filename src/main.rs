mod vector;
mod matrix;
mod neuron;
mod layer;
mod network;
mod activation;

use network::{Network, NetworkError};
use vector::Vector;
use activation::sigmoid;
use layer::LayerActivation;
use mnist::MnistBuilder;

fn main() -> Result<(), NetworkError> {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50000)
        .validation_set_length(10000)
        .test_set_length(10000)
        .finalize();

    let train_data = mnist.trn_img;
    let train_labels = mnist.trn_lbl;
    let test_data = mnist.tst_img;
    let test_labels = mnist.tst_lbl;

    let train_data: Vec<Vector> = train_data.chunks(784).map(|chunk| Vector::from_vec(chunk.iter().map(|&x| x as f64 / 255.0).collect())).collect();

    let test_data: Vec<Vector> = test_data.chunks(784).map(|chunk| Vector::from_vec(chunk.iter().map(|&x| x as f64 / 255.0).collect())).collect();

    let train_labels: Vec<Vector> = train_labels.iter().map(|&label| { 
        let mut v = Vector::new(10, 0.0);
        v[label as usize] = 1.0;
            v
        })
        .collect();

    //let structure = vec![784, 16, 16, 10];
    //let w_range = -0.2..0.2;
    //let b_range = -0.2..0.2;
    //let activation_config = vec![
    //    (LayerActivation::ElementWise(sigmoid), 3)
    //];
    //let mut mlp = Network::random(&structure, w_range, b_range, activation_config)?;

    //println!("Starting training...");
    //mlp.train(&train_data, &train_labels, 10, 0.001)?; //325
    let mlp = Network::deserialize("output.txt").expect("Error");


    println!("Testing the network...");
    let mut correct = 0;
    for (input, &label) in test_data.iter().zip(test_labels.iter()) {
        let output = mlp.feed_forward(input.clone())?;
        let predicted = output.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        if predicted == label as usize {
            correct += 1;
        }
    }

    mlp.serialize("output.txt").expect("Issue");

    let accuracy = correct as f64 / test_labels.len() as f64;
    println!("Test accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}