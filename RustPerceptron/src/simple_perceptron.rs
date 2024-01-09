use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::f64;

pub struct Perceptron {
    learning_rate: f64,
    n_iters: usize,
    weights: Array1<f64>,
    bias: f64,
}

impl Perceptron {
    pub fn new(learning_rate: f64, n_iters: usize, n_features: usize) -> Self {
        Perceptron {
            learning_rate,
            n_iters,
            weights: Array1::zeros(n_features),
            bias: 0.0,
        }
    }

    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) {
        let n_samples = x.nrows();

        for _ in 0..self.n_iters {
            for idx in 0..n_samples {
                let x_i = x.row(idx);
                let y_i = y[idx];

                let linear_output = x_i.dot(&self.weights) + self.bias;
                let y_predicted = relu(linear_output);

                let update = self.learning_rate * (y_i - y_predicted);
                self.weights += &(&x_i * update);
                self.bias += update;
            }
        }
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let n_samples = x.nrows();

        let mut y_pred = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let x_i = x.row(i);

            let linear_output = x_i.dot(&self.weights) + self.bias;
            y_pred[i] = relu(linear_output);
        }

        y_pred
    }
}

fn relu(z: f64) -> f64 {
    z.max(0.0)
}