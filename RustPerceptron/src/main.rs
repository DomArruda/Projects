
mod simple_perceptron;


use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::f64;
use simple_perceptron::Perceptron;

fn main() {
    let x: Array2<f64> = Array2::random((100, 10), Uniform::new(0., 1.));
    let y: Array1<f64> = x.sum_axis(ndarray::Axis(1)) + Array1::random(100, Uniform::new(0., 0.1));

    let mut p = Perceptron::new(0.1, 1000, 10);
    p.fit(x.view(), y.view());

    let predictions = p.predict(x.view());

    println!("Perceptron MSE: {}", mse(&y, &predictions));
    println!("Perceptron MAPE: {}", mape(&y, &predictions));
}

fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    ((y_true - y_pred).mapv(|x| x.powi(2))).mean().unwrap()
}

fn mape(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    ((y_true - y_pred).mapv(f64::abs) / y_true).mean().unwrap() * 100.0
}
