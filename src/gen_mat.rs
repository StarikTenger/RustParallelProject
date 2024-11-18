use std::{io::BufRead, usize};
use std::cmp::max;
use rand::Rng;
use indicatif;
use noise::{NoiseFn, Perlin, Seedable};
use plotters::prelude::*;

pub fn get_noize_val(x: f64, y: f64, noise_palette: &Vec<Perlin>) -> f64 {
    let mut res: f64 = 0.0;
    let mut freq: f64 = 1.0;
    for noise in noise_palette {
        let val = noise.get([x * freq, y * freq, 0.0]);
        res += val / freq;
        freq *= 2.0;
    }
    
    res
}

pub fn gen_source_matrix(rows: usize, columns: usize) -> Vec<Vec<i32>> {
    let mut matrix = vec![vec![0; columns]; rows];

    // Create a noise palette
    let mut noise_palette = Vec::new();
    let noise_levels = 1;
    for _ in 0..noise_levels {
        let seed = rand::thread_rng().gen_range(0..1000);
        let noise = Perlin::new(seed);
        noise_palette.push(noise);
    }

    // Decrease values in the matrix
    for i in 0..rows {
        for j in 0..columns {
            let rand_val = get_noize_val(i as f64 / rows as f64, j as f64 / columns as f64, &noise_palette);
            let threshold = 0.0;
            let rand_val = if rand_val < threshold { threshold } else { rand_val };
            let rand_val = (rand_val * 100.0) as i32;
            matrix[i][j] = rand_val;
            
        }

    }

    matrix
}

pub fn gen_decreasing_matrix(source_mat: &Vec<Vec<i32>>) -> (Vec<Vec<i32>>, i32) {
    let rows = source_mat.len();
    let columns = source_mat[0].len();
    let mut matrix = vec![vec![0; columns]; rows];

    println!("Generating {}x{} matrix", rows, columns);
    let pb = indicatif::ProgressBar::new((rows * columns) as u64);

    let update_rate = 1000;
    let mut iters = 0;

    // Decrease values in the matrix
    for i in 0..rows {
        for j in 0..columns {
            let rand_val = source_mat[i][j];
            if i == 0 && j == 0 {
                matrix[i][j] = 0;
            } else if i == 0 {
                matrix[i][j] = matrix[i][j - 1] + rand_val;
            } else if j == 0 {
                matrix[i][j] = matrix[i - 1][j] + rand_val;
            } else {
                matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1]) + rand_val;
            }
            iters += 1;
            if iters % update_rate == 0 {
                pb.inc(update_rate as u64);
                iters = 0;
            }
            
        }

    }

    pb.finish_with_message("Matrix generation complete");
    println!("Matrix generation complete");

    let top = matrix[rows - 1][columns - 1];
    let mut count_neg = 0;
    for i in 0..rows {
        for j in 0..columns {
            matrix[i][j] = matrix[i][j] - ((top / 2) as f64) as i32;
            matrix[i][j] = -matrix[i][j];

            if matrix[i][j] < 0 {
                count_neg += 1;
            }
        }
    }

    (matrix, count_neg)
}

pub fn gen_matrix(rows: usize, columns: usize) -> (Vec<Vec<i32>>, i32) {
    let source_mat = gen_source_matrix(rows, columns);
    gen_decreasing_matrix(&source_mat)
}