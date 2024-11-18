use std::io::BufRead;

use crate::gen_matrix;
use crate::count_negatives_seq;
use crate::count_negatives_par;

pub fn display_matrix(matrix: &Vec<Vec<i32>>) {
    for row in matrix {
        for &value in row {
            print!("{:4} ", value); 
        }
        println!();
    }
}

// Evaluate the average sum of the matrix for a given number of steps
// If matrix is generated correctly, the average sum should be around 0
pub fn evaluate_generation(size: usize, steps: usize) -> i32 {
    let mut total_sum = 0;

    for _ in 0..steps {
        let (matrix, _) = gen_matrix(size, size);
        let sum: i32 = matrix.iter().flat_map(|row| row.iter()).sum();
        total_sum += sum;
    }

    total_sum / (size * size * steps) as i32
}

pub fn load_matrix_from_file(file_name: &str) -> Vec<Vec<i32>> {
    let file = std::fs::File::open(file_name).expect("File not found");
    let reader = std::io::BufReader::new(file);
    let mut matrix = Vec::new();
    let mut row_length = None;

    for line in reader.lines() {
        let row: Vec<i32> = line.unwrap()
                                .split_whitespace()
                                .map(|s| s.parse().unwrap())
                                .collect();
        if let Some(len) = row_length {
            if row.len() != len {
                panic!("Inconsistent row length in matrix");
            }
        } else {
            row_length = Some(row.len());
        }
        matrix.push(row);
    }

    matrix
}

pub fn evaluate_count_negatives(size: usize, steps: usize) {
    let pb = indicatif::ProgressBar::new(steps as u64);
    println!("Evaluating {}x{} matrix for {} steps", size, size, steps);
    for _ in 0..steps {
        let (matrix, sum_true) = gen_matrix(size, size);
        let sum_seq = count_negatives_seq(&matrix);
        let sum_par = count_negatives_par(&matrix, 100);
        if !(sum_seq == sum_par && sum_true == sum_seq) {
            println!("Error: seq = {}, par = {}", sum_seq, sum_par);
            return;
        }
        pb.inc(1);
    }
    pb.finish_with_message("done");
    println!("All results are correct");
}

pub fn measure_time<F, T>(f: F) -> (std::time::Duration, T)
where
    F: FnOnce() -> T
{
    let start = std::time::Instant::now();
    let res = f();
    (start.elapsed(), res)
}

pub fn mat_to_png(matrix: &Vec<Vec<i32>>, file_name: &str) {
    let mut imgbuf = image::ImageBuffer::new(matrix[0].len() as u32, matrix.len() as u32);

    let max_value = matrix.iter().flat_map(|row| row.iter()).cloned().max().unwrap_or(1);
    let min_value = matrix.iter().flat_map(|row| row.iter()).cloned().min().unwrap_or(0);

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let value = matrix[y as usize][x as usize];
        let normalized_value = if value < 0 {
            (255.0 * (value as f64 / min_value as f64).abs()) as u8
        } else {
            (255.0 * (1.0 - (value as f64 / max_value as f64).abs())) as u8
        };

        if value < 0 {
            *pixel = image::Rgb([normalized_value, normalized_value, normalized_value]);
        } else {
            *pixel = image::Rgb([normalized_value, normalized_value, normalized_value]);
        }
    }

    imgbuf.save(file_name).unwrap();
}