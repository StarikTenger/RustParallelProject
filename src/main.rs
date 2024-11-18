use std::{io::BufRead, usize};
use full_palette::ORANGE_A400;

use std::fs::OpenOptions;
use std::io::Write;
use plotters::prelude::*;

mod gen_mat;
use gen_mat::*;

mod calc_neg;
use calc_neg::*;

mod util;
use util::*;

mod eval;
use eval::*;

fn main() {

    // With this parameters it was run in a report
    // WARNING: This will take a long time to run, if you want
    // to run it, decrease the number of steps, sizes and segment sizes
    // measure_time_count_neg_sizes(
    //     &vec![1000, 2000, 4000, 8000, 16000, 32000],
    //     &vec![500, 1000, 2000, 4000],
    //     "count-neg-plot.png",
    //     10
    // );


    let mat_size = 1000;
    let m_source = gen_source_matrix(mat_size, mat_size);
    let (gen_time, (m, sum_true)) = measure_time(|| {
        gen_decreasing_matrix(&m_source)
    });
    if mat_size < 20 {
        display_matrix(&m);
    }
    println!("Time taken to generate matrix: {:?}", gen_time);
    mat_to_png(&m_source, "count-neg-source.png");
    mat_to_png(&m, "count-neg.png");

    // Sequential
    let mut res_seq = 0;
    let (time_seq, _) = measure_time(|| {
        res_seq = count_negatives_seq(&m);
    });

    let segment_size = 100;

    // Parallel with join
    let mut res_par = 0;
    let (time_par, _) = measure_time(|| {
        diam::svg("count-neg.svg", || {
            res_par = count_negatives_par(&m, segment_size);
        }).expect("failed saving svg file");
    });

    // Parallel with iterator
    let mut res_par_iter = 0;
    let (time_par_iter, _) = measure_time(|| {
        res_par_iter = count_negatives_par_iter(&m, segment_size);
    });

    // Print the results and time taken
    println!("Number of negatives: {}", sum_true);
    println!("Sequential result:   {}", res_seq);
    println!("Parallel result:     {}", res_par);
    println!("Time taken for sequential:             {:?}", time_seq);
    println!("Time taken for parallel:               {:?}", time_par);
    println!("Time taken for parallel with iterator: {:?}", time_par_iter);
    // Speedup
    println!("Speedup:               {}", time_seq.as_secs_f64() / time_par.as_secs_f64());
    println!("Speedup with iterator: {}", time_seq.as_secs_f64() / time_par_iter.as_secs_f64());
}
