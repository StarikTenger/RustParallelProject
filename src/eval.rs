use std::{io::BufRead, usize};
use full_palette::ORANGE_A400;

use std::fs::OpenOptions;
use std::io::Write;
use plotters::prelude::*;

use crate::gen_matrix;
use crate::count_negatives_seq;
use crate::count_negatives_par;
use crate::count_negatives_par_iter;
use crate::measure_time;




// sudo apt-get install libfontconfig libfontconfig1-dev

// Plot the time taken to count negatives for different matrix sizes
pub fn measure_time_count_neg_sizes(sizes: &Vec<usize>, segment_sizes: &Vec<usize>, file_name: &str, steps: usize) {
    let root = BitMapBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Time to count negatives", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            (sizes[0] as f64).log2()..(*sizes.last().unwrap() as f64).log2(),
            0.0001_f64.log10()..0.01_f64.log10(),
        )
        .unwrap();

    chart.configure_mesh()
        .x_desc("Matrix Size (log2 scale)")
        .y_desc("Time (log scale)")
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| format!("{:.0}", 2_f64.powf(*x)))
        .y_label_formatter(&|y| format!("{:.4}", 10_f64.powf(*y)))
        .draw()
        .unwrap();

    for segment in segment_sizes.iter().enumerate() {
        let (segment_idx, &segment_size) = segment;

        let mut times_seq = Vec::new();
        let mut times_par = Vec::new();
        let mut times_par_iter = Vec::new();

        println!("\n\nSegment size: {}", segment_size);
        let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open("timings.txt")
                .unwrap();

        // Color palettes
        let palette_par = 
        [
            &BLUE,
            &CYAN,
            &MAGENTA,
            &RGBColor(0, 128, 255),
            &RGBColor(128, 0, 255),
        ];

        let palette_par_iter = 
        [
            &GREEN,
            &ORANGE_A400,
            &YELLOW,
            &RGBColor(128, 255, 0),
            &RGBColor(255, 128, 0),
        ];

        for &size in sizes {
            println!("\nSize: {}", size);

            let mut avg_time_seq = 0.0;
            let mut avg_time_par = 0.0;
            let mut avg_time_par_iter = 0.0;

            for i in 0..steps {
                println!("Step: {}/{}", i + 1, steps);

                let (matrix, _) = gen_matrix(size, size);

                let (time_seq, res_seq) = measure_time(|| {
                    count_negatives_seq(&matrix)
                });
                avg_time_seq += time_seq.as_secs_f64();

                let (time_par, res_par) = measure_time(|| {
                    count_negatives_par(&matrix, segment_size)
                });
                avg_time_par += time_par.as_secs_f64();

                let (time_par_iter, res_par_iter) = measure_time(|| {
                    count_negatives_par_iter(&matrix, segment_size)
                });
                avg_time_par_iter += time_par_iter.as_secs_f64();

                // Check if results are correct
                if !(res_seq == res_par && res_par == res_par_iter) {
                    println!("Error: seq = {}, par = {}, par_iter = {}", res_seq, res_par, res_par_iter);
                    return;
                }
            }

            avg_time_seq /= steps as f64;
            avg_time_par /= steps as f64;
            avg_time_par_iter /= steps as f64;

            println!("Size: {}, Segment size: {}, Seq: {}, Par: {}, ParIter: {}", size, segment_size, avg_time_seq, avg_time_par, avg_time_par_iter);
            writeln!(file, "Size: {}, Segment size: {}, Seq: {}, Par: {}, ParIter: {}", size, segment_size, avg_time_seq, avg_time_par, avg_time_par_iter).unwrap();


            times_seq.push((size as f64, avg_time_seq));
            times_par.push((size as f64, avg_time_par));
            times_par_iter.push((size as f64, avg_time_par_iter));
        }

        if segment_idx == 0 {
            chart
                .draw_series(LineSeries::new(
                    times_seq.into_iter().map(|(x, y)| (x.log2(), y.log10())),
                    &RED,
                ))
                .unwrap()
                .label(format!("Sequential"))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        }
        

        chart
            .draw_series(LineSeries::new(
                times_par.into_iter().map(|(x, y)| (x.log2(), y.log10())),
                palette_par[segment_idx],
            ))
            .unwrap()
            .label(format!("Parallel (segment size {})", segment_size))
            .legend({
                let color = palette_par[segment_idx].clone();
                move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color)
            });

        chart
            .draw_series(LineSeries::new(
                times_par_iter.into_iter().map(|(x, y)| (x.log2(), y.log10())),
                palette_par_iter[segment_idx],
            ))
            .unwrap()
            .label(format!("Parallel Iterator (segment size {})", segment_size))
            .legend({
                let color = palette_par_iter[segment_idx].clone();
                move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color)
            });
    }

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()
        .unwrap();
}
