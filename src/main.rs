use std::{io::BufRead, usize};
use std::cmp::max;
use full_palette::ORANGE_A400;
use rand::Rng;
use rayon::prelude::*;
use diam::join;
use indicatif;
use image;
use noise::{NoiseFn, Perlin, Seedable};
use std::fs::OpenOptions;
use std::io::Write;
use plotters::prelude::*;

fn find_first_negative(nums: &Vec<i32>) -> i32 {
    let mut left = 0;
    let mut right = nums.len() as i32 - 1;

    while left <= right {
        let mid = left + (right - left) / 2;
        
        if nums[mid as usize] < 0 {
            if mid == 0 || nums[mid as usize - 1] >= 0 {
                return mid;
            }
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    // If no negative number is found, return -1
    -1
}

fn count_negatives_segment(grid: &Vec<Vec<i32>>, begin: usize, end: usize) -> i32 {
    let rows: usize = grid.len();
    let columns = if rows > 0 { grid[0].len() } else { 0 };
    let gap = end - begin;
    let mut sum : i32 = (gap as i32) * (columns as i32);
    

    let first_negative = find_first_negative(&grid[end - 1]);
    if first_negative == -1 {
        return 0;
    }

    let mut i : usize = end - 1;
    let mut j : usize = first_negative as usize;

    loop {
        while j < columns && grid[i][j] >= 0 {
            j += 1;
        }
        sum -= j as i32;

        if i == begin {
            break;
        }

        i -= 1;
    }
    sum
}

fn count_negatives_seq(grid: &Vec<Vec<i32>>) -> i32 {
    count_negatives_segment(grid, 0, grid.len())
}

fn count_negatives_par_inner(grid: &Vec<Vec<i32>>, begin: usize, end: usize, segment_size: usize) -> i32 {
    if end - begin <= segment_size {
        let res = count_negatives_segment(grid, begin, end);
        return res;
    }

    let mut a : i32 = 0;
    let mut b : i32 = 0;
    let mid = (begin + end) / 2;

    join(
        || a = count_negatives_par_inner(grid, begin, mid, segment_size),
        || b = count_negatives_par_inner(grid, mid, end, segment_size)
    );

    a + b
}

fn count_negatives_par(grid: &Vec<Vec<i32>>, segment_size: usize) -> i32 {
    count_negatives_par_inner(grid, 0, grid.len(), segment_size)
}

fn count_negatives_par_iter(grid: &Vec<Vec<i32>>, segment_size: usize) -> i32 {
    grid.par_chunks(segment_size)
        .enumerate()
        .map(|(i, segment)| {
            let begin = i * segment_size;
            let end = begin + segment.len();
            count_negatives_segment(grid, begin, end)
        })
        .sum()
}

fn display_matrix(matrix: &Vec<Vec<i32>>) {
    for row in matrix {
        for &value in row {
            print!("{:4} ", value); 
        }
        println!();
    }
}

fn get_noize_val(x: f64, y: f64, noise_palette: &Vec<Perlin>) -> f64 {
    let mut res: f64 = 0.0;
    let mut freq: f64 = 1.0;
    for noise in noise_palette {
        let val = noise.get([x * freq, y * freq, 0.0]);
        res += val / freq;
        freq *= 2.0;
    }
    
    res
}

fn gen_matrix(rows: usize, columns: usize) -> (Vec<Vec<i32>>, i32) {
    let mut matrix = vec![vec![0; columns]; rows];

    // Create a noise palette
    let mut noise_palette = Vec::new();
    let noise_levels = 1;
    for _ in 0..noise_levels {
        let seed = rand::thread_rng().gen_range(0..1000);
        let noise = Perlin::new(seed);
        noise_palette.push(noise);
    }

    println!("Generating {}x{} matrix", rows, columns);
    let pb = indicatif::ProgressBar::new((rows * columns) as u64);

    let update_rate = 1000;
    let mut iters = 0;

    // Decrease values in the matrix
    for i in 0..rows {
        for j in 0..columns {
            let rand_val = get_noize_val(i as f64 / rows as f64, j as f64 / columns as f64, &noise_palette);
            let threshold = 0.0;
            let rand_val = if rand_val < threshold { threshold } else { rand_val };
            let rand_val = (rand_val * 100.0) as i32;
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

// Evaluate the average sum of the matrix for a given number of steps
// If matrix is generated correctly, the average sum should be around 0
fn evaluate_generation(size: usize, steps: usize) -> i32 {
    let mut total_sum = 0;

    for _ in 0..steps {
        let (matrix, _) = gen_matrix(size, size);
        let sum: i32 = matrix.iter().flat_map(|row| row.iter()).sum();
        total_sum += sum;
    }

    total_sum / (size * size * steps) as i32
}

fn load_matrix_from_file(file_name: &str) -> Vec<Vec<i32>> {
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

fn evaluate_count_negatives(size: usize, steps: usize) {
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

fn measure_time<F, T>(f: F) -> (std::time::Duration, T)
where
    F: FnOnce() -> T
{
    let start = std::time::Instant::now();
    let res = f();
    (start.elapsed(), res)
}

fn mat_to_png(matrix: &Vec<Vec<i32>>, file_name: &str) {
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

// sudo apt-get install libfontconfig libfontconfig1-dev

// Plot the time taken to count negatives for different matrix sizes
fn measure_time_count_neg_sizes(sizes: &Vec<usize>, segment_sizes: &Vec<usize>, file_name: &str, steps: usize) {
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

// How to run with rayon and specify num of threads?
// RAYON_NUM_THREADS=4 cargo run --release

fn main() {

    measure_time_count_neg_sizes(
        &vec![1000, 2000, 4000, 8000, 16000, 32000],
        &vec![500, 1000, 2000, 4000],
        "count-neg-plot.png",
        10
    );

    // let matSize = 100;
    // let (gen_time, (m, sum_true)) = measure_time(|| {
    //     gen_matrix(matSize * 10000, matSize)
    // });
    // if matSize < 20 {
    //     display_matrix(&m);
    // }
    // println!("Time taken to generate matrix: {:?}", gen_time);
    // //mat_to_png(&m, "count-neg.png");

    // // Sequential
    // let mut res_seq = 0;
    // let (time_seq, _) = measure_time(|| {
    //     res_seq = count_negatives_seq(&m);
    // });

    // let segment_size = 100;

    // // Parallel with join
    // let mut res_par = 0;
    // let (time_par, _) = measure_time(|| {
    //     diam::svg("count-neg.svg", || {
    //         res_par = count_negatives_par(&m, segment_size);
    //     }).expect("failed saving svg file");
    // });

    // // Parallel with iterator
    // let mut res_par_iter = 0;
    // let (time_par_iter, _) = measure_time(|| {
    //     res_par_iter = count_negatives_par_iter(&m, segment_size);
    // });

    // // Print the results and time taken
    // println!("Number of negatives: {}", sum_true);
    // println!("Sequential result:   {}", res_seq);
    // println!("Parallel result:     {}", res_par);
    // println!("Time taken for sequential:             {:?}", time_seq);
    // println!("Time taken for parallel:               {:?}", time_par);
    // println!("Time taken for parallel with iterator: {:?}", time_par_iter);
    // // Speedup
    // println!("Speedup:               {}", time_seq.as_secs_f64() / time_par.as_secs_f64());
    // println!("Speedup with iterator: {}", time_seq.as_secs_f64() / time_par_iter.as_secs_f64());
}
