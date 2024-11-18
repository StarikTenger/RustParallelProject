use rand::Rng;
use rayon::prelude::*;
use diam::join;
use std::cmp::max;

pub fn find_first_negative(nums: &Vec<i32>) -> i32 {
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

pub fn count_negatives_seq(grid: &Vec<Vec<i32>>) -> i32 {
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

pub fn count_negatives_par(grid: &Vec<Vec<i32>>, segment_size: usize) -> i32 {
    count_negatives_par_inner(grid, 0, grid.len(), segment_size)
}

pub fn count_negatives_par_iter(grid: &Vec<Vec<i32>>, segment_size: usize) -> i32 {
    grid.par_chunks(segment_size)
        .enumerate()
        .map(|(i, segment)| {
            let begin = i * segment_size;
            let end = begin + segment.len();
            count_negatives_segment(grid, begin, end)
        })
        .sum()
}