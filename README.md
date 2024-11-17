# Counting negatives in sorted matrix in parallel

## The problem

The problem on leetcode:
https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/

>Given a `m x n` matrix grid which is sorted in non-increasing order both row-wise and column-wise, return the number of negative numbers in grid.

The sequential solution for the problem is starting from bottom-left corner follow the frontier between positive and negatives and add up negatives to the left of it.
So we are moving right until we meet the negative number and after we add the number of negatives in this row to the `sum` and move up.

This way to complete the algorithm for $N \times M$ matrix we need $N$ horizontal and $M$ vertical steps. So the complexity of the algorithm is $O(N + M)$.

here is the sequential implementation.

```rust
fn count_negatives(grid: Vec<Vec<i32>>) -> i32 {
    let rows = grid.len(); // N
    let columns = if rows > 0 { grid[0].len() } else { 0 }; // M

    let mut sum : i32 = (rows as i32) * (columns as i32);

    let mut i : usize = rows - 1;
    let mut j : usize = 0;

    loop {
        while j < columns && grid[i][j] >= 0 { j += 1; }
        sum -= j as i32;
        if i == 0 { break; }
        i -= 1;
    }

    sum
}
```


## Parallel version

### Division into subtasks
We will split the matrix horizontaly into several blocks. In each the algorithm will calculate the number of negatives, after all the values are added up.

Applying the same algorithm as before for each block in inefficient as the complexity inside the block would be $O(M + S)$, where $S$ is the number of rows in each block. With the increase of the number of blocks $K$ the number of rows in the block decreases being $S = \frac{N}{K}$, however the number of columns $M$ is constant.

This way the time complexity is $O(N + M \cdot K)$.

The better option is to use binary search to find the intersection of frontier with the bottom edge of the block. Overall there will be $K$ binary search operations, $N$ vertical and $M$ horizontal steps.

The time complexity in this scenario is $O(N + M + K \cdot log_2M)$.

For a subtask we define function `count_negatives_segment` that calculates the number of negatives in the segment of the matrix.

![alt text](images/diagram.png)


### With join

The simplest way to implement the parallel version is to use `rayon` library with join function.

```rust
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
```

### With parallel iterators

The other way is using parallel iterators. Divide the matrix into chunks and calculate the number of negatives in each chunk, than reduce the results.

```rust
fn count_negatives_par_iter(grid: &Vec<Vec<i32>>, segment_size: usize) -> i32 {
    grid.chunks(segment_size)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|chunk| count_negatives_seq(&chunk.to_vec()))
        .reduce(|| 0, |a, b| a + b)
}
```

### Evaluation



## Bonus: generating matrix in parallel