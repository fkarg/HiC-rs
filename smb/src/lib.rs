use std::process::Command;

use num::traits::{zero, Num};

use std::mem;
use std::slice;

use std::thread;

/// Testing the spawning of processes. This operation is too fast to be a good
/// example for parallelization.
#[no_mangle]
fn process() {
    let handles: Vec<_> = (0..10)
        .map(|i| {
            thread::spawn(move || {
                let mut x = i;
                for _ in 0..500_000_000 {
                    x += 1
                }
                x
            })
        })
        .collect();

    for h in handles {
        println!(
            "Thread finished with count={}",
            h.join().map_err(|_| "Could not join a thread!").unwrap()
        );
    }
    println!("done!");
}

/// Also only test function.
#[no_mangle]
fn double(x: i32) -> i32 {
    x + x
}

/// For testing the passing of lists back and forth.
#[no_mangle]
fn listtest(array: Array) -> Array {
    // let nums = unsafe { slice::from_raw_parts(data, length as usize) };
    // println!("Got to the Rust side");
    let nums = unsafe { array.as_u32_slice() };
    let data: Vec<u32> = nums.iter().map(|a| a * a).collect();
    Array::from_vec(data)
}

/// For passing Tuples.
#[repr(C)]
pub struct Tuple {
    a: u32,
    b: u32,
}

/// For passing Arrays of yet unspecified type.
#[derive(Debug)]
#[repr(C)]
pub struct Array {
    data: *const libc::c_void,
    len: libc::size_t,
}

impl Array {
    unsafe fn as_u32_slice(&self) -> &[u32] {
        assert!(!self.data.is_null());
        slice::from_raw_parts(self.data as *const u32, self.len as usize)
    }

    #[allow(dead_code)]
    unsafe fn as_f32_slice(&self) -> &[f32] {
        assert!(!self.data.is_null());
        slice::from_raw_parts(self.data as *const f32, self.len as usize)
    }

    unsafe fn as_f64_slice(&self) -> &[f64] {
        assert!(!self.data.is_null());
        slice::from_raw_parts(self.data as *const f64, self.len as usize)
    }

    #[allow(dead_code)]
    unsafe fn as_usize_slice(&self) -> &[usize] {
        assert!(!self.data.is_null());
        slice::from_raw_parts(self.data as *const usize, self.len as usize)
    }

    fn from_vec<T>(mut vec: Vec<T>) -> Array {
        // Important to make length and capacity match
        // A better solution is to track both length and capacity
        vec.shrink_to_fit();

        let array = Array {
            data: vec.as_ptr() as *const libc::c_void,
            len: vec.len() as libc::size_t,
        };

        // Whee! Leak the memory, and now the raw pointer (and eventually C) is the owner.
        mem::forget(vec);
        array
    }
}

/// Another test function.
#[no_mangle]
pub extern "C" fn convert_vec(lon: Array, lat: Array) -> Array {
    let lon = unsafe { lon.as_u32_slice() };
    let lat = unsafe { lat.as_u32_slice() };

    let vec = lat
        .iter()
        .zip(lon.iter())
        .map(|(&lat, &lon)| Tuple { a: lat, b: lon })
        .collect();

    Array::from_vec(vec)
}

fn invert_if_not_zero(num: f64) -> f64 {
    if num == 0. {
        num
    } else {
        1.0 / num
    }
}

/// Implementation of Compressed Sparse Row Matrix. Barely has any features.
#[repr(C)]
#[derive(Debug)]
pub struct CSRMatrix<T> {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    length: usize,
    data: Vec<T>,
}

impl<T: Num + Clone> CSRMatrix<T> {
    fn new(indptr: Vec<usize>, indices: Vec<usize>, data: Vec<T>) -> Self {
        let len = indptr.len() - 1;
        CSRMatrix {
            indptr,
            indices,
            length: len,
            data,
        }
    }

    fn map_col<F>(&mut self, coln: usize, f: F)
    where
        F: Fn(&usize, &T) -> T,
        // giving it the element and the index (within that col)
    {
        let (ind_begin, ind_end) = (self.indptr[coln], self.indptr[coln + 1]);
        let indices = &self.indices[ind_begin..ind_end];

        for (index, row_index) in indices.iter().enumerate() {
            let data_index = self.indptr[coln] + index;
            self.data[data_index] = f(&row_index, &self.data[data_index]);
        }
    }

    fn get_col(&self, coln: usize, zero: T) -> Vec<T> {
        let size = self.indptr.len() - 1;
        let (ind_begin, ind_end) = (self.indptr[coln], self.indptr[coln + 1]);
        let mut col = vec![zero; size];

        let slice = self.indices[ind_begin..ind_end].to_vec();
        for i in 0..slice.len() {
            col[slice[i]] = self.data[ind_begin + i].clone();
        }

        col
    }

    #[allow(dead_code)]
    fn to_vec(&self) -> Vec<Vec<T>> {
        let mut full = Vec::new();

        let size = self.indptr.len() - 1;
        for i in 0..size {
            full.push(self.get_col(i, zero()));
        }

        full
    }
}

/// The actual function called from Python.
#[no_mangle]
fn wrapper_iterative_correct(indptr: Array, indices: Array, data: Array, numiter: i32) -> Array {
    let matrix = CSRMatrix::new(
        // indptr:
        unsafe { indptr.as_u32_slice() }
            .iter()
            .map(|&a| a as usize)
            .collect(),
        // indices:
        unsafe { indices.as_u32_slice() }
            .iter()
            .map(|&a| a as usize)
            .collect(),
        // data:
        unsafe { data.as_f64_slice() }
            .iter()
            .map(|&a| a as f64)
            .collect(),
    );

    iterative_correction(matrix, numiter)
}

/// Here is the iterative correction actually happening.
fn iterative_correction(mut matrix: CSRMatrix<f64>, numiter: i32) -> Array {
    println!("Starting iterative correction");

    let tolerance = 1e-5;

    let mut biases = Vec::new();

    biases.resize(matrix.length, 1.0);

    if matrix.data.iter().sum::<f64>().is_nan() {
        eprintln!(
            "[iterative correction] the matrix contains nans, they will be replaced by zeros."
        );
        matrix.data = matrix.data.iter().map(|&v| 0_f64.max(v)).collect();
    }

    // Description of algorithm from paper:
    //
    // We begin each iteration by calculating the coverage S_i = sum_j W_ij. Next, additional
    // biases ^B_i are calculated by renormalizing S_i to have the unit mean ^Bi = Si / mean(S_i).
    // We then divide W_ij by ^B_i*^B/j for all (i, j) and update the total vector of biases by
    // multiplying by the additional biases. Iterations are repeatet until the variance of the
    // additional biases becomes negligible; at this point W_ij has converged to T_ij.

    let mut biases = Vec::new();

    biases.resize(matrix.length, 1.0);

    let mut s: Vec<f64> = Vec::with_capacity(matrix.length);

    for _i in 0..numiter {
        println!("[iteration]: {}", _i);
        let mut s_sum: f64 = 0.0;

        for c in 0..matrix.length {
            let colsum = matrix.get_col(c, 0_f64).iter().sum();
            s_sum += colsum;
            s.push(colsum);
        }

        let mean: f64 = ((matrix.length as f64) / s_sum) as f64; // inverse of: S_i / mean(S_i)

        // would have been v / mean, but the mean is the inverse already.
        s = s.iter().map(|&v: &f64| v * mean).collect();

        biases = biases.iter().zip(s.clone()).map(|(a, b)| a * b).collect();

        let deviation: f64 = s
            .iter()
            // for parallel:
            // .iter_map()
            // .map(|a| (a - 1_f64).abs())
            // .reduce(|| 0_f64, |a, b| a.max(b));
            .fold(0_f64, |a, b| a.max((b - 1.).abs()));

        s = s.iter().map(|&v| invert_if_not_zero(v)).collect();

        // Iterate through all cols, and through the data in each col.
        // 'Generate' the index, from this multiply the s-value accordingly.
        // Also, for each col, have the other s-value (s[j]) cached for multiplication.

        // this would be way faster when parallel
        for (coln, c) in s.iter().enumerate().take(matrix.length) {
            matrix.map_col(coln, |&i, v| v * c * s[i]);
        }
        s.clear();

        // Sufficient if checked in the end.
        // if matrix.data.iter().any(|&v| v > 1e100) {
        //     // println!("{:?}", matrix);
        //     panic!("Error: matrix correction is producing extremely large values.")
        // }

        if deviation < tolerance {
            println!("Deviation smaller tolerance! {} < {}", deviation, tolerance);
            break;
        }

        // Printing information about current memory usage at the fifth
        // iteration. Here everything should have stabilized.
        if _i == 5 {
            let memusage = Command::new("free")
                .arg("-g")
                .output()
                .expect("failed to execute process");
            let var = String::from_utf8(memusage.stdout).unwrap();
            let out: Vec<&str> = var.split_whitespace().collect();
            let memfree = out[8];
            let memused = out[9];
            println!("Memory used: {}", memfree);
            println!("Memory free: {}", memused);
        }
    }
    drop(s); // not needed any further, was only used for temporary biases

    if matrix.data.iter().filter(|&v| *v > 1e10).any(|_| true) {
        panic!("Error: matrix correction produced extremely large values");
    }

    // scale the total bias such that the sum is 1.0

    let (s, c) = biases
        .iter()
        .filter(|&v| *v > 0.0)
        .fold((0_f64, 0_f64), |(s, c), i| (s + i, c + 1.));
    let corr = s / c;
    let biases = biases.iter().map(|&v| v / corr).collect();

    Array::from_vec(biases)
}

#[cfg(test)]
pub mod tests {
    use super::{iterative_correction, CSRMatrix};

    #[test]
    pub fn test_iterative_correct() {
        // let indptr  = vec![0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12];
        // let indices = vec![5 , 4 , 0 , 9 , 3 , 7 , 8 , 1 , 2 , 6, 3 , 4];
        let indptr = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let indices = vec![5, 4, 0, 9, 3, 7, 8, 1, 2, 6];
        let data = vec![1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.];
        let matrix = CSRMatrix::new(indptr, indices, data);
        println!("{:?}", matrix);
        let e = iterative_correction(matrix);
        println!("{:?}", e);
    }

    #[test]
    pub fn test_iterative_correct_2() {
        let indptr = vec![
            0, 1, 1, 1, 1, 2, 3, 3, 3, 5, 6, 7, 7, 9, 9, 10, 11, 11, 13, 13, 16, 16, 17, 17, 17,
            17, 19, 20, 23, 24, 25,
        ];
        let indices = vec![
            5, 4, 0, 19, 25, 9, 12, 10, 14, 12, 19, 17, 25, 8, 15, 21, 19, 8, 17, 27, 26, 28, 29,
            27, 27,
        ];
        let data = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let matrix = CSRMatrix::new(indptr, indices, data);
        println!("{:?}", matrix);
        let e = iterative_correction(matrix);
        println!("{:?}", e);
    }

    #[test]
    pub fn test_to_vec() {
        let indptr = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let indices = vec![5, 4, 0, 9, 3, 7, 8, 1, 2, 6];
        let data = vec![1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.];
        let matrix = CSRMatrix::new(indptr, indices, data);
        let vec = vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ];
        assert!(vec == matrix.to_vec());
    }
}
