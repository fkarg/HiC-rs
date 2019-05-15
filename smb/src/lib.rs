// #![feature(type_ascription)]
#![allow(dead_code)]
#![allow(unused_attributes)]


extern crate libc;
extern crate num;
// extern crate backtrace;


use num::traits::{zero, Num};

// use libc::{int32_t, size_t};
use std::mem;
use std::slice;

use std::thread;

// use backtrace::Backtrace;

#[no_mangle]
fn process() {
    let handles: Vec<_> = (0..10)
        .map(|_| {
            thread::spawn(|| {
                let mut x = 0;
                for _ in 0..5_000_000 {
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

#[no_mangle]
fn double(x: i32) -> i32 {
    x + x
}

#[no_mangle]
// fn listtest(data: *const i32, length: usize) -> Array {
fn listtest(array: Array) -> Array {
    // let nums = unsafe { slice::from_raw_parts(data, length as usize) };
    // println!("Got to the Rust side");
    let nums = unsafe { array.as_u32_slice() };
    let data: Vec<u32> = nums.iter().map(|a| a * a).collect();
    Array::from_vec(data)
}

#[repr(C)]
pub struct Tuple {
    a: libc::uint32_t,
    b: libc::uint32_t,
}

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

    unsafe fn as_f32_slice(&self) -> &[f32] {
        assert!(!self.data.is_null());
        slice::from_raw_parts(self.data as *const f32, self.len as usize)
    }

    unsafe fn as_f64_slice(&self) -> &[f64] {
        assert!(!self.data.is_null());
        slice::from_raw_parts(self.data as *const f64, self.len as usize)
    }

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

    // #[allow(unused_assignments)]
    fn map_col<F>(&mut self, coln: usize, f: F)
    where
        F: Fn(&usize, &T) -> T,
        // giving it the element and the index (within that col)
    {
        let (ind_begin, ind_end) = (self.indptr[coln], self.indptr[coln + 1]);
        // let indices = &self.indices[ind_begin..(ind_end - 1)];
        let indices = &self.indices[ind_begin..ind_end];
        // let mut slice: &mut [T] = &mut self.data[ind_begin..ind_end];
        // let mut inter: Vec<T> = slice.iter_mut().zip(indices).map(|(d, i)| f(i, d)).collect();

        // #[allow(unused_assignments)]
        // slice = inter.as_mut_slice();
        // self.data[ind_begin..ind_end] = &*inter.as_mut_slice();

        // println!("Begin at: {}", ind_begin);
        for index in 0..indices.len() {
            self.data[self.indptr[coln] + index] = f(&index, &self.data[self.indptr[coln] + index]);
        }
    }

    fn map_col_dbg<F>(&self, coln: usize, f: F)
    where
        F: Fn(&usize, &T),
        // giving it the element and the index (within that col)
    {
        let (ind_begin, ind_end) = (self.indptr[coln], self.indptr[coln + 1]);
        let indices = &self.indices[ind_begin..ind_end];
        let slice = &self.data[ind_begin..ind_end];
        let _inter: Vec<_> = slice.iter().zip(indices).map(|(d, i)| f(i, d)).collect();
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

    fn to_vec(&self) -> Vec<Vec<T>> {
        let mut full = Vec::new();

        let size = self.indptr.len() - 1;
        for i in 0..size {
            full.push(self.get_col(i, zero()));
        }

        full
    }
}

#[derive(Debug)]
pub struct COOMatrix<T> {
    row: Vec<usize>,
    col: Vec<usize>,
    data: Vec<T>,
}

impl<T: Num + Clone> COOMatrix<T> {
    fn new(row: Vec<usize>, col: Vec<usize>, data: Vec<T>) -> Self {
        COOMatrix { row, col, data }
    }
}

#[no_mangle]
fn csrtest(indptr: Array, indices: Array, data: Array) -> Array {
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

    // println!("CSRMatrix From Rust:\n{:?}", matrix);
    // println!("Full CSRMatrix From Rust:\n{:?}", matrix.to_vec());

    iterative_correction(matrix)
}

fn iterative_correction(mut matrix: CSRMatrix<f64>) -> Array {
    println!("Starting iterative correction");
    let mut biases = Vec::new();
    biases.resize(matrix.length, 1.0);

    let sum: f64 = matrix.data.iter().sum();
    if sum.is_nan() {
        // TODO: log.warn
        println!(
            "[iterative correction] the matrix contains nans, they will be replaced by zeros."
        );
        matrix.data = matrix.data.iter().map(|&v| 0_f64.max(v)).collect();
    }

    let m = 50;
    let tolerance = 1e-5;

    // Description of algorithm from paper:
    //
    // We begin each iteration by calculating the coverage S_i = sum_j W_ij. Next, additional
    // biases ^B_i are calculated by renormalizing S_i to have the unit mean ^Bi = Si / mean(S_i).
    // We then divide W_ij by ^B_i*^B/j for all (i, j) and update the total vector of biases by
    // multiplying by the additional biases. Iterations are repeatet until the variance of the
    // additional biases becomes negligible; at this point W_ij has converged to T_ij.
    //
    // TODO: implement.

    // def iterativeCorrection(matrix, v=None, M=50, tolerance=1e-5, verbose=False):

    // DONE
    // total_bias = np.ones(matrix.shape[0], 'float64')

    // DONE
    // if np.isnan(matrix.sum()):
    //     log.warn("[iterative correction] the matrix contains nans, they will be replaced by zeros.")
    //     matrix.data[np.isnan(matrix.data)] = 0

    // TODO: check for symmetry:
    //
    // if np.abs(matrix - matrix.T).mean() / (1. * np.abs(matrix.mean())) > 1e-10:
    //     raise ValueError("Please provide symmetric matrix!")

    // TODO ?
    // W = matrix.tocoo()

    // DONE
    // for iternum in range(M):
    //     iternum += 1
    //     s = np.array(W.sum(axis=1)).flatten()
    //     mask = (s == 0)
    //     s = s / np.mean(s[~mask])

    //      DONE
    //     total_bias *= s
    //     deviation = np.abs(s - 1).max()
    //     s = 1.0 / s

    //      TODO
    //     # The following code  is an optimization of this
    //     # for i in range(N):
    //     #     for j in range(N):
    //     #         W[i,j] = W[i,j] / (s[i] * s[j])
    //
    //     W.data *= np.take(s, W.row)
    //     W.data *= np.take(s, W.col)

    //     TODO
    //     if np.any(W.data > 1e100):
    //         log.error("*Error* matrix correction is producing extremely large values. "
    //                   "This is often caused by bins of low counts. Use a more stringent "
    //                   "filtering of bins.")
    //         exit(1)

    //     if deviation < tolerance:
    //         break

    // TODO
    // # scale the total bias such that the sum is 1.0
    // corr = total_bias[total_bias != 0].mean()
    // total_bias /= corr
    // W.data = W.data * corr * corr
    // if np.any(W.data > 1e10):
    //     log.error("*Error* matrix correction produced extremely large values. "
    //               "This is often caused by bins of low counts. Use a more stringent "
    //               "filtering of bins.")
    //     exit(1)

    // let mut iternum = 0;
    let mut s: Vec<f64> = Vec::with_capacity(matrix.length);

    println!("{:?}", matrix);

    // DONE
    // for iternum in range(M):
    //     s = np.array(W.sum(axis=1)).flatten()
    //     mask = (s == 0)
    //     s = s / np.mean(s[~mask])
    for _i in 0..m {
        println!("[iteration]: {}", _i);
        for c in 0..matrix.length {
            s.push(matrix.get_col(c, 0_f64).iter().sum());
            println!("{:?}, sum: {}", matrix.get_col(c, 0_f64), s[c]);
        }
        s.iter().enumerate().filter(|(_i, &v)| v.is_infinite()).for_each(|(i, _)| println!("Found Inf at: {}", i));
        let s_sum: f64 = dbg!(s.iter().sum());
        let mean: f64 = ((s.len() as f64) / s_sum) as f64; // inverse of: S_i / mean(S_i)
        dbg!(s.len());
        dbg!(mean);
        s = s.iter().map(|&v: &f64| v * mean).collect();

        //  DONE
        // total_bias *= s
        // deviation = np.abs(s - 1).max()
        // s = 1.0 / s
        biases = biases.iter().zip(s.clone()).map(|(a, b)| a * b).collect();
        // biases = biases.iter().zip(s.clone()).map(|(a, b)| a * b).collect();
        // let deviation: f64 = s.iter().fold(0_f64, |a, b| a.max((b - 1.).abs()));
        let deviation: f64 = s.iter().fold(0_f64, |a, b| a.max((b - 1.).abs()));
        dbg!(deviation);
        s = dbg!(s).iter().map(|&v| 1.0 / v).collect();

        // DONE
        // # The following code  is an optimization of this
        // # for i in range(N):
        // #     for j in range(N):
        // #         W[i,j] = W[i,j] / (s[i] * s[j])
        //
        // W.data *= np.take(s, W.row)
        // W.data *= np.take(s, W.col)

        // Iterate through all cols, and through the data in each col.
        // 'Generate' the index, from this multiply the s-value accordingly.
        // Also, for each col, have the other s-value (s[j]) cached for multiplication.

        for coln in 0..matrix.length {
            let c = s[coln];
            if coln == 0 {
                println!("Before updating values, c: {}", c);
                matrix.map_col_dbg(0, |&i, &v| println!("v: {}, s[i]: {}", v, s[i]));
            }
            matrix.map_col(coln, |&i, v| v * c * s[i]);
            if coln == 0 {
                println!("After updating the values");
                matrix.map_col_dbg(0, |&i, &v| println!("v: {}, i: {}", v, i));
            }
            // let row_index = matrix.indices[begin_index..];
            // matrix.get_col_iter(coln).mut_iter().map
        }
        // DONE
        // if np.any(W.data > 1e100):
        //     log.error("*Error* matrix correction is producing extremely large values. "
        //               "This is often caused by bins of low counts. Use a more stringent "
        //               "filtering of bins.")
        //     exit(1)
        if matrix.data.iter().any(|&v| v > 1e100) {
            panic!("Error: matrix correction is producing extremely large values.")
        }

        // if deviation < tolerance:
        //     break
        if deviation < tolerance {
            println!("Deviation smaller tolerance! {} < {}", deviation, tolerance);
            break;
        }
    }

    // DONE
    // # scale the total bias such that the sum is 1.0
    // corr = total_bias[total_bias != 0].mean()
    // total_bias /= corr
    // W.data = W.data * corr * corr
    // if np.any(W.data > 1e10):
    //     log.error("*Error* matrix correction produced extremely large values. "
    //               "This is often caused by bins of low counts. Use a more stringent "
    //               "filtering of bins.")
    //     exit(1)

    s.clear();

    let (s, c) = biases
        .iter()
        .filter(|&v| *v > 0.0)
        .fold((0_f64, 0_f64), |(s, c), i| (s + i, c + 1.));
    let corr = s / c;
    let biases = biases.iter().map(|&v| v / corr).collect();

    println!("printing matrix data:");
    for i in matrix.data {
        if i > 1e10 {
            panic!("Error: matrix correction produced extremely large values");
        }
        // println!("{}", i);
    }

    Array::from_vec(biases)
}



#[cfg(test)]
pub mod tests {
    use super::{CSRMatrix, iterative_correction};

    #[test]
    pub fn test_iterative_correct() {
        let indptr = vec![0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4];
        let indices = vec![5, 4, 0, 9];
        let data = vec![1., 1., 1., 1.];
        let matrix = CSRMatrix::new(indptr, indices, data);
        println!("{:?}", matrix);
        let e = iterative_correction(matrix);
        println!("{:?}", e);
    }
}


