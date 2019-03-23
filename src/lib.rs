extern crate libc;

use libc::{int32_t, size_t};
use std::mem;
use std::slice;

use std::thread;

#[no_mangle]
fn process() {
    let handles: Vec<_> = (0..10)
        .map(|_| {
            thread::spawn(|| {
                let mut x = 0;
                for _ in (0..5_000_000) {
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
    let nums = unsafe { array.as_u32_slice() };
    let data: Vec<u32> = nums.iter().map(|a| a * a).collect();
    Array::from_vec(data)
}

#[repr(C)]
pub struct Tuple {
    a: libc::uint32_t,
    b: libc::uint32_t,
}

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
