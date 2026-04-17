//! Print FizzBuzz from 1 through N (default 100).
//!
//! ```text
//! cargo run --example fizzbuzz -- 20
//! ```

use std::env;

fn main() {
    let max = env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    for s in ripgrep::fizzbuzz::lines(max) {
        println!("{s}");
    }
}
