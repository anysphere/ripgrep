//! Deliberately defective code for testing CI (lint gates, failure reporting, etc.).
//! This crate is **not** a workspace member; build it explicitly:
//! `cargo check --manifest-path ci/fixtures/intentionally_buggy/Cargo.toml`

#![warn(unused)]
#![warn(clippy::all)]

#[cfg(feature = "force_compile_error")]
compile_error!("intentional compile_error for CI fixture testing");

/// Returns `a / b`. Panics when `b == 0` — intentional logic defect.
pub fn sloppy_div(a: i32, b: i32) -> i32 {
    a / b
}

// Dead code: should trigger `dead_code` under `-D warnings` / strict CI.
fn never_called() -> u32 {
    42
}

pub fn unused_binding() -> i32 {
    let x = 1;
    let noise = 99; // unused variable: should warn
    x + 2
}
