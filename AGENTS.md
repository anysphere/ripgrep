# AGENTS.md

## Cursor Cloud specific instructions

This is a Cursor fork of ripgrep (`rg`), a Rust CLI tool (no web server or ports). Version: `15.1.0-cursor4`.

### Rust toolchain

- Requires **Rust 1.85+** (edition 2024). The VM's default Rust may be older; the update script handles installing the correct version.

### System dependencies

Tests for compressed-file searching require: `zsh`, `xz-utils`, `liblz4-tool`, `musl-tools`, `brotli`, `zstd`, `g++`. These are installed by the update script.

### Key commands

| Task | Command |
|------|---------|
| Build | `cargo build` |
| Test | `cargo test --all` |
| Lint | `cargo clippy --all` |
| Run | `cargo run -- <args>` or `./target/debug/rg <args>` |

### Notes

- This is a pure Rust/Cargo workspace with 9 library crates under `crates/` and the binary at `crates/core/main.rs`.
- `cargo clippy --all` produces ~64 style warnings (pre-existing); these are not errors.
- The optional `pcre2` feature (`cargo build --features pcre2`) requires `libpcre2` C library; it is not needed for default builds.
- Integration tests are at `tests/tests.rs`; unit tests are in each crate.
