//! Intentionally failing tests for exercising CI failure reporting.
//! Remove this module (and `mod ci_fixture` in `tests.rs`) before merging to the default branch.
//!
//! The failure is gated to **native glibc x86_64 Linux** only (typical `ubuntu-latest` jobs).
//! Other CI matrix rows (macOS, Windows, musl, 32-bit/cross Linux, etc.) skip this test so they stay green.

/// Only compiled on native x86_64 Linux with the GNU toolchain (not musl), i.e. Ubuntu `*-linux-gnu` hosts.
#[cfg(all(
    target_os = "linux",
    target_arch = "x86_64",
    not(target_env = "musl")
))]
#[test]
fn deliberate_ci_failure() {
    assert_eq!(2 + 2, 5);
}
