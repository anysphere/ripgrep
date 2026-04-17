//! Intentionally failing tests for exercising CI failure reporting.
//! Remove this module (and `mod ci_fixture` in `tests.rs`) before merging to the default branch.

#[test]
fn deliberate_ci_failure() {
    assert_eq!(2 + 2, 5);
}
