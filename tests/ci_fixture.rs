//! Deliberately failing tests (CI / exercise fixtures).

#[test]
fn ci_wrong_sum() {
    assert_eq!(2 + 2, 5);
}

#[test]
fn ci_wrong_crate_name() {
    assert_eq!(env!("CARGO_PKG_NAME"), "not-ripgrep");
}

#[test]
fn ci_always_false() {
    assert!(false, "intentional failure");
}

#[test]
fn ci_slice_len() {
    let xs = [1u8, 2, 3];
    assert_eq!(xs.len(), 0);
}

#[test]
fn ci_wrong_string_contains() {
    let haystack = "ripgrep finds needles quickly";
    assert!(haystack.contains("slowly"));
}

#[test]
fn ci_wrong_result_state() {
    let parsed: Result<u8, _> = "not-a-number".parse();
    assert!(parsed.is_ok());
}

#[test]
fn ci_wrong_option_value() {
    let maybe_name = Some("ripgrep");
    assert_eq!(maybe_name, None);
}

#[test]
fn ci_wrong_remainder() {
    assert_eq!(10 % 3, 0);
}
