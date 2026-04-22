//! Spot checks that only run (and fail) on selected cross targets.

#[cfg(target_arch = "riscv64")]
#[test]
fn ci_spot_fail_riscv64() {
    assert!(false, "intentional failure on riscv64 CI only");
}

#[cfg(target_arch = "s390x")]
#[test]
fn ci_spot_fail_s390x() {
    assert!(false, "intentional failure on s390x CI only");
}
