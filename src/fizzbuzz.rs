//! Classic FizzBuzz: multiples of 3 print "Fizz", 5 print "Buzz", both "FizzBuzz".

/// Returns the FizzBuzz line for `n` (1-based).
pub fn line(n: u32) -> String {
    match (n % 3 == 0, n % 5 == 0) {
        (true, true) => "FizzBuzz".into(),
        (true, false) => "Fizz".into(),
        (false, true) => "Buzz".into(),
        (false, false) => n.to_string(),
    }
}

/// Lines for `n` = 1 through `max` inclusive.
pub fn lines(max: u32) -> impl Iterator<Item = String> {
    (1..=max).map(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_samples() {
        assert_eq!(line(1), "1");
        assert_eq!(line(3), "Fizz");
        assert_eq!(line(5), "Buzz");
        assert_eq!(line(15), "FizzBuzz");
    }

    #[test]
    fn first_twenty() {
        let got: Vec<_> = lines(20).collect();
        assert_eq!(got[0], "1");
        assert_eq!(got[2], "Fizz");
        assert_eq!(got[4], "Buzz");
        assert_eq!(got[14], "FizzBuzz");
        assert_eq!(got.len(), 20);
    }
}
