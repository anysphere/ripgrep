#!/usr/bin/env node
/**
 * Quick FizzBuzz: print 1..n (default 100). Run: node scripts/fizzbuzz.js [n]
 */
const max = Math.max(1, parseInt(process.argv[2] || "100", 10) || 100);
for (let n = 1; n <= max; n++) {
  const f = n % 3 === 0;
  const b = n % 5 === 0;
  if (f && b) console.log("FizzBuzz");
  else if (f) console.log("Fizz");
  else if (b) console.log("Buzz");
  else console.log(String(n));
}
