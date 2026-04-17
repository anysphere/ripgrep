#!/usr/bin/env ruby
# Quick FizzBuzz: print 1..n (default 100). Run: ruby scripts/fizzbuzz.rb [n]

max = (ARGV[0] || 100).to_i
max = 1 if max < 1

(1..max).each do |n|
  puts(
    if (n % 15).zero? then "FizzBuzz"
    elsif (n % 3).zero? then "Fizz"
    elsif (n % 5).zero? then "Buzz"
    else n.to_s
    end
  )
end
