#![feature(assert_matches)]
use std::env::args;

mod token;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
}

fn main() {
    let mut args = args();
    println!("{}", args.nth(1).unwrap());
}
