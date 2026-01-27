#![feature(assert_matches)]
use std::env::args;

mod ir;
mod scope;
mod syntax;
mod types;

fn main() {
    let mut args = args();
    println!("{}", args.nth(1).unwrap());
}
