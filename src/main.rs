#![feature(assert_matches)]
use std::env::args;

use crate::{ir::Value, types::Typ};

mod codegen;
mod ir;
mod scope;
mod stdlib;
mod syntax;
mod types;

// TODO: place somewhere appropriate
#[derive(Debug, Clone)]
pub struct Builtin<'a> {
    pub name: &'a str,
    pub params: &'a [Typ],
    pub ret: Typ,
    pub function: fn(&[Value]) -> Value,
}

fn main() {
    let mut args = args();
    println!("{}", args.nth(1).unwrap());
}
