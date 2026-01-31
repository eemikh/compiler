use std::{env::args, fs::File, io::Read};

use crate::{
    codegen::{gen_ir, gen_module},
    ir::Value,
    syntax::parse,
    types::{Typ, typecheck},
};

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
    let mut f = File::open(args.nth(1).unwrap()).unwrap();
    let mut code = String::new();
    f.read_to_string(&mut code).unwrap();

    let ast = parse(&code).0.unwrap();
    let typmap = typecheck(&ast, stdlib::BUILTINS).unwrap();
    let ir = gen_ir(&ast, &typmap, stdlib::BUILTINS);
    let mut res = String::new();
    gen_module(&ir, &mut res);
    print!("{}", res);
}
