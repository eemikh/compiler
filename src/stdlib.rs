use std::io;

use crate::{Builtin, ir::interpreter::Value, types::Typ};

pub static BUILTINS: &[Builtin] = &[
    Builtin {
        name: "print_int",
        params: &[Typ::Int],
        ret: Typ::Unit,
        function: print_value,
    },
    Builtin {
        name: "print_bool",
        params: &[Typ::Bool],
        ret: Typ::Unit,
        function: print_value,
    },
    Builtin {
        name: "read_int",
        params: &[],
        ret: Typ::Int,
        function: print_value,
    },
];

fn print_value(params: &[Value]) -> Value {
    match params[0] {
        Value::Int(value) => println!("{}", value),
        Value::Bool(value) => println!("{}", value),
        _ => panic!("print_value got invalid value {:?}", params),
    }

    Value::Unit
}

fn read_int(params: &[Value]) -> Value {
    let mut line = String::new();
    io::stdin().read_line(&mut line).unwrap();

    Value::Int(line.trim_end().parse().unwrap())
}
