use std::collections::HashMap;

use crate::{
    Builtin,
    ir::{
        BoolOperation, Function, FunctionId, Instruction, IntOperation, InternalFunction, LabelId,
        Module, Value, Variable,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum InstructionFlow {
    Return(Option<Variable>),
    Jump(LabelId),
    Continue,
}

#[derive(Debug, Clone)]
struct Context<'a> {
    variables: Vec<HashMap<Variable, Value>>,
    module: &'a Module,
    builtins: HashMap<FunctionId, fn(&[Value]) -> Value>,
}

impl Context<'_> {
    fn new_scope(&mut self) {
        self.variables.push(HashMap::new());
    }

    fn new_scope_from_variables(&mut self, variables: HashMap<Variable, Value>) {
        self.variables.push(variables);
    }

    fn remove_scope(&mut self) {
        self.variables.pop();
    }

    fn get_value(&self, variable: Variable) -> Option<Value> {
        self.variables
            .iter()
            .next_back()
            .expect("there is always at least one scope")
            .get(&variable)
            .copied()
    }

    fn set_value(&mut self, variable: Variable, value: Value) {
        self.variables
            .iter_mut()
            .next_back()
            .expect("there is always at least one scope")
            .insert(variable, value);
    }
}

pub fn interpret(module: &Module, builtins: &[Builtin]) -> Value {
    let mut builtin_map = HashMap::new();

    for (id, name) in
        module
            .functions
            .iter()
            .enumerate()
            .filter_map(|(id, function)| match function {
                Function::External(name) => Some((FunctionId(id.try_into().unwrap()), name)),
                _ => None,
            })
    {
        let function = builtins
            .iter()
            .find_map(|builtin| match builtin.name == name {
                true => Some(builtin.function),
                _ => None,
            })
            .unwrap();

        builtin_map.insert(id, function);
    }

    let mut ctx = Context {
        variables: Vec::new(),
        module,
        builtins: builtin_map,
    };

    ctx.new_scope();

    call_function(&mut ctx, module.entry, &[])
}

fn call_function(ctx: &mut Context, function: FunctionId, parameters: &[Variable]) -> Value {
    let f = ctx.module.get_function(function).unwrap();

    match f {
        Function::Internal(internal_function) => {
            call_internal_function(ctx, internal_function, parameters)
        }
        Function::External(_) => call_external_function(ctx, function, parameters),
    }
}

fn call_external_function(
    ctx: &mut Context,
    function: FunctionId,
    parameters: &[Variable],
) -> Value {
    let builtin = ctx.builtins.get(&function).unwrap();
    let values: Vec<Value> = parameters
        .iter()
        .map(|variable| ctx.get_value(*variable).unwrap())
        .collect();

    builtin(&values)
}

fn call_internal_function(
    ctx: &mut Context,
    function: &InternalFunction,
    parameters: &[Variable],
) -> Value {
    let mut variables = HashMap::new();
    for (i, variable) in parameters.iter().enumerate() {
        variables.insert(
            Variable(i.try_into().unwrap()),
            ctx.get_value(*variable).unwrap(),
        );
    }

    ctx.new_scope_from_variables(variables);

    let mut pc = 0;
    let mut ret = Value::Unit;
    while let Some(instruction) = function.instructions.get(pc) {
        let flow = execute_instruction(ctx, instruction);

        match flow {
            InstructionFlow::Return(Some(var)) => {
                ret = ctx.get_value(var).unwrap();
                break;
            }
            InstructionFlow::Return(None) => break,
            InstructionFlow::Jump(label_id) => {
                pc = *function.labels.get(&label_id).unwrap();
                continue;
            }
            InstructionFlow::Continue => {}
        }

        pc += 1;
    }

    ctx.remove_scope();

    ret
}

fn execute_instruction(ctx: &mut Context, instruction: &Instruction) -> InstructionFlow {
    match instruction {
        Instruction::Call {
            function,
            parameters,
            return_value,
        } => {
            let fid = ctx.get_value(*function).unwrap();

            let fid = match fid {
                Value::Function(fid) => fid,
                _ => panic!("tried to call non-function variable"),
            };

            let value = call_function(ctx, fid, parameters);

            if let Some(ret_var) = return_value {
                ctx.set_value(*ret_var, value);
            }

            InstructionFlow::Continue
        }
        Instruction::Copy { from, to } => {
            let value = ctx.get_value(*from).unwrap();
            ctx.set_value(*to, value);

            InstructionFlow::Continue
        }
        Instruction::Jump(label_id) => InstructionFlow::Jump(*label_id),
        Instruction::CondJump {
            cond_var,
            then,
            els,
        } => match ctx.get_value(*cond_var) {
            Some(Value::Bool(true)) => InstructionFlow::Jump(*then),
            Some(Value::Bool(false)) => InstructionFlow::Jump(*els),
            Some(_) => panic!("variable {:?} has wrong type", cond_var),
            None => panic!("variable {:?} has no value", cond_var),
        },
        Instruction::IntOp {
            operation,
            target,
            lhs,
            rhs,
        } => {
            let lhs = ctx.get_value(*lhs).unwrap();
            let rhs = ctx.get_value(*rhs).unwrap();

            match (lhs, rhs) {
                (Value::Int(lhs), Value::Int(rhs)) => {
                    let res = match operation {
                        IntOperation::Add => Value::Int(lhs.wrapping_add(rhs)),
                        IntOperation::Subtract => Value::Int(lhs.wrapping_sub(rhs)),
                        IntOperation::Multiply => Value::Int(lhs.wrapping_mul(rhs)),
                        IntOperation::Divide => Value::Int(lhs / rhs),
                        IntOperation::Modulo => Value::Int(lhs % rhs),
                        IntOperation::Equal => Value::Bool(lhs == rhs),
                        IntOperation::NotEqual => Value::Bool(lhs != rhs),
                        IntOperation::LessThan => Value::Bool(lhs < rhs),
                        IntOperation::LessEqual => Value::Bool(lhs <= rhs),
                        IntOperation::GreaterThan => Value::Bool(lhs > rhs),
                        IntOperation::GreaterEqual => Value::Bool(lhs >= rhs),
                    };

                    ctx.set_value(*target, res);
                }
                _ => panic!("variables have wrong types"),
            }

            InstructionFlow::Continue
        }
        Instruction::BoolOp {
            operation,
            target,
            lhs,
            rhs,
        } => {
            let lhs = ctx.get_value(*lhs).unwrap();
            let rhs = ctx.get_value(*rhs).unwrap();

            match (lhs, rhs) {
                (Value::Bool(lhs), Value::Bool(rhs)) => {
                    let res = match operation {
                        BoolOperation::Or => lhs || rhs,
                        BoolOperation::And => lhs && rhs,
                        BoolOperation::Xor => lhs ^ rhs,
                        BoolOperation::Equal => lhs == rhs,
                        BoolOperation::NotEqual => lhs != rhs,
                    };

                    ctx.set_value(*target, Value::Bool(res));
                }
                _ => panic!("variables have wrong types"),
            }

            InstructionFlow::Continue
        }
        Instruction::Load { target, value } => {
            ctx.set_value(*target, *value);

            InstructionFlow::Continue
        }
        Instruction::Return(variable) => InstructionFlow::Return(*variable),
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::builder::{FunctionBuilder, ModuleBuilder};

    use super::*;

    enum InstructionOrLabel {
        L,
        I(Instruction),
    }

    use InstructionOrLabel::*;

    fn build_module(instructions: &[&[InstructionOrLabel]]) -> Module {
        let mut builder = ModuleBuilder::new();

        for instructions in instructions {
            let mut f = FunctionBuilder::new();

            for i in *instructions {
                match i {
                    InstructionOrLabel::L => {
                        let label = f.label();
                        f.emit_label(label);
                    }
                    InstructionOrLabel::I(instruction) => f.emit_instruction(instruction.clone()),
                }
            }

            let fid = builder.function();
            builder.add_function(fid, Function::Internal(f.build()));
        }

        builder.set_entry(FunctionId(0));

        builder.build()
    }

    fn test_wrapper(instructions: &[&[InstructionOrLabel]]) -> Value {
        let module = build_module(instructions);
        let mut ctx = Context {
            variables: Vec::new(),
            module: &module,
            builtins: HashMap::new(),
        };

        ctx.new_scope();

        call_function(&mut ctx, module.entry, &[])
    }

    #[test]
    fn test_return() {
        assert_eq!(
            test_wrapper(&[&[I(Instruction::Return(None)),]]),
            Value::Unit
        );

        assert_eq!(test_wrapper(&[&[]]), Value::Unit);
    }

    #[test]
    fn test_load() {
        assert_eq!(
            test_wrapper(&[&[
                I(Instruction::Load {
                    target: Variable(0),
                    value: Value::Int(1234)
                }),
                I(Instruction::Return(Some(Variable(0)))),
            ]]),
            Value::Int(1234)
        );

        assert_eq!(
            test_wrapper(&[&[
                I(Instruction::Load {
                    target: Variable(2),
                    value: Value::Int(9)
                }),
                I(Instruction::Load {
                    target: Variable(2),
                    value: Value::Bool(true)
                }),
                I(Instruction::Return(Some(Variable(2)))),
            ]]),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_int_op() {
        assert_eq!(
            test_wrapper(&[&[
                I(Instruction::Load {
                    target: Variable(0),
                    value: Value::Int(1234)
                }),
                I(Instruction::Load {
                    target: Variable(1),
                    value: Value::Int(8765)
                }),
                I(Instruction::IntOp {
                    operation: IntOperation::Add,
                    target: Variable(0),
                    lhs: Variable(0),
                    rhs: Variable(1)
                }),
                I(Instruction::IntOp {
                    operation: IntOperation::Multiply,
                    target: Variable(0),
                    lhs: Variable(0),
                    rhs: Variable(1)
                }),
                I(Instruction::Load {
                    target: Variable(2),
                    value: Value::Int(32)
                }),
                I(Instruction::IntOp {
                    operation: IntOperation::Subtract,
                    target: Variable(0),
                    lhs: Variable(0),
                    rhs: Variable(2)
                }),
                I(Instruction::IntOp {
                    operation: IntOperation::Divide,
                    target: Variable(0),
                    lhs: Variable(0),
                    rhs: Variable(2)
                }),
                I(Instruction::IntOp {
                    operation: IntOperation::Modulo,
                    target: Variable(0),
                    lhs: Variable(0),
                    rhs: Variable(2)
                }),
                I(Instruction::Return(Some(Variable(0)))),
            ]]),
            Value::Int(3)
        );

        assert_eq!(
            test_wrapper(&[&[
                I(Instruction::Load {
                    target: Variable(0),
                    value: Value::Int(1234)
                }),
                I(Instruction::Load {
                    target: Variable(1),
                    value: Value::Int(1235)
                }),
                I(Instruction::IntOp {
                    operation: IntOperation::LessThan,
                    target: Variable(0),
                    lhs: Variable(0),
                    rhs: Variable(1)
                }),
                I(Instruction::Return(Some(Variable(0)))),
            ]]),
            Value::Bool(true)
        );

        assert_eq!(
            test_wrapper(&[&[
                I(Instruction::Load {
                    target: Variable(0),
                    value: Value::Int(1235)
                }),
                I(Instruction::Load {
                    target: Variable(1),
                    value: Value::Int(1235)
                }),
                I(Instruction::IntOp {
                    operation: IntOperation::LessThan,
                    target: Variable(0),
                    lhs: Variable(0),
                    rhs: Variable(1)
                }),
                I(Instruction::Return(Some(Variable(0)))),
            ]]),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_jump() {
        assert_eq!(
            test_wrapper(&[&[
                I(Instruction::Load {
                    target: Variable(0),
                    value: Value::Int(1)
                }),
                I(Instruction::Jump(LabelId(1))),
                I(Instruction::Load {
                    target: Variable(0),
                    value: Value::Int(2)
                }),
                L, // 0
                L, // 1
                L, // 2
                I(Instruction::Return(Some(Variable(0)))),
            ]]),
            Value::Int(1)
        );

        assert_eq!(
            test_wrapper(&[&[
                I(Instruction::Load {
                    target: Variable(0),
                    value: Value::Int(1)
                }),
                I(Instruction::Load {
                    target: Variable(1),
                    value: Value::Bool(true)
                }),
                I(Instruction::CondJump {
                    cond_var: Variable(1),
                    then: LabelId(1),
                    els: LabelId(0),
                }),
                L, // 0
                I(Instruction::Load {
                    target: Variable(0),
                    value: Value::Int(2)
                }),
                L, // 1
                I(Instruction::Return(Some(Variable(0)))),
            ]]),
            Value::Int(1)
        );

        assert_eq!(
            test_wrapper(&[&[
                I(Instruction::Load {
                    target: Variable(0),
                    value: Value::Int(1)
                }),
                I(Instruction::Load {
                    target: Variable(1),
                    value: Value::Bool(false)
                }),
                I(Instruction::CondJump {
                    cond_var: Variable(1),
                    then: LabelId(1),
                    els: LabelId(0),
                }),
                L, // 0
                I(Instruction::Load {
                    target: Variable(0),
                    value: Value::Int(2)
                }),
                L, // 1
                I(Instruction::Return(Some(Variable(0)))),
            ]]),
            Value::Int(2)
        );
    }

    #[test]
    fn test_fun() {
        assert_eq!(
            test_wrapper(&[
                &[
                    I(Instruction::Load {
                        target: Variable(0),
                        value: Value::Function(FunctionId(1))
                    }),
                    I(Instruction::Call {
                        function: Variable(0),
                        parameters: vec![],
                        return_value: Some(Variable(0)),
                    }),
                    I(Instruction::Return(Some(Variable(0)))),
                ],
                &[]
            ]),
            Value::Unit
        );

        assert_eq!(
            test_wrapper(&[
                &[
                    I(Instruction::Load {
                        target: Variable(0),
                        value: Value::Function(FunctionId(1))
                    }),
                    I(Instruction::Call {
                        function: Variable(0),
                        parameters: vec![],
                        return_value: Some(Variable(7)),
                    }),
                    I(Instruction::Return(Some(Variable(7)))),
                ],
                &[
                    I(Instruction::Load {
                        target: Variable(0),
                        value: Value::Int(1234),
                    }),
                    I(Instruction::Return(Some(Variable(0)))),
                ]
            ]),
            Value::Int(1234)
        );

        assert_eq!(
            test_wrapper(&[
                &[
                    I(Instruction::Load {
                        target: Variable(0),
                        value: Value::Function(FunctionId(1))
                    }),
                    I(Instruction::Load {
                        target: Variable(1),
                        value: Value::Int(832)
                    }),
                    I(Instruction::Load {
                        target: Variable(2),
                        value: Value::Int(22)
                    }),
                    I(Instruction::Call {
                        function: Variable(0),
                        parameters: vec![Variable(1), Variable(2)],
                        return_value: Some(Variable(7)),
                    }),
                    I(Instruction::Return(Some(Variable(7)))),
                ],
                &[
                    I(Instruction::IntOp {
                        operation: IntOperation::Add,
                        target: Variable(2),
                        lhs: Variable(0),
                        rhs: Variable(1)
                    }),
                    I(Instruction::Return(Some(Variable(2)))),
                ]
            ]),
            Value::Int(854)
        );
    }
}
