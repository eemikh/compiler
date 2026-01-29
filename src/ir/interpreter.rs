use std::collections::HashMap;

use crate::{
    Builtin,
    ir::{
        BoolOperation, Function, FunctionId, Instruction, IntOperation, InternalFunction, LabelId,
        Module, Variable,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Unit,
}

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
    builtins: &'a [Builtin<'a>],
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
    let mut ctx = Context {
        variables: Vec::new(),
        module,
        builtins,
    };

    ctx.new_scope();

    call_function(&mut ctx, module.entry)
}

fn call_function(ctx: &mut Context, function: FunctionId) -> Value {
    let f = ctx.module.get_function(function).unwrap();

    match f {
        Function::Internal(internal_function) => call_internal_function(ctx, internal_function),
        Function::External(name) => todo!(),
    }
}

fn call_internal_function(ctx: &mut Context, function: &InternalFunction) -> Value {
    let mut pc = 0;
    while let Some(instruction) = function.instructions.get(pc) {
        let flow = execute_instruction(ctx, instruction);

        match flow {
            InstructionFlow::Return(Some(var)) => return ctx.get_value(var).unwrap(),
            InstructionFlow::Return(None) => return Value::Unit,
            InstructionFlow::Jump(label_id) => {
                pc = *function.labels.get(&label_id).unwrap();
                continue;
            }
            InstructionFlow::Continue => {}
        }

        pc += 1;
    }

    Value::Unit
}

fn execute_instruction(ctx: &mut Context, instruction: &Instruction) -> InstructionFlow {
    match instruction {
        Instruction::Call {
            function,
            parameters,
            return_value,
        } => {
            let mut variables = HashMap::new();
            for (i, variable) in parameters.iter().enumerate() {
                variables.insert(
                    Variable(i.try_into().unwrap()),
                    ctx.get_value(*variable).unwrap(),
                );
            }

            ctx.new_scope_from_variables(variables);

            let value = call_function(ctx, *function);

            ctx.remove_scope();

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
        Instruction::LoadInt { target, value } => {
            ctx.set_value(*target, Value::Int(*value as i64));

            InstructionFlow::Continue
        }
        Instruction::LoadBool { target, value } => {
            ctx.set_value(*target, Value::Bool(*value));

            InstructionFlow::Continue
        }
        Instruction::LoadUnit(variable) => {
            ctx.set_value(*variable, Value::Unit);

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

    fn build_module(instructions: &[InstructionOrLabel]) -> Module {
        let mut builder = ModuleBuilder::new();
        let mut f = FunctionBuilder::new();

        for i in instructions {
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
        builder.set_entry(fid);

        builder.build()
    }

    fn test_wrapper(instructions: &[InstructionOrLabel]) -> Value {
        let module = build_module(instructions);
        let mut ctx = Context {
            variables: Vec::new(),
            module: &module,
            builtins: &[],
        };

        ctx.new_scope();

        call_function(&mut ctx, module.entry)
    }

    #[test]
    fn test_return() {
        assert_eq!(test_wrapper(&[I(Instruction::Return(None)),]), Value::Unit);

        assert_eq!(test_wrapper(&[]), Value::Unit);
    }

    #[test]
    fn test_load() {
        assert_eq!(
            test_wrapper(&[
                I(Instruction::LoadInt {
                    target: Variable(0),
                    value: 1234
                }),
                I(Instruction::Return(Some(Variable(0)))),
            ]),
            Value::Int(1234)
        );

        assert_eq!(
            test_wrapper(&[
                I(Instruction::LoadInt {
                    target: Variable(2),
                    value: 9
                }),
                I(Instruction::LoadBool {
                    target: Variable(2),
                    value: true
                }),
                I(Instruction::Return(Some(Variable(2)))),
            ]),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_int_op() {
        assert_eq!(
            test_wrapper(&[
                I(Instruction::LoadInt {
                    target: Variable(0),
                    value: 1234
                }),
                I(Instruction::LoadInt {
                    target: Variable(1),
                    value: 8765
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
                I(Instruction::LoadInt {
                    target: Variable(2),
                    value: 32
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
            ]),
            Value::Int(3)
        );

        assert_eq!(
            test_wrapper(&[
                I(Instruction::LoadInt {
                    target: Variable(0),
                    value: 1234
                }),
                I(Instruction::LoadInt {
                    target: Variable(1),
                    value: 1235
                }),
                I(Instruction::IntOp {
                    operation: IntOperation::LessThan,
                    target: Variable(0),
                    lhs: Variable(0),
                    rhs: Variable(1)
                }),
                I(Instruction::Return(Some(Variable(0)))),
            ]),
            Value::Bool(true)
        );

        assert_eq!(
            test_wrapper(&[
                I(Instruction::LoadInt {
                    target: Variable(0),
                    value: 1235
                }),
                I(Instruction::LoadInt {
                    target: Variable(1),
                    value: 1235
                }),
                I(Instruction::IntOp {
                    operation: IntOperation::LessThan,
                    target: Variable(0),
                    lhs: Variable(0),
                    rhs: Variable(1)
                }),
                I(Instruction::Return(Some(Variable(0)))),
            ]),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_jump() {
        assert_eq!(
            test_wrapper(&[
                I(Instruction::LoadInt {
                    target: Variable(0),
                    value: 1
                }),
                I(Instruction::Jump(LabelId(1))),
                I(Instruction::LoadInt {
                    target: Variable(0),
                    value: 2
                }),
                L, // 0
                L, // 1
                L, // 2
                I(Instruction::Return(Some(Variable(0)))),
            ]),
            Value::Int(1)
        );

        assert_eq!(
            test_wrapper(&[
                I(Instruction::LoadInt {
                    target: Variable(0),
                    value: 1
                }),
                I(Instruction::LoadBool {
                    target: Variable(1),
                    value: true
                }),
                I(Instruction::CondJump {
                    cond_var: Variable(1),
                    then: LabelId(1),
                    els: LabelId(0),
                }),
                L, // 0
                I(Instruction::LoadInt {
                    target: Variable(0),
                    value: 2
                }),
                L, // 1
                I(Instruction::Return(Some(Variable(0)))),
            ]),
            Value::Int(1)
        );

        assert_eq!(
            test_wrapper(&[
                I(Instruction::LoadInt {
                    target: Variable(0),
                    value: 1
                }),
                I(Instruction::LoadBool {
                    target: Variable(1),
                    value: false
                }),
                I(Instruction::CondJump {
                    cond_var: Variable(1),
                    then: LabelId(1),
                    els: LabelId(0),
                }),
                L, // 0
                I(Instruction::LoadInt {
                    target: Variable(0),
                    value: 2
                }),
                L, // 1
                I(Instruction::Return(Some(Variable(0)))),
            ]),
            Value::Int(2)
        );
    }
}
