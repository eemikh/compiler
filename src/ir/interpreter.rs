use std::collections::HashMap;

use crate::ir::{BoolOperation, FunctionId, Instruction, IntOperation, LabelId, Module, Variable};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Value {
    Int(i64),
    Bool(bool),
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

pub fn interpret(module: &Module) {
    let mut ctx = Context {
        variables: Vec::new(),
        module,
    };

    ctx.new_scope();

    call_function(&mut ctx, module.entry);
}

fn call_function(ctx: &mut Context, function: FunctionId) -> Option<Value> {
    let f = ctx.module.get_function(function).unwrap();

    let mut pc = 0;
    while let Some(instruction) = f.instructions.get(pc) {
        let flow = execute_instruction(ctx, instruction);

        match flow {
            InstructionFlow::Return(Some(var)) => return Some(ctx.get_value(var).unwrap()),
            InstructionFlow::Return(None) => return None,
            InstructionFlow::Jump(label_id) => {
                pc = *f.labels.get(&label_id).unwrap();
                continue;
            }
            InstructionFlow::Continue => {}
        }

        pc += 1;
    }

    None
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
                ctx.set_value(*ret_var, value.unwrap());
            }

            InstructionFlow::Continue
        }
        Instruction::Copy { from, to } => {
            let value = ctx.get_value(*from).unwrap();
            ctx.set_value(*to, value);

            InstructionFlow::Continue
        }
        Instruction::Jump(label_id) => InstructionFlow::Jump(*label_id),
        Instruction::CondJump { cond_var, target } => match ctx.get_value(*cond_var) {
            Some(Value::Bool(true)) => InstructionFlow::Jump(*target),
            Some(Value::Bool(false)) => InstructionFlow::Continue,
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
                        IntOperation::LessThan => Value::Bool(lhs < rhs),
                        IntOperation::LessEqual => Value::Bool(lhs <= rhs),
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
        Instruction::Return(variable) => InstructionFlow::Return(*variable),
    }
}
