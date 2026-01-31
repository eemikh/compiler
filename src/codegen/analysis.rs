use crate::ir::{Instruction, InternalFunction, Variable};

pub fn function_max_variable(function: &InternalFunction) -> Variable {
    let mut res = Variable(0);

    for instruction in &function.instructions {
        match instruction {
            Instruction::Call {
                function: _,
                parameters,
                return_value,
            } => {
                res = *[
                    res,
                    return_value.unwrap_or(Variable(0)),
                    *parameters.iter().max().unwrap_or(&Variable(0)),
                ]
                .iter()
                .max()
                .unwrap()
            }
            Instruction::Copy { from, to } => res = *[res, *from, *to].iter().max().unwrap(),
            Instruction::Jump(_) => {}
            Instruction::CondJump {
                cond_var,
                then: _,
                els: _,
            } => res = res.max(*cond_var),
            Instruction::IntOp {
                operation: _,
                target,
                lhs,
                rhs,
            } => res = *[res, *target, *lhs, *rhs].iter().max().unwrap(),
            Instruction::BoolOp {
                operation: _,
                target,
                lhs,
                rhs,
            } => res = *[res, *target, *lhs, *rhs].iter().max().unwrap(),
            Instruction::Load { target, value: _ } => res = res.max(*target),
            Instruction::Return(variable) => res = res.max(variable.unwrap_or(Variable(0))),
            Instruction::Label(_) => {}
        }
    }

    res
}
