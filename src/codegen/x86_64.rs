use std::fmt::Write;

use crate::ir::{BoolOperation, Instruction, IntOperation, LabelId, Module, Value, Variable};

struct Context<'a, W: Write> {
    module: &'a Module,
    writer: W,
    indent: usize,
}

macro_rules! emit {
    ($expression:expr) => {{
        ::std::write!($expression.writer, "\n").unwrap();
    }};
    ($expression:expr, $($arg:tt)*) => {{
        ::std::write!($expression.writer, "{:1$}", " ", $expression.indent).unwrap();
        ::std::write!($expression.writer, $($arg)*).unwrap();
        ::std::write!($expression.writer, "\n").unwrap();
    }};

}

// stack frame in function:
// rbp => previous rbp
// rbp - 8 => variable 0
// rbp - 16 => variable 1
// ...

fn rbp_offset(variable: Variable) -> i32 {
    -8 * i32::try_from(variable.0 + 1).unwrap()
}

fn gen_instruction<W: Write>(ctx: &mut Context<W>, instruction: &Instruction) {
    match instruction {
        Instruction::Call {
            function,
            parameters,
            return_value,
        } => gen_call(ctx, *function, parameters, *return_value),
        Instruction::Copy { from, to } => gen_copy(ctx, *from, *to),
        Instruction::Jump(label_id) => gen_jump(ctx, *label_id),
        Instruction::CondJump {
            cond_var,
            then,
            els,
        } => gen_cond_jump(ctx, *cond_var, *then, *els),
        Instruction::IntOp {
            operation,
            target,
            lhs,
            rhs,
        } => gen_int_op(ctx, *operation, *target, *lhs, *rhs),
        Instruction::BoolOp {
            operation,
            target,
            lhs,
            rhs,
        } => gen_bool_op(ctx, *operation, *target, *lhs, *rhs),
        Instruction::Load { target, value } => gen_load(ctx, *target, value),
        Instruction::Return(variable) => gen_return(ctx, *variable),
    }
}

fn gen_call<W: Write>(
    ctx: &mut Context<W>,
    function: Variable,
    parameters: &[Variable],
    return_variable: Option<Variable>,
) {
    emit!(ctx, "mov {}(%rbp), %rax", rbp_offset(function));

    for (parameter, register) in parameters
        .iter()
        .zip(&["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"])
    {
        emit!(ctx, "mov {}(%rbp), {}", rbp_offset(*parameter), register);
    }

    emit!(ctx, "call *%rax");

    if let Some(return_variable) = return_variable {
        emit!(ctx, "mov %rax, {}(%rbp)", rbp_offset(return_variable));
    }
}

fn gen_cond_jump<W: Write>(ctx: &mut Context<W>, cond_var: Variable, then: LabelId, els: LabelId) {
    emit!(ctx, "cmpb $1, {}(%rbp)", rbp_offset(cond_var));
    emit!(ctx, "je .L{}", then.0);
    emit!(ctx, "jmp .L{}", els.0);
}

fn gen_int_op<W: Write>(
    ctx: &mut Context<W>,
    operation: IntOperation,
    target: Variable,
    lhs: Variable,
    rhs: Variable,
) {
    emit!(ctx, "movq {}(%rbp), %rax", rbp_offset(lhs));
    emit!(ctx, "movq {}(%rbp), %rcx", rbp_offset(rhs));

    let mut res_reg = "%rax";

    match operation {
        IntOperation::Add => emit!(ctx, "addq %rcx, %rax"),
        IntOperation::Subtract => emit!(ctx, "subq %rcx, %rax"),
        IntOperation::Multiply => emit!(ctx, "imulq %rcx, %rax"),
        IntOperation::Divide => {
            emit!(ctx, "cqto");
            emit!(ctx, "idivq %rcx");
        }
        IntOperation::Modulo => {
            emit!(ctx, "cqto");
            emit!(ctx, "idivq %rcx");

            res_reg = "%rdx";
        }
        IntOperation::Equal => {
            emit!(ctx, "cmpq %rcx, %rax");
            emit!(ctx, "sete %dl");

            res_reg = "%rdx";
        }
        IntOperation::NotEqual => {
            emit!(ctx, "cmpq %rcx, %rax");
            emit!(ctx, "setne %dl");

            res_reg = "%rdx";
        }
        IntOperation::LessThan => {
            emit!(ctx, "cmpq %rcx, %rax");
            emit!(ctx, "setl %dl");

            res_reg = "%rdx";
        }
        IntOperation::LessEqual => {
            emit!(ctx, "cmpq %rcx, %rax");
            emit!(ctx, "setle %dl");

            res_reg = "%rdx";
        }
        IntOperation::GreaterThan => {
            emit!(ctx, "cmpq %rcx, %rax");
            emit!(ctx, "setg %dl");

            res_reg = "%rdx";
        }
        IntOperation::GreaterEqual => {
            emit!(ctx, "cmpq %rcx, %rax");
            emit!(ctx, "setge %dl");

            res_reg = "%rdx";
        }
    }

    emit!(ctx, "movq {}, {}(%rbp)", res_reg, rbp_offset(target));
}

fn gen_bool_op<W: Write>(
    ctx: &mut Context<W>,
    operation: BoolOperation,
    target: Variable,
    lhs: Variable,
    rhs: Variable,
) {
    emit!(ctx, "movq {}(%rbp), %rax", rbp_offset(lhs));
    emit!(ctx, "movq {}(%rbp), %rcx", rbp_offset(rhs));

    match operation {
        BoolOperation::Or => emit!(ctx, "orq %rcx, %rax"),
        BoolOperation::And => emit!(ctx, "andq %rcx, %rax"),
        BoolOperation::Xor => emit!(ctx, "xorq %rcx, %rax"),
        BoolOperation::Equal => emit!(ctx, "andq %rcx, %rax"),
        BoolOperation::NotEqual => {
            emit!(ctx, "cmpq %rcx, %rax");
            emit!(ctx, "setne %al");
        }
    }

    emit!(ctx, "movq %rax, {}(%rbp)", rbp_offset(target));
}

fn gen_jump<W: Write>(ctx: &mut Context<W>, label_id: LabelId) {
    emit!(ctx, "jmp .L{}", label_id.0);
}

fn gen_copy<W: Write>(ctx: &mut Context<W>, from: Variable, to: Variable) {
    emit!(ctx, "movq {}(%rbp), %rax", rbp_offset(from));
    emit!(ctx, "movq %rax, {}(%rbp)", rbp_offset(to));
}

fn gen_load<W: Write>(ctx: &mut Context<W>, target: Variable, value: &Value) {
    match value {
        Value::Int(value) => {
            let value = *value as u64;

            if value <= u32::MAX.into() {
                emit!(ctx, "movq ${}, {}(%rbp)", value, rbp_offset(target));
            } else {
                emit!(ctx, "movabsq ${}, %rax", value);
                emit!(ctx, "movq %rax, {}(%rbp)", rbp_offset(target));
            }
        }
        Value::Bool(value) => {
            let value = match *value {
                true => 1,
                false => 0,
            };

            emit!(ctx, "movq ${}, {}(%rbp)", value, rbp_offset(target));
        }
        Value::Function(function_id) => {
            emit!(
                ctx,
                "leaq {}(%rip), %rax",
                ctx.module.functions[usize::try_from(function_id.0).unwrap()].name
            );
            emit!(ctx, "movq %rax, {}(%rbp)", rbp_offset(target));
            emit!(ctx, "callq *%rax");
        }
        Value::Unit => {}
    }
}

fn gen_return<W: Write>(ctx: &mut Context<W>, variable: Option<Variable>) {
    if let Some(variable) = variable {
        emit!(ctx, "movq {}(%rbp), %rax", rbp_offset(variable));
    }

    emit!(ctx, "ret");
}
