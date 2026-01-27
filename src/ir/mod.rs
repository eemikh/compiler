use std::collections::HashMap;

mod builder;
mod interpreter;

pub use builder::{FunctionBuilder, ModuleBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LabelId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(u32);

/// A reference to a variable. The reference is only valid for the function it was created in; using
/// it in any other function may cause undesired behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variable(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoolOperation {
    Or,
    And,
    Xor,
    Equal,
    NotEqual,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Instruction {
    /// Calls the function with the given parameters. The parameters are copied to variables 0, 1,
    /// ... in the scope of the target function.
    Call {
        function: FunctionId,
        parameters: Vec<Variable>,
        return_value: Option<Variable>,
    },
    Copy {
        from: Variable,
        to: Variable,
    },
    Jump(LabelId),
    CondJump {
        cond_var: Variable,
        target: LabelId,
    },
    IntOp {
        operation: IntOperation,
        target: Variable,
        lhs: Variable,
        rhs: Variable,
    },
    BoolOp {
        operation: BoolOperation,
        target: Variable,
        lhs: Variable,
        rhs: Variable,
    },
    LoadInt {
        target: Variable,
        value: u64,
    },
    LoadBool {
        target: Variable,
        value: bool,
    },
    Return(Option<Variable>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub instructions: Vec<Instruction>,
    /// Maps a label ID to index in instructions
    pub labels: HashMap<LabelId, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub functions: Vec<Function>,
    pub entry: FunctionId,
}

impl Module {
    pub fn get_function(&self, function: FunctionId) -> Option<&Function> {
        self.functions.get(usize::try_from(function.0).ok()?)
    }
}
