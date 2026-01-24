use std::fmt::Display;

use crate::syntax::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub body: BlockExpression,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    Binary(BinaryExpression),
    Unary(UnaryExpression),
    Primary(Primary),
    If(IfExpression),
    While(WhileExpression),
    Call(CallExpression),
    Block(BlockExpression),
    Var(VarExpression),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryExpression {
    pub operator: BinaryOperator,
    pub lhs: Box<Node<Expression>>,
    pub rhs: Box<Node<Expression>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarExpression {
    pub name: Identifier,
    pub typ: Option<Identifier>,
    pub value: Box<Node<Expression>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockExpression {
    pub expressions: Vec<Node<Expression>>,
    pub result_expression: Option<Box<Node<Expression>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallExpression {
    pub function: Box<Node<Expression>>,
    pub args: Vec<Node<Expression>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfExpression {
    pub condition: Box<Node<Expression>>,
    pub then: Box<Node<Expression>>,
    pub els: Option<Box<Node<Expression>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhileExpression {
    pub condition: Box<Node<Expression>>,
    pub body: Box<Node<Expression>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnaryExpression {
    pub operator: UnaryOperator,
    pub operand: Box<Node<Expression>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Not,
    Negate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Or,
    And,
    Equals,
    EqualEqual,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Primary {
    Bool(bool),
    Integer(u64),
    Identifier(Identifier),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Identifier(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node<T> {
    pub item: T,
    pub span: Span,
}

impl<T: Display> Display for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.item)
    }
}

impl Display for Primary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Primary::Bool(b) => write!(f, "{}", b),
            Primary::Integer(i) => write!(f, "{}", i),
            Primary::Identifier(identifier) => write!(f, "{}", identifier.0),
        }
    }
}

impl Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.body)
    }
}

impl Display for BinaryExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {} {})", self.operator, self.lhs, self.rhs)
    }
}

impl Display for IfExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.els {
            Some(els) => write!(f, "(if {} {} {})", self.condition, self.then, els),
            None => write!(f, "(if {} {})", self.condition, self.then),
        }
    }
}

impl Display for WhileExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(while {} {})", self.condition, self.body)
    }
}

impl Display for UnaryExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {})", self.operator, self.operand)
    }
}

impl Display for VarExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.typ {
            Some(typ) => write!(f, "(var {} {} {})", self.name, typ, self.value),
            None => write!(f, "(var {} {})", self.name, self.value),
        }
    }
}

impl Display for CallExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(call {}", self.function)?;

        for arg in &self.args {
            write!(f, " {}", arg)?;
        }

        write!(f, ")")
    }
}

impl Display for BlockExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(block")?;

        for expr in &self.expressions {
            write!(f, " {}", expr)?;
        }

        match &self.result_expression {
            Some(expr) => write!(f, " {})", expr),
            None => write!(f, " ())"),
        }
    }
}

impl Display for UnaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                UnaryOperator::Not => "not",
                UnaryOperator::Negate => "-",
            }
        )
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Binary(binary_expression) => write!(f, "{}", binary_expression),
            Expression::Primary(primary) => write!(f, "{}", primary),
            Expression::Unary(unary_expression) => write!(f, "{}", unary_expression),
            Expression::If(if_expression) => write!(f, "{}", if_expression),
            Expression::While(while_expression) => write!(f, "{}", while_expression),
            Expression::Call(call_expression) => write!(f, "{}", call_expression),
            Expression::Block(block_expression) => write!(f, "{}", block_expression),
            Expression::Var(var_expression) => write!(f, "{}", var_expression),
        }
    }
}

impl Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                BinaryOperator::Or => "or",
                BinaryOperator::And => "and",
                BinaryOperator::Equals => "=",
                BinaryOperator::EqualEqual => "==",
                BinaryOperator::NotEqual => "!=",
                BinaryOperator::LessThan => "<",
                BinaryOperator::LessEqual => "<=",
                BinaryOperator::GreaterThan => ">",
                BinaryOperator::GreaterEqual => ">=",
                BinaryOperator::Add => "+",
                BinaryOperator::Subtract => "-",
                BinaryOperator::Multiply => "*",
                BinaryOperator::Divide => "/",
                BinaryOperator::Modulo => "%",
            }
        )
    }
}
