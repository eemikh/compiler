use crate::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseErrorKind {
    UnexpectedCharacter(char),
    ExpectedCharacter { expected: char, got: char },
    IntegerOverflow,
    UnexpectedToken,
    UnexpectedEof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
}
