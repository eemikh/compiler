use crate::Span;

#[derive(Debug, Clone, Copy)]
pub enum ParseErrorKind {
    UnexpectedCharacter(char),
    ExpectedCharacter { expected: char, got: char },
    IntegerOverflow,
    UnexpectedEof,
}

#[derive(Debug, Clone, Copy)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
}
