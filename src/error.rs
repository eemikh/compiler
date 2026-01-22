use crate::Span;

#[derive(Debug, Clone)]
pub enum ParseErrorKind {
    UnexpectedCharacter(char),
    ExpectedCharacter { expected: char, got: char },
    IntegerOverflow,
    UnexpectedEof,
}

#[derive(Debug, Clone)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
}
