use crate::{Span, syntax::token::TokenKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseErrorKind {
    UnexpectedCharacter(char),
    ExpectedCharacter { expected: char, got: char },
    IntegerOverflow,
    ExpectedToken(TokenKind<'static>),
    ExpectedTokens(&'static [&'static str]),
    ExpectedIdentifier,
    UnexpectedEof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
}
