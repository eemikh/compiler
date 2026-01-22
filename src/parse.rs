use std::iter::Peekable;

use crate::{
    Span,
    error::ParseError,
    token::{Token, TokenKind},
};

#[derive(Debug, Clone)]
pub enum Expression {
    Binary(BinaryExpression),
}

#[derive(Debug, Clone)]
pub struct BinaryExpression {
    operator: BinaryOperator,
    lhs: Box<Expression>,
    rhs: Box<Expression>,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOperator {
    Add,
    Subtract,
}

#[derive(Debug, Clone, Copy)]
pub enum Literal {
    Bool(bool),
    Integer(u64),
}

#[derive(Debug, Clone)]
pub struct Node<T> {
    item: T,
    span: Span,
}

struct Parser<'code, It: Iterator<Item = Result<Token<'code>, ParseError>>> {
    tokens: Peekable<It>,
}

impl<'code, It: Iterator<Item = Result<Token<'code>, ParseError>>> Parser<'code, It> {
    fn parse_literal(&mut self) -> Result<Node<Literal>, ParseError> {
        let token = self.next()?;

        let item = match token.kind {
            TokenKind::Identifier("true") => Literal::Bool(true),
            TokenKind::Identifier("false") => Literal::Bool(false),
            TokenKind::Integer(int) => Literal::Integer(int),
            _ => todo!("invalid"),
        };

        Ok(Node {
            item,
            span: token.span,
        })
    }

    fn peek(&mut self) -> Result<&Token<'code>, &ParseError> {
        // FIXME: unwrap
        self.tokens.peek().unwrap().as_ref()
    }

    fn next(&mut self) -> Result<Token<'code>, ParseError> {
        // FIXME: unwrap
        self.tokens.next().unwrap()
    }
}

fn parse<'code>(tokens: impl Iterator<Item = Result<Token<'code>, ParseError>>) {
    let tokens = tokens.peekable();

    for token in tokens {
        dbg!(token);
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use crate::{Span, token::TokenKind};

    use super::*;

    /// A wrapper for creating a [`Parser`] out of statically allocated tokens
    fn quick_parser(
        tokens: &'static [Result<Token<'static>, ParseError>],
    ) -> Parser<'static, impl Iterator<Item = Result<Token<'static>, ParseError>>> {
        Parser {
            tokens: tokens.iter().cloned().peekable(),
        }
    }

    #[test]
    fn test_literal() {
        assert_matches!(
            quick_parser(&[Ok(Token {
                kind: TokenKind::Identifier("true"),
                span: Span { start: 0, end: 4 },
            })])
            .parse_literal(),
            Ok(Node {
                item: Literal::Bool(true),
                span: Span { start: 0, end: 4 },
            })
        );

        assert_matches!(
            quick_parser(&[Ok(Token {
                kind: TokenKind::Identifier("false"),
                span: Span { start: 0, end: 5 },
            })])
            .parse_literal(),
            Ok(Node {
                item: Literal::Bool(false),
                span: Span { start: 0, end: 5 },
            })
        );

        assert_matches!(
            quick_parser(&[Ok(Token {
                kind: TokenKind::Integer(1234),
                span: Span { start: 0, end: 4 },
            })])
            .parse_literal(),
            Ok(Node {
                item: Literal::Integer(1234),
                span: Span { start: 0, end: 4 },
            })
        );
    }
}
