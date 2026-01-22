use std::iter::Peekable;

use crate::{
    Span,
    error::ParseError,
    token::{Token, TokenKind},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    Binary(BinaryExpression),
    Primary(Primary),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryExpression {
    operator: BinaryOperator,
    lhs: Box<Node<Expression>>,
    rhs: Box<Node<Expression>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Or,
    And,
    Equals,
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

static BINARY_OP_TOKENS: &[&[TokenKind<'static>]] = &[
    &[TokenKind::Identifier("or")],
    &[TokenKind::Identifier("and")],
    &[TokenKind::EqualEqual, TokenKind::NotEqual],
    &[
        TokenKind::LessThan,
        TokenKind::LessEqual,
        TokenKind::GreaterThan,
        TokenKind::GreaterEqual,
    ],
    &[TokenKind::Plus, TokenKind::Minus],
    &[TokenKind::Asterisk, TokenKind::Slash, TokenKind::Percent],
];

impl TryFrom<TokenKind<'_>> for BinaryOperator {
    type Error = ();

    fn try_from(value: TokenKind<'_>) -> Result<Self, Self::Error> {
        match value {
            TokenKind::Identifier("or") => Ok(BinaryOperator::Or),
            TokenKind::Identifier("and") => Ok(BinaryOperator::And),
            TokenKind::Plus => Ok(BinaryOperator::Add),
            TokenKind::Minus => Ok(BinaryOperator::Subtract),
            TokenKind::Asterisk => Ok(BinaryOperator::Multiply),
            TokenKind::Slash => Ok(BinaryOperator::Divide),
            TokenKind::Percent => Ok(BinaryOperator::Modulo),
            TokenKind::EqualEqual => Ok(BinaryOperator::Equals),
            TokenKind::NotEqual => Ok(BinaryOperator::NotEqual),
            TokenKind::LessThan => Ok(BinaryOperator::LessThan),
            TokenKind::LessEqual => Ok(BinaryOperator::LessEqual),
            TokenKind::GreaterThan => Ok(BinaryOperator::GreaterThan),
            TokenKind::GreaterEqual => Ok(BinaryOperator::GreaterEqual),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Primary {
    Bool(bool),
    Integer(u64),
    Identifier(Identifier),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identifier(String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node<T> {
    item: T,
    span: Span,
}

struct Parser<'code, It: Iterator<Item = Result<Token<'code>, ParseError>>> {
    tokens: Peekable<It>,
}

impl<'code, It: Iterator<Item = Result<Token<'code>, ParseError>>> Parser<'code, It> {
    fn parse_expression(&mut self, level: usize) -> Result<Node<Expression>, ParseError> {
        assert!(level <= BINARY_OP_TOKENS.len());

        if level == BINARY_OP_TOKENS.len() {
            let primary = self.parse_primary()?;

            return Ok(Node {
                item: Expression::Primary(primary.item),
                span: primary.span,
            });
        }

        let ops = BINARY_OP_TOKENS[level];
        let mut lhs = self.parse_expression(level + 1)?;

        loop {
            // FIXME: handle tokenization errors
            let token = self.peek().unwrap();

            if ops.contains(&token.kind) {
                let token = self.next().expect("already peeked, it's fine");

                let op = BinaryOperator::try_from(token.kind)
                    .expect("already know the token is a binary operator");
                let rhs = self.parse_expression(level + 1)?;
                let span = lhs.span.to(rhs.span);

                lhs = Node {
                    item: Expression::Binary(BinaryExpression {
                        operator: op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    }),
                    span,
                };
            } else {
                break;
            }
        }

        Ok(lhs)
    }

    fn parse_primary(&mut self) -> Result<Node<Primary>, ParseError> {
        let token = self.next()?;

        let item = match token.kind {
            TokenKind::Identifier("true") => Primary::Bool(true),
            TokenKind::Identifier("false") => Primary::Bool(false),
            TokenKind::Integer(int) => Primary::Integer(int),
            TokenKind::Identifier(x) => Primary::Identifier(Identifier(x.to_string())),
            _ => todo!("invalid {:?}", token.kind),
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
            .parse_primary(),
            Ok(Node {
                item: Primary::Bool(true),
                span: Span { start: 0, end: 4 },
            })
        );

        assert_matches!(
            quick_parser(&[Ok(Token {
                kind: TokenKind::Identifier("false"),
                span: Span { start: 0, end: 5 },
            })])
            .parse_primary(),
            Ok(Node {
                item: Primary::Bool(false),
                span: Span { start: 0, end: 5 },
            })
        );

        assert_matches!(
            quick_parser(&[Ok(Token {
                kind: TokenKind::Integer(1234),
                span: Span { start: 0, end: 4 },
            })])
            .parse_primary(),
            Ok(Node {
                item: Primary::Integer(1234),
                span: Span { start: 0, end: 4 },
            })
        );

        assert_eq!(
            quick_parser(&[Ok(Token {
                kind: TokenKind::Identifier("hello"),
                span: Span { start: 0, end: 5 },
            })])
            .parse_primary(),
            Ok(Node {
                item: Primary::Identifier(Identifier(String::from("hello"))),
                span: Span { start: 0, end: 5 },
            })
        );
    }

    #[test]
    fn test_binary_expression() {
        assert_matches!(
            quick_parser(&[
                Ok(Token {
                    kind: TokenKind::Integer(1),
                    span: Span { start: 0, end: 1 }
                }),
                Ok(Token {
                    kind: TokenKind::Eof,
                    span: Span { start: 1, end: 1 }
                })
            ])
            .parse_expression(0),
            Ok(Node {
                item: Expression::Primary(Primary::Integer(1)),
                span: Span { start: 0, end: 1 }
            }),
        );

        assert_eq!(
            quick_parser(&[
                Ok(Token {
                    kind: TokenKind::Integer(1),
                    span: Span { start: 0, end: 1 }
                }),
                Ok(Token {
                    kind: TokenKind::Plus,
                    span: Span { start: 1, end: 2 }
                }),
                Ok(Token {
                    kind: TokenKind::Integer(2),
                    span: Span { start: 2, end: 3 }
                }),
                Ok(Token {
                    kind: TokenKind::Eof,
                    span: Span { start: 3, end: 3 }
                })
            ])
            .parse_expression(0),
            Ok(Node {
                item: Expression::Binary(BinaryExpression {
                    operator: BinaryOperator::Add,
                    lhs: Box::new(Node {
                        item: Expression::Primary(Primary::Integer(1)),
                        span: Span { start: 0, end: 1 }
                    }),
                    rhs: Box::new(Node {
                        item: Expression::Primary(Primary::Integer(2)),
                        span: Span { start: 2, end: 3 }
                    })
                }),
                span: Span { start: 0, end: 3 }
            })
        );

        assert_eq!(
            quick_parser(&[
                Ok(Token {
                    kind: TokenKind::Integer(1),
                    span: Span { start: 0, end: 1 }
                }),
                Ok(Token {
                    kind: TokenKind::Plus,
                    span: Span { start: 1, end: 2 }
                }),
                Ok(Token {
                    kind: TokenKind::Integer(2),
                    span: Span { start: 2, end: 3 }
                }),
                Ok(Token {
                    kind: TokenKind::Asterisk,
                    span: Span { start: 3, end: 4 }
                }),
                Ok(Token {
                    kind: TokenKind::Integer(3),
                    span: Span { start: 4, end: 5 }
                }),
                Ok(Token {
                    kind: TokenKind::Minus,
                    span: Span { start: 5, end: 6 }
                }),
                Ok(Token {
                    kind: TokenKind::Integer(4),
                    span: Span { start: 6, end: 7 }
                }),
                Ok(Token {
                    kind: TokenKind::Eof,
                    span: Span { start: 7, end: 7 }
                })
            ])
            .parse_expression(0),
            Ok(Node {
                item: Expression::Binary(BinaryExpression {
                    operator: BinaryOperator::Subtract,
                    lhs: Box::new(Node {
                        item: Expression::Binary(BinaryExpression {
                            operator: BinaryOperator::Add,
                            lhs: Box::new(Node {
                                item: Expression::Primary(Primary::Integer(1)),
                                span: Span { start: 0, end: 1 }
                            }),
                            rhs: Box::new(Node {
                                item: Expression::Binary(BinaryExpression {
                                    operator: BinaryOperator::Multiply,
                                    lhs: Box::new(Node {
                                        item: Expression::Primary(Primary::Integer(2)),
                                        span: Span { start: 2, end: 3 }
                                    }),
                                    rhs: Box::new(Node {
                                        item: Expression::Primary(Primary::Integer(3)),
                                        span: Span { start: 4, end: 5 }
                                    })
                                }),
                                span: Span { start: 2, end: 5 }
                            })
                        }),
                        span: Span { start: 0, end: 5 }
                    }),
                    rhs: Box::new(Node {
                        item: Expression::Primary(Primary::Integer(4)),
                        span: Span { start: 6, end: 7 }
                    })
                }),
                span: Span { start: 0, end: 7 }
            })
        );
    }
}
