use std::{fmt::Display, iter::Peekable};

use crate::{
    Span,
    error::{ParseError, ParseErrorKind},
    token::{Token, TokenKind},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    Binary(BinaryExpression),
    Unary(UnaryExpression),
    Primary(Primary),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryExpression {
    operator: BinaryOperator,
    lhs: Box<Node<Expression>>,
    rhs: Box<Node<Expression>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnaryExpression {
    operator: UnaryOperator,
    operand: Box<Node<Expression>>,
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
            TokenKind::EqualEqual => Ok(BinaryOperator::EqualEqual),
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

impl Display for BinaryExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {} {})", self.operator, self.lhs, self.rhs)
    }
}

impl Display for UnaryExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {})", self.operator, self.operand)
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

struct Parser<'code, It: Iterator<Item = Result<Token<'code>, ParseError>>> {
    tokens: Peekable<It>,
}

impl<'code, It: Iterator<Item = Result<Token<'code>, ParseError>>> Parser<'code, It> {
    fn parse_expression(&mut self) -> Result<Node<Expression>, ParseError> {
        let lhs = self.parse_expression_left(0)?;

        if self.peek()?.kind == TokenKind::Equal {
            self.next().expect("already peeked");

            let rhs = self.parse_expression()?;
            let span = lhs.span.to(rhs.span);

            return Ok(Node {
                item: Expression::Binary(BinaryExpression {
                    operator: BinaryOperator::Equals,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }),
                span,
            });
        }

        Ok(lhs)
    }

    fn parse_expression_left(&mut self, level: usize) -> Result<Node<Expression>, ParseError> {
        assert!(level <= BINARY_OP_TOKENS.len());

        if level == BINARY_OP_TOKENS.len() {
            return self.parse_unary_expression();
        }

        let ops = BINARY_OP_TOKENS[level];
        let mut lhs = self.parse_expression_left(level + 1)?;

        loop {
            let token = self.peek()?;

            if ops.contains(&token.kind) {
                let token = self.next().expect("already peeked");

                let op = BinaryOperator::try_from(token.kind)
                    .expect("already know the token is a binary operator");
                let rhs = self.parse_expression_left(level + 1)?;
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

    fn parse_unary_expression(&mut self) -> Result<Node<Expression>, ParseError> {
        let token = self.peek()?;
        let token_span = token.span;

        let op = match token.kind {
            TokenKind::Minus => UnaryOperator::Negate,
            TokenKind::Identifier("not") => UnaryOperator::Not,
            _ => return self.parse_paren_expression(),
        };

        self.next().expect("already peeked");

        let rhs = self.parse_unary_expression()?;
        let rhs_span = rhs.span;

        Ok(Node {
            item: Expression::Unary(UnaryExpression {
                operator: op,
                operand: Box::new(rhs),
            }),
            span: token_span.to(rhs_span),
        })
    }

    fn parse_paren_expression(&mut self) -> Result<Node<Expression>, ParseError> {
        match self.peek()?.kind == TokenKind::LParen {
            true => {
                self.next().expect("already peeked");
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RParen)?;

                Ok(expr)
            }
            false => {
                let primary = self.parse_primary()?;

                Ok(Node {
                    item: Expression::Primary(primary.item),
                    span: primary.span,
                })
            }
        }
    }

    fn parse_primary(&mut self) -> Result<Node<Primary>, ParseError> {
        let token = self.next()?;
        expect_some(&token)?;

        let item = match token.kind {
            TokenKind::Identifier("true") => Primary::Bool(true),
            TokenKind::Identifier("false") => Primary::Bool(false),
            TokenKind::Integer(int) => Primary::Integer(int),
            TokenKind::Identifier(x) => Primary::Identifier(Identifier(x.to_string())),
            _ => {
                return Err(ParseError {
                    kind: ParseErrorKind::ExpectedTokens(&["boolean", "integer", "identifier"]),
                    span: token.span,
                });
            }
        };

        Ok(Node {
            item,
            span: token.span,
        })
    }

    fn peek(&mut self) -> Result<&Token<'code>, ParseError> {
        // FIXME: unwrap
        self.tokens.peek().unwrap().as_ref().map_err(|err| *err)
    }

    fn next(&mut self) -> Result<Token<'code>, ParseError> {
        // only advance if eof is not reached yet
        match self.peek() {
            Ok(Token {
                kind: TokenKind::Eof,
                ..
            }) => self.peek().cloned(), // cheap clone if eof
            _ => self.tokens.next().unwrap(),
        }
    }

    fn expect(&mut self, token_kind: TokenKind<'static>) -> Result<(), ParseError> {
        let tok = self.next()?;

        if tok.kind == token_kind {
            Ok(())
        } else {
            Err(ParseError {
                kind: ParseErrorKind::ExpectedToken(token_kind),
                span: tok.span,
            })
        }
    }
}

/// Returns an error of [`ParseErrorKind::UnexpectedEof`] if the token is [`TokenKind::Eof`]
fn expect_some(token: &Token<'_>) -> Result<(), ParseError> {
    match token.kind {
        TokenKind::Eof => Err(ParseError {
            kind: ParseErrorKind::UnexpectedEof,
            span: token.span,
        }),
        _ => Ok(()),
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

    use crate::token::tests::{Tok::*, token_vec};
    use crate::{Span, token::TokenKind};

    use super::*;

    /// A wrapper for creating a [`Parser`] out of statically allocated tokens
    fn quick_parser<'a>(
        tokens: &[Result<Token<'a>, ParseError>],
    ) -> Parser<'a, impl Iterator<Item = Result<Token<'a>, ParseError>>> {
        Parser {
            tokens: tokens.iter().cloned().peekable(),
        }
    }

    #[test]
    fn test_literal() {
        assert_matches!(
            quick_parser(&token_vec(&[T(TokenKind::Identifier("true"), 4)])).parse_primary(),
            Ok(Node {
                item: Primary::Bool(true),
                span: Span { start: 0, end: 4 },
            })
        );

        assert_matches!(
            quick_parser(&token_vec(&[T(TokenKind::Identifier("false"), 5)])).parse_primary(),
            Ok(Node {
                item: Primary::Bool(false),
                span: Span { start: 0, end: 5 },
            })
        );

        assert_matches!(
            quick_parser(&token_vec(&[T(TokenKind::Integer(1234), 4)])).parse_primary(),
            Ok(Node {
                item: Primary::Integer(1234),
                span: Span { start: 0, end: 4 },
            })
        );

        assert_eq!(
            quick_parser(&token_vec(&[T(TokenKind::Identifier("hello"), 5)])).parse_primary(),
            Ok(Node {
                item: Primary::Identifier(Identifier(String::from("hello"))),
                span: Span { start: 0, end: 5 },
            })
        );
    }

    #[test]
    fn test_binary_expression() {
        assert_eq!(
            quick_parser(&token_vec(&[T(TokenKind::Integer(1), 1)]))
                .parse_expression()
                .unwrap()
                .to_string(),
            "1"
        );

        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Plus, 1),
                T(TokenKind::Integer(2), 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(+ 1 2)"
        );

        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Plus, 1),
                T(TokenKind::Integer(2), 1),
                T(TokenKind::Asterisk, 1),
                T(TokenKind::Integer(3), 1),
                T(TokenKind::Minus, 1),
                T(TokenKind::Integer(4), 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(- (+ 1 (* 2 3)) 4)"
        );
    }

    #[test]
    fn test_assignment() {
        // a=b=c
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Identifier("a"), 1),
                T(TokenKind::Equal, 1),
                T(TokenKind::Identifier("b"), 1),
                T(TokenKind::Equal, 1),
                T(TokenKind::Identifier("c"), 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(= a (= b c))"
        );

        // a=b+c=d
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Identifier("a"), 1),
                T(TokenKind::Equal, 1),
                T(TokenKind::Identifier("b"), 1),
                T(TokenKind::Plus, 1),
                T(TokenKind::Identifier("c"), 1),
                T(TokenKind::Equal, 1),
                T(TokenKind::Identifier("d"), 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(= a (= (+ b c) d))"
        );
    }

    #[test]
    fn test_unary() {
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Identifier("not"), 3),
                T(TokenKind::Identifier("not"), 3),
                T(TokenKind::Identifier("a"), 3),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(not (not a))"
        );

        // 1*not-1=2
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Asterisk, 1),
                T(TokenKind::Identifier("not"), 3),
                T(TokenKind::Minus, 1),
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Equal, 1),
                T(TokenKind::Integer(2), 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(= (* 1 (not (- 1))) 2)"
        );

        // 1---3
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Minus, 1),
                T(TokenKind::Minus, 1),
                T(TokenKind::Minus, 1),
                T(TokenKind::Integer(3), 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(- 1 (- (- 3)))"
        );
    }

    #[test]
    fn test_parentheses() {
        // 1*(2+3)
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Asterisk, 1),
                T(TokenKind::LParen, 1),
                T(TokenKind::Integer(2), 1),
                T(TokenKind::Plus, 1),
                T(TokenKind::Integer(3), 1),
                T(TokenKind::RParen, 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(* 1 (+ 2 3))"
        );

        // 1-(-2)
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Minus, 1),
                T(TokenKind::LParen, 1),
                T(TokenKind::Minus, 1),
                T(TokenKind::Integer(2), 1),
                T(TokenKind::RParen, 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(- 1 (- 2))"
        );

        // ((2)+3)
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::LParen, 1),
                T(TokenKind::LParen, 1),
                T(TokenKind::Integer(2), 1),
                T(TokenKind::RParen, 1),
                T(TokenKind::Plus, 1),
                T(TokenKind::Integer(3), 1),
                T(TokenKind::RParen, 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(+ 2 3)"
        );
    }

    #[test]
    fn test_expr_errors() {
        assert_eq!(
            quick_parser(&token_vec(&[T(TokenKind::Minus, 1),])).parse_expression(),
            Err(ParseError {
                kind: ParseErrorKind::UnexpectedEof,
                span: Span { start: 1, end: 1 }
            })
        );

        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Plus, 1),
                T(TokenKind::Plus, 1),
            ]))
            .parse_expression(),
            Err(ParseError {
                kind: ParseErrorKind::ExpectedTokens(&["boolean", "integer", "identifier"]),
                span: Span { start: 2, end: 3 }
            })
        );
    }
}
