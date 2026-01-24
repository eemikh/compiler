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
    If(IfExpression),
    While(WhileExpression),
    Call(CallExpression),
    Block(BlockExpression),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryExpression {
    operator: BinaryOperator,
    lhs: Box<Node<Expression>>,
    rhs: Box<Node<Expression>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockExpression {
    expressions: Vec<Node<Expression>>,
    result_expression: Option<Box<Node<Expression>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallExpression {
    function: Box<Node<Expression>>,
    args: Vec<Node<Expression>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfExpression {
    condition: Box<Node<Expression>>,
    then: Box<Node<Expression>>,
    els: Option<Box<Node<Expression>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhileExpression {
    condition: Box<Node<Expression>>,
    body: Box<Node<Expression>>,
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
            _ => return self.parse_ternary_expression(),
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

    fn parse_ternary_expression(&mut self) -> Result<Node<Expression>, ParseError> {
        match self.peek()?.kind {
            TokenKind::Identifier("if") => {
                let token = self.next().expect("already peeked");
                let cond = self.parse_expression()?;
                self.expect(TokenKind::Identifier("then"))?;
                let then = self.parse_expression()?;

                let els = if self.peek()?.kind == TokenKind::Identifier("else") {
                    self.next().expect("already peeked");
                    Some(self.parse_expression()?)
                } else {
                    None
                };

                let span = token.span.to(els.as_ref().unwrap_or(&then).span);

                Ok(Node {
                    item: Expression::If(IfExpression {
                        condition: Box::new(cond),
                        then: Box::new(then),
                        els: els.map(Box::new),
                    }),
                    span,
                })
            }
            TokenKind::Identifier("while") => {
                let token = self.next().expect("already peeked");
                let cond = self.parse_expression()?;
                self.expect(TokenKind::Identifier("do"))?;
                let body = self.parse_expression()?;
                let span = token.span.to(body.span);

                Ok(Node {
                    item: Expression::While(WhileExpression {
                        condition: Box::new(cond),
                        body: Box::new(body),
                    }),
                    span,
                })
            }
            _ => self.parse_secondary_expression(),
        }
    }

    fn parse_secondary_expression(&mut self) -> Result<Node<Expression>, ParseError> {
        let mut expr = match self.peek()?.kind {
            TokenKind::LParen => {
                self.next().expect("already peeked");
                let expr = self.parse_expression()?;
                self.expect(TokenKind::RParen)?;

                Ok(expr)
            }
            TokenKind::LBrace => self.parse_block_expression(),
            _ => match self.parse_primary() {
                Some(primary) => Ok(Node {
                    item: Expression::Primary(primary.item),
                    span: primary.span,
                }),
                None => Err(ParseError {
                    kind: ParseErrorKind::ExpectedTokens(&[
                        "boolean",
                        "integer",
                        "identifier",
                        "parentheses",
                        "if expression",
                    ]),
                    span: self.peek()?.span,
                }),
            },
        }?;

        while self.peek().map(|token| token.kind) == Ok(TokenKind::LParen) {
            expr = self.parse_call(expr)?;
        }

        Ok(expr)
    }

    fn parse_block_expression(&mut self) -> Result<Node<Expression>, ParseError> {
        let start = self.expect(TokenKind::LBrace)?.span;

        let mut exprs = Vec::new();
        let mut ret_expression = None;

        while self.peek()?.kind != TokenKind::RBrace {
            let expr = self.parse_expression()?;

            if self.peek()?.kind == TokenKind::Semicolon {
                self.next().expect("already peeked");
                exprs.push(expr);
            } else {
                ret_expression = Some(expr);
                break;
            }
        }

        let end = self.expect(TokenKind::RBrace)?.span;

        Ok(Node {
            item: Expression::Block(BlockExpression {
                expressions: exprs,
                result_expression: ret_expression.map(Box::new),
            }),
            span: start.to(end),
        })
    }

    fn parse_call(&mut self, function: Node<Expression>) -> Result<Node<Expression>, ParseError> {
        self.expect(TokenKind::LParen)?;

        let mut args = Vec::new();

        if self.peek()?.kind != TokenKind::RParen {
            args.push(self.parse_expression()?);
        }

        while self.peek()?.kind == TokenKind::Comma {
            self.next().expect("peeked already");

            // allow trailing comma
            if self.peek()?.kind == TokenKind::RParen {
                break;
            }

            args.push(self.parse_expression()?);
        }

        let end = self.expect(TokenKind::RParen)?.span;
        let span = function.span.to(end);

        Ok(Node {
            item: Expression::Call(CallExpression {
                function: Box::new(function),
                args,
            }),
            span,
        })
    }

    fn parse_primary(&mut self) -> Option<Node<Primary>> {
        let token = self.peek().ok()?;
        expect_some(token).ok()?;

        let item = match token.kind {
            TokenKind::Identifier("true") => Primary::Bool(true),
            TokenKind::Identifier("false") => Primary::Bool(false),
            TokenKind::Integer(int) => Primary::Integer(int),
            TokenKind::Identifier(x) => Primary::Identifier(Identifier(x.to_string())),
            _ => return None,
        };

        let span = token.span;
        self.next().expect("already peeked");

        Some(Node { item, span })
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

    fn expect(&mut self, token_kind: TokenKind<'static>) -> Result<Token<'code>, ParseError> {
        let tok = self.next()?;

        if tok.kind == token_kind {
            Ok(tok)
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
            Some(Node {
                item: Primary::Bool(true),
                span: Span { start: 0, end: 4 },
            })
        );

        assert_matches!(
            quick_parser(&token_vec(&[T(TokenKind::Identifier("false"), 5)])).parse_primary(),
            Some(Node {
                item: Primary::Bool(false),
                span: Span { start: 0, end: 5 },
            })
        );

        assert_matches!(
            quick_parser(&token_vec(&[T(TokenKind::Integer(1234), 4)])).parse_primary(),
            Some(Node {
                item: Primary::Integer(1234),
                span: Span { start: 0, end: 4 },
            })
        );

        assert_eq!(
            quick_parser(&token_vec(&[T(TokenKind::Identifier("hello"), 5)])).parse_primary(),
            Some(Node {
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
                kind: ParseErrorKind::ExpectedTokens(&[
                    "boolean",
                    "integer",
                    "identifier",
                    "parentheses",
                    "if expression"
                ]),
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
                kind: ParseErrorKind::ExpectedTokens(&[
                    "boolean",
                    "integer",
                    "identifier",
                    "parentheses",
                    "if expression"
                ]),
                span: Span { start: 2, end: 3 }
            })
        );
    }

    #[test]
    fn test_if() {
        // 1 + if a == b then c * 3
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Integer(1), 2),
                T(TokenKind::Plus, 1),
                T(TokenKind::Identifier("if"), 2),
                T(TokenKind::Identifier("a"), 1),
                T(TokenKind::EqualEqual, 2),
                T(TokenKind::Identifier("b"), 1),
                T(TokenKind::Identifier("then"), 4),
                T(TokenKind::Identifier("c"), 1),
                T(TokenKind::Asterisk, 1),
                T(TokenKind::Integer(3), 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(+ 1 (if (== a b) (* c 3)))"
        );

        // if a == b then c + d else e
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Identifier("if"), 2),
                T(TokenKind::Identifier("a"), 1),
                T(TokenKind::EqualEqual, 2),
                T(TokenKind::Identifier("b"), 1),
                T(TokenKind::Identifier("then"), 4),
                T(TokenKind::Identifier("c"), 1),
                T(TokenKind::Plus, 1),
                T(TokenKind::Identifier("d"), 1),
                T(TokenKind::Identifier("else"), 4),
                T(TokenKind::Identifier("e"), 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(if (== a b) (+ c d) e)"
        );

        // if if a then b then if c then d else e else f
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Identifier("if"), 2),
                T(TokenKind::Identifier("if"), 2),
                T(TokenKind::Identifier("a"), 1),
                T(TokenKind::Identifier("then"), 4),
                T(TokenKind::Identifier("b"), 1),
                T(TokenKind::Identifier("then"), 4),
                T(TokenKind::Identifier("if"), 2),
                T(TokenKind::Identifier("c"), 1),
                T(TokenKind::Identifier("then"), 4),
                T(TokenKind::Identifier("d"), 1),
                T(TokenKind::Identifier("else"), 4),
                T(TokenKind::Identifier("e"), 1),
                T(TokenKind::Identifier("else"), 4),
                T(TokenKind::Identifier("f"), 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(if (if a b) (if c d e) f)"
        );
    }

    #[test]
    fn test_call() {
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Identifier("test"), 4),
                T(TokenKind::LParen, 1),
                T(TokenKind::RParen, 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(call test)"
        );

        // test(1+2,test1(),)
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Identifier("test"), 4),
                T(TokenKind::LParen, 1),
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Plus, 1),
                T(TokenKind::Integer(2), 1),
                T(TokenKind::Comma, 1),
                T(TokenKind::Identifier("test1"), 5),
                T(TokenKind::LParen, 1),
                T(TokenKind::RParen, 1),
                T(TokenKind::Comma, 1),
                T(TokenKind::RParen, 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(call test (+ 1 2) (call test1))"
        );

        // (test+1)()(a)
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::LParen, 1),
                T(TokenKind::Identifier("test"), 4),
                T(TokenKind::Plus, 1),
                T(TokenKind::Integer(1), 1),
                T(TokenKind::RParen, 1),
                T(TokenKind::LParen, 1),
                T(TokenKind::RParen, 1),
                T(TokenKind::LParen, 1),
                T(TokenKind::Identifier("a"), 1),
                T(TokenKind::RParen, 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(call (call (+ test 1)) a)"
        );
    }

    #[test]
    fn test_block() {
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::LBrace, 1),
                T(TokenKind::Identifier("test"), 4),
                T(TokenKind::Semicolon, 1),
                T(TokenKind::RBrace, 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(block test ())"
        );

        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::LBrace, 1),
                T(TokenKind::Identifier("test"), 4),
                T(TokenKind::Semicolon, 1),
                T(TokenKind::Identifier("test1"), 5),
                T(TokenKind::RBrace, 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(block test test1)"
        );

        // 1+{{1;2}*3}
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Plus, 1),
                T(TokenKind::LBrace, 1),
                T(TokenKind::LBrace, 1),
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Semicolon, 1),
                T(TokenKind::Integer(2), 1),
                T(TokenKind::RBrace, 1),
                T(TokenKind::Asterisk, 1),
                T(TokenKind::Integer(3), 1),
                T(TokenKind::RBrace, 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(+ 1 (block (* (block 1 2) 3)))"
        );
    }

    #[test]
    fn test_while() {
        assert_eq!(
            quick_parser(&token_vec(&[
                T(TokenKind::Identifier("while"), 4),
                T(TokenKind::Identifier("test"), 4),
                T(TokenKind::Identifier("do"), 2),
                T(TokenKind::Integer(1), 1),
                T(TokenKind::Plus, 1),
                T(TokenKind::Integer(1), 1),
            ]))
            .parse_expression()
            .unwrap()
            .to_string(),
            "(while test (+ 1 1))"
        );
    }
}
