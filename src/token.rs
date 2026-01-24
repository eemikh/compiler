use crate::{
    Source, Span,
    error::{ParseError, ParseErrorKind},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind<'a> {
    Identifier(&'a str),
    Integer(u64),
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Asterisk,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `=`
    Equal,
    /// `==`
    EqualEqual,
    /// `!=`
    NotEqual,
    /// `<`
    LessThan,
    /// `<=`
    LessEqual,
    /// `>`
    GreaterThan,
    /// `>=`
    GreaterEqual,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `:`
    Colon,
    Eof,
}

impl TokenKind<'_> {
    pub fn identifier(&self) -> Option<&str> {
        match self {
            TokenKind::Identifier(identifier) => Some(identifier),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token<'code> {
    pub kind: TokenKind<'code>,
    pub span: Span,
}

struct Tokenizer<'source, 'code> {
    source: &'source mut Source<'code>,
    index: usize,
    had_eof: bool,
}

impl<'source, 'code> Iterator for Tokenizer<'source, 'code> {
    type Item = Result<Token<'code>, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

impl<'source, 'code> Tokenizer<'source, 'code> {
    fn next_token(&mut self) -> Option<Result<Token<'code>, ParseError>> {
        if self.had_eof {
            return None;
        }

        let res = match self.peek() {
            Some(c) => match c {
                '#' => {
                    self.advance_until_newline();
                    self.next_token()?
                }
                '/' => {
                    if self.npeek(2) == Some('/') {
                        self.advance_until_newline();
                        self.next_token()?
                    } else {
                        // must be TokenKind::Slash
                        let token = self.tokenize_operator();
                        assert_eq!(token.as_ref().unwrap().kind, TokenKind::Slash);
                        token
                    }
                }
                'a'..='z' | 'A'..='Z' | '_' => self.tokenize_identifier(),
                ' ' | '\t' | '\n' => {
                    self.advance();
                    self.next_token()?
                }
                '0'..='9' => self.tokenize_integer(),
                _ => self.tokenize_operator(),
            },
            None => self.eof_token(),
        };

        if let Ok(Token {
            kind: TokenKind::Eof,
            ..
        }) = res
        {
            self.had_eof = true;
        }

        Some(res)
    }

    fn tokenize_operator(&mut self) -> Result<Token<'code>, ParseError> {
        let start = self.index;
        let Some(c) = self.peek() else {
            return self.eof_token();
        };
        let kind = match c {
            '+' => TokenKind::Plus,
            '-' => TokenKind::Minus,
            '*' => TokenKind::Asterisk,
            '/' => TokenKind::Slash,
            '%' => TokenKind::Percent,
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            ',' => TokenKind::Comma,
            ';' => TokenKind::Semicolon,
            ':' => TokenKind::Colon,

            // potential multicharacter operators
            '<' => {
                self.advance();

                match self.peek() {
                    Some('=') => TokenKind::LessEqual,
                    _ => TokenKind::LessThan,
                }
            }
            '>' => {
                self.advance();

                match self.peek() {
                    Some('=') => TokenKind::GreaterEqual,
                    _ => TokenKind::GreaterThan,
                }
            }
            '=' => {
                self.advance();

                match self.peek() {
                    Some('=') => TokenKind::EqualEqual,
                    _ => TokenKind::Equal,
                }
            }

            // forced multicharacter operator
            '!' => {
                self.advance();
                self.expect_peek('=')?;

                TokenKind::NotEqual
            }
            _ => {
                self.advance();

                return Err(ParseError {
                    kind: ParseErrorKind::UnexpectedCharacter(c),
                    span: Span {
                        start,
                        end: self.index,
                    },
                });
            }
        };
        self.advance();

        Ok(Token {
            kind,
            span: Span {
                start,
                end: self.index,
            },
        })
    }

    fn tokenize_identifier(&mut self) -> Result<Token<'code>, ParseError> {
        let start = self.index;

        while let Some(c) = self.peek() {
            match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => {
                    self.advance();
                }
                _ => break,
            }
        }

        let identifier = &self.source.code[start..self.index];
        Ok(Token {
            kind: TokenKind::Identifier(identifier),
            span: Span {
                start,
                end: self.index,
            },
        })
    }

    fn tokenize_integer(&mut self) -> Result<Token<'code>, ParseError> {
        let mut integer: u64 = 0;
        let start = self.index;
        let mut had_overflow = false;

        while let Some(c) = self.peek() {
            match c {
                '0'..='9' => {
                    self.advance();

                    if !had_overflow {
                        let digit = c
                            .to_digit(10)
                            .expect("checked that the character is in digit range");

                        match integer
                            .checked_mul(10)
                            .and_then(|int| int.checked_add(digit.into()))
                        {
                            Some(res) => integer = res,
                            None => {
                                had_overflow = true;
                            }
                        }
                    }
                }
                _ => break,
            }
        }

        match had_overflow {
            false => Ok(Token {
                kind: TokenKind::Integer(integer),
                span: Span {
                    start,
                    end: self.index,
                },
            }),
            true => Err(ParseError {
                kind: ParseErrorKind::IntegerOverflow,
                span: Span {
                    start,
                    end: self.index,
                },
            }),
        }
    }

    fn eof_token(&self) -> Result<Token<'code>, ParseError> {
        Ok(Token {
            kind: TokenKind::Eof,
            span: Span {
                start: self.index,
                end: self.index,
            },
        })
    }

    fn advance_until_newline(&mut self) {
        loop {
            match self.peek() {
                Some('\n') | None => break,
                _ => {
                    self.advance();
                }
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.npeek(1)
    }

    fn npeek(&self, n: usize) -> Option<char> {
        self.source.code.get(self.index..)?.chars().nth(n - 1)
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.source.code.get(self.index..)?.chars().nth(0)?;

        self.index += c.len_utf8();

        if c == '\n' {
            self.source.newlines.push(self.index);
        }

        Some(c)
    }

    fn expect_peek(&mut self, c: char) -> Result<(), ParseError> {
        match self.peek() {
            Some(got) => match got == c {
                true => Ok(()),
                false => Err(ParseError {
                    kind: ParseErrorKind::ExpectedCharacter { expected: c, got },
                    span: Span {
                        start: self.index,
                        end: self.index + got.len_utf8(),
                    },
                }),
            },
            None => Err(ParseError {
                kind: ParseErrorKind::UnexpectedEof,
                span: Span {
                    start: self.index,
                    end: self.index,
                },
            }),
        }
    }
}

pub fn tokenize<'a>(
    source: &mut Source<'a>,
) -> impl Iterator<Item = Result<Token<'a>, ParseError>> {
    Tokenizer {
        index: 0,
        source,
        had_eof: false,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::Source;

    use super::*;

    /// A wrapper used for testing to not have to provide a full [`Source`] or dealing with iterators or errors
    pub(crate) fn tokenize_str<'code>(code: &'code str) -> Vec<Result<Token<'code>, ParseError>> {
        let mut code = Source::new(code);

        tokenize(&mut code).collect()
    }

    pub(crate) enum Tok<'a> {
        /// Token type, length
        T(TokenKind<'a>, usize),
        /// Whitespace
        W(usize),
        /// ParseError
        E(ParseErrorKind, usize),
    }

    use Tok::*;

    pub(crate) fn token_vec<'a>(tokens: &[Tok<'a>]) -> Vec<Result<Token<'a>, ParseError>> {
        let mut v = Vec::new();
        let mut i = 0;

        for a in tokens {
            match a {
                T(t, l) => {
                    let start = i;
                    i += l;

                    v.push(Ok(Token {
                        kind: *t,
                        span: Span { start, end: i },
                    }));
                }
                W(l) => {
                    i += l;
                }
                E(e, l) => {
                    v.push(Err(ParseError {
                        kind: *e,
                        span: Span {
                            start: i,
                            // the length is not consumed as an error may overlap with a token
                            end: i + l,
                        },
                    }));
                }
            }
        }

        v.push(Ok(Token {
            kind: TokenKind::Eof,
            span: Span { start: i, end: i },
        }));

        v
    }

    #[test]
    fn simple_identifier() {
        assert_eq!(
            tokenize_str("if"),
            token_vec(&[T(TokenKind::Identifier("if"), 2)])
        );

        assert_eq!(
            tokenize_str("if while"),
            token_vec(&[
                T(TokenKind::Identifier("if"), 2),
                W(1),
                T(TokenKind::Identifier("while"), 5)
            ])
        );
    }

    #[test]
    fn complex_identifier() {
        assert_eq!(
            tokenize_str("ifififi if if hile \t\nn\nwhile a"),
            token_vec(&[
                T(TokenKind::Identifier("ifififi"), 7),
                W(1),
                T(TokenKind::Identifier("if"), 2),
                W(1),
                T(TokenKind::Identifier("if"), 2),
                W(1),
                T(TokenKind::Identifier("hile"), 4),
                W(3),
                T(TokenKind::Identifier("n"), 1),
                W(1),
                T(TokenKind::Identifier("while"), 5),
                W(1),
                T(TokenKind::Identifier("a"), 1),
            ])
        );
    }

    #[test]
    fn simple_integer() {
        assert_eq!(
            tokenize_str("834"),
            token_vec(&[T(TokenKind::Integer(834), 3)])
        );

        assert_eq!(
            tokenize_str("18446744073709551615"),
            token_vec(&[T(TokenKind::Integer(u64::MAX), 20)])
        );

        assert_eq!(
            tokenize_str("18446744073709551616").as_slice(),
            token_vec(&[E(ParseErrorKind::IntegerOverflow, 20), W(20)])
        );
    }

    #[test]
    fn complex_integer() {
        assert_eq!(
            tokenize_str("1 328 129123 8432"),
            token_vec(&[
                T(TokenKind::Integer(1), 1),
                W(1),
                T(TokenKind::Integer(328), 3),
                W(1),
                T(TokenKind::Integer(129123), 6),
                W(1),
                T(TokenKind::Integer(8432), 4),
            ])
        );
    }

    #[test]
    fn more() {
        assert_eq!(
            tokenize_str("abcdef 1212 asnthueoa"),
            token_vec(&[
                T(TokenKind::Identifier("abcdef"), 6),
                W(1),
                T(TokenKind::Integer(1212), 4),
                W(1),
                T(TokenKind::Identifier("asnthueoa"), 9),
            ])
        );
    }

    #[test]
    fn simple_operator() {
        assert_eq!(tokenize_str("-"), token_vec(&[T(TokenKind::Minus, 1),]));

        assert_eq!(
            tokenize_str("- + + <="),
            token_vec(&[
                T(TokenKind::Minus, 1),
                W(1),
                T(TokenKind::Plus, 1),
                W(1),
                T(TokenKind::Plus, 1),
                W(1),
                T(TokenKind::LessEqual, 2),
            ])
        );
    }

    #[test]
    fn test_slashslash_comment() {
        assert_eq!(
            tokenize_str("a///+/=0[)!{+)@#}]\nb/a"),
            token_vec(&[
                T(TokenKind::Identifier("a"), 1),
                W(18),
                T(TokenKind::Identifier("b"), 1),
                T(TokenKind::Slash, 1),
                T(TokenKind::Identifier("a"), 1),
            ])
        );
    }

    #[test]
    fn test_hashtag_comment() {
        assert_eq!(
            tokenize_str("a/#/+/=0[)!{+)@#}]\nb/a"),
            token_vec(&[
                T(TokenKind::Identifier("a"), 1),
                T(TokenKind::Slash, 1),
                W(17),
                T(TokenKind::Identifier("b"), 1),
                T(TokenKind::Slash, 1),
                T(TokenKind::Identifier("a"), 1),
            ])
        );
    }

    #[test]
    fn test_line_counting() {
        let mut source = Source::new("a\nb\n\n//\nb");

        tokenize(&mut source).for_each(|_| ());
        assert_eq!(source.newlines, vec![0, 2, 4, 5, 8]);
    }

    #[test]
    fn test_errors() {
        assert_eq!(
            tokenize_str("a!"),
            token_vec(&[
                T(TokenKind::Identifier("a"), 1),
                W(1),
                E(ParseErrorKind::UnexpectedEof, 0),
            ])
        );

        assert_eq!(
            tokenize_str("a!a"),
            token_vec(&[
                T(TokenKind::Identifier("a"), 1),
                W(1),
                E(
                    ParseErrorKind::ExpectedCharacter {
                        expected: '=',
                        got: 'a'
                    },
                    1
                ),
                T(TokenKind::Identifier("a"), 1),
            ])
        );

        assert_eq!(
            tokenize_str("ä"),
            token_vec(&[E(ParseErrorKind::UnexpectedCharacter('ä'), 2), W(2),])
        );
    }
}
