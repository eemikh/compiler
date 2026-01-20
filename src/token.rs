use crate::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind<'a> {
    Identifier(&'a str),
    Integer(u32),
    Plus,
    Minus,
    Asterisk,
    Slash,
    Percent,
    Equal,
    EqualEqual,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token<'a> {
    kind: TokenKind<'a>,
    span: Span,
}

struct Tokenizer<'a> {
    code: &'a str,
    tokens: Vec<Token<'a>>,
    index: usize,
}

impl<'a> Tokenizer<'a> {
    fn tokenize(mut self) -> Vec<Token<'a>> {
        while let Some(c) = self.peek() {
            match c {
                '/' => {
                    if self.npeek(2) == Some('/') {
                        self.advance();
                        self.advance();

                        loop {
                            match self.peek() {
                                Some('\n') | None => break,
                                _ => {
                                    self.advance();
                                }
                            }
                        }
                    } else {
                        // must be the TokenKind::Slash
                        let tokenize_success = self.try_tokenize_operator();
                        assert!(tokenize_success);
                    }
                }
                'a'..='z' | 'A'..='Z' | '_' => {
                    self.tokenize_identifier();
                }
                ' ' | '\t' | '\n' => {
                    self.advance();
                }
                '0'..='9' => {
                    self.tokenize_integer();
                }
                _ => {
                    if !self.try_tokenize_operator() {
                        panic!("unexpected character");
                    }
                }
            }
        }

        self.tokens
    }

    fn try_tokenize_operator(&mut self) -> bool {
        let start = self.index;
        let Some(c) = self.peek() else { return false };
        let kind = match c {
            '+' => TokenKind::Plus,
            '-' => TokenKind::Minus,
            '*' => TokenKind::Asterisk,
            '/' => TokenKind::Slash,
            '%' => TokenKind::Percent,
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '{' => TokenKind::LBracket,
            '}' => TokenKind::RBracket,
            ',' => TokenKind::Comma,
            ';' => TokenKind::Semicolon,

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
                self.expect_peek('=');

                TokenKind::NotEqual
            }
            _ => return false,
        };
        self.advance();

        self.push(Token {
            kind,
            span: Span {
                start,
                end: self.index,
            },
        });

        true
    }

    fn tokenize_identifier(&mut self) {
        let start = self.index;

        while let Some(c) = self.peek() {
            match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => {
                    self.advance();
                }
                _ => break,
            }
        }

        let identifier = &self.code[start..self.index];
        self.push(Token {
            kind: TokenKind::Identifier(identifier),
            span: Span {
                start,
                end: self.index,
            },
        });
    }

    fn tokenize_integer(&mut self) {
        let mut integer: u32 = 0;
        let start = self.index;

        while let Some(c) = self.peek() {
            match c {
                '0'..='9' => {
                    self.advance();

                    // TODO: handle overflow
                    integer *= 10;
                    integer += c
                        .to_digit(10)
                        .expect("checked that the character is in digit range");
                }
                _ => break,
            }
        }

        self.push(Token {
            kind: TokenKind::Integer(integer),
            span: Span {
                start,
                end: self.index,
            },
        });
    }

    fn push(&mut self, token: Token<'a>) {
        self.tokens.push(token);
    }

    fn peek(&self) -> Option<char> {
        self.npeek(1)
    }

    fn npeek(&self, n: usize) -> Option<char> {
        self.code.get(self.index..)?.chars().nth(n - 1)
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.code.get(self.index..)?.chars().nth(0)?;
        self.index += c.len_utf8();
        Some(c)
    }

    fn expect_peek(&mut self, c: char) {
        if self.peek() != Some(c) {
            panic!("expected character {c}");
        }
    }
}

pub fn tokenize(code: &str) -> Vec<Token<'_>> {
    let tokenizer = Tokenizer {
        index: 0,
        code,
        tokens: Vec::new(),
    };

    tokenizer.tokenize()
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use super::*;

    #[test]
    fn simple_identifier() {
        assert_eq!(
            tokenize("if"),
            vec![Token {
                kind: TokenKind::Identifier("if"),
                span: Span { start: 0, end: 2 }
            }]
        );

        assert_eq!(
            tokenize("if while"),
            vec![
                Token {
                    kind: TokenKind::Identifier("if"),
                    span: Span { start: 0, end: 2 }
                },
                Token {
                    kind: TokenKind::Identifier("while"),
                    span: Span { start: 3, end: 8 }
                }
            ]
        );
    }

    #[test]
    fn complex_identifier() {
        assert_eq!(
            tokenize("ifififi if if hile \t\nn\nwhile a"),
            vec![
                Token {
                    kind: TokenKind::Identifier("ifififi"),
                    span: Span { start: 0, end: 7 }
                },
                Token {
                    kind: TokenKind::Identifier("if"),
                    span: Span { start: 8, end: 10 }
                },
                Token {
                    kind: TokenKind::Identifier("if"),
                    span: Span { start: 11, end: 13 }
                },
                Token {
                    kind: TokenKind::Identifier("hile"),
                    span: Span { start: 14, end: 18 }
                },
                Token {
                    kind: TokenKind::Identifier("n"),
                    span: Span { start: 21, end: 22 }
                },
                Token {
                    kind: TokenKind::Identifier("while"),
                    span: Span { start: 23, end: 28 }
                },
                Token {
                    kind: TokenKind::Identifier("a"),
                    span: Span { start: 29, end: 30 }
                }
            ]
        );
    }

    #[test]
    fn simple_integer() {
        assert_eq!(
            tokenize("834"),
            vec![Token {
                kind: TokenKind::Integer(834),
                span: Span { start: 0, end: 3 }
            }]
        );

        assert_eq!(
            tokenize("1239293234"),
            vec![Token {
                kind: TokenKind::Integer(1239293234),
                span: Span { start: 0, end: 10 }
            }]
        );
    }

    #[test]
    fn complex_integer() {
        assert_eq!(
            tokenize("1 328 129123 8432"),
            vec![
                Token {
                    kind: TokenKind::Integer(1),
                    span: Span { start: 0, end: 1 }
                },
                Token {
                    kind: TokenKind::Integer(328),
                    span: Span { start: 2, end: 5 }
                },
                Token {
                    kind: TokenKind::Integer(129123),
                    span: Span { start: 6, end: 12 }
                },
                Token {
                    kind: TokenKind::Integer(8432),
                    span: Span { start: 13, end: 17 }
                }
            ]
        );
    }

    #[test]
    fn more() {
        assert_eq!(
            tokenize("abcdef 1212 asnthueoa"),
            vec![
                Token {
                    kind: TokenKind::Identifier("abcdef"),
                    span: Span { start: 0, end: 6 }
                },
                Token {
                    kind: TokenKind::Integer(1212),
                    span: Span { start: 7, end: 11 }
                },
                Token {
                    kind: TokenKind::Identifier("asnthueoa"),
                    span: Span { start: 12, end: 21 }
                },
            ]
        );
    }

    #[test]
    fn simple_operator() {
        assert_matches!(
            tokenize("-").as_slice(),
            &[Token {
                kind: TokenKind::Minus,
                ..
            }]
        );

        assert_matches!(
            tokenize("- + + <=").as_slice(),
            &[
                Token {
                    kind: TokenKind::Minus,
                    ..
                },
                Token {
                    kind: TokenKind::Plus,
                    ..
                },
                Token {
                    kind: TokenKind::Plus,
                    ..
                },
                Token {
                    kind: TokenKind::LessEqual,
                    ..
                },
            ]
        );
    }

    #[test]
    fn test_singleline_comment() {
        assert_matches!(
            tokenize("a///+/=0[)!{+)@#}]\nb/a").as_slice(),
            &[
                Token {
                    kind: TokenKind::Identifier("a"),
                    ..
                },
                Token {
                    kind: TokenKind::Identifier("b"),
                    ..
                },
                Token {
                    kind: TokenKind::Slash,
                    ..
                },
                Token {
                    kind: TokenKind::Identifier("a"),
                    ..
                },
            ]
        );
    }
}
