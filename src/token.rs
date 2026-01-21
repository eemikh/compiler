use crate::{Source, Span};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind<'a> {
    Identifier(&'a str),
    Integer(u32),
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token<'code> {
    kind: TokenKind<'code>,
    span: Span,
}

struct Tokenizer<'source, 'code> {
    source: &'source mut Source<'code>,
    index: usize,
}

impl<'source, 'code> Iterator for Tokenizer<'source, 'code> {
    type Item = Token<'code>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

impl<'source, 'code> Tokenizer<'source, 'code> {
    fn next_token(&mut self) -> Option<Token<'code>> {
        match self.peek()? {
            '#' => {
                self.advance_until_newline();
                self.next_token()
            }
            '/' => {
                if self.npeek(2) == Some('/') {
                    self.advance_until_newline();
                    self.next_token()
                } else {
                    // must be TokenKind::Slash
                    let token = self.tokenize_operator();
                    assert_eq!(
                        token.as_ref().map(|token| token.kind),
                        Some(TokenKind::Slash)
                    );
                    token
                }
            }
            'a'..='z' | 'A'..='Z' | '_' => self.tokenize_identifier(),
            ' ' | '\t' | '\n' => {
                self.advance();
                self.next_token()
            }
            '0'..='9' => self.tokenize_integer(),
            _ => Some(
                self.tokenize_operator()
                    .expect("TODO: return error if fails"),
            ),
        }
    }

    fn tokenize_operator(&mut self) -> Option<Token<'code>> {
        let start = self.index;
        let kind = match self.peek()? {
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
            _ => return None,
        };
        self.advance();

        Some(Token {
            kind,
            span: Span {
                start,
                end: self.index,
            },
        })
    }

    fn tokenize_identifier(&mut self) -> Option<Token<'code>> {
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
        Some(Token {
            kind: TokenKind::Identifier(identifier),
            span: Span {
                start,
                end: self.index,
            },
        })
    }

    fn tokenize_integer(&mut self) -> Option<Token<'code>> {
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

        Some(Token {
            kind: TokenKind::Integer(integer),
            span: Span {
                start,
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

    fn expect_peek(&mut self, c: char) {
        if self.peek() != Some(c) {
            panic!("expected character {c}");
        }
    }
}

pub fn tokenize<'a>(source: &mut Source<'a>) -> impl Iterator<Item = Token<'a>> {
    Tokenizer { index: 0, source }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use crate::Source;

    use super::*;

    /// A wrapper used for testing to not have to provide a full [`Source`] or dealing with iterators
    fn tokenize_str<'code>(code: &'code str) -> Vec<Token<'code>> {
        let mut code = Source::new(code);

        tokenize(&mut code).collect()
    }

    #[test]
    fn simple_identifier() {
        assert_eq!(
            tokenize_str("if"),
            vec![Token {
                kind: TokenKind::Identifier("if"),
                span: Span { start: 0, end: 2 }
            }]
        );

        assert_eq!(
            tokenize_str("if while"),
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
            tokenize_str("ifififi if if hile \t\nn\nwhile a"),
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
            tokenize_str("834"),
            vec![Token {
                kind: TokenKind::Integer(834),
                span: Span { start: 0, end: 3 }
            }]
        );

        assert_eq!(
            tokenize_str("1239293234"),
            vec![Token {
                kind: TokenKind::Integer(1239293234),
                span: Span { start: 0, end: 10 }
            }]
        );
    }

    #[test]
    fn complex_integer() {
        assert_eq!(
            tokenize_str("1 328 129123 8432"),
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
            tokenize_str("abcdef 1212 asnthueoa"),
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
            tokenize_str("-").as_slice(),
            &[Token {
                kind: TokenKind::Minus,
                ..
            }]
        );

        assert_matches!(
            tokenize_str("- + + <=").as_slice(),
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
    fn test_slashslash_comment() {
        assert_matches!(
            tokenize_str("a///+/=0[)!{+)@#}]\nb/a").as_slice(),
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

    #[test]
    fn test_hashtag_comment() {
        assert_matches!(
            tokenize_str("a/#/+/=0[)!{+)@#}]\nb/a").as_slice(),
            &[
                Token {
                    kind: TokenKind::Identifier("a"),
                    ..
                },
                Token {
                    kind: TokenKind::Slash,
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

    #[test]
    fn test_line_counting() {
        let mut source = Source::new("a\nb\n\n//\nb");

        tokenize(&mut source).for_each(|_| ());
        assert_eq!(source.newlines, vec![0, 2, 4, 5, 8]);
    }
}
