#![feature(assert_matches)]
use std::env::args;

mod syntax;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn to(&self, other: Span) -> Span {
        assert!(self.start < other.start);

        Span {
            start: self.start,
            end: other.end,
        }
    }
}

#[derive(Debug, Clone)]
struct Source<'code> {
    code: &'code str,
    newlines: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SourceLocation {
    pub line: usize,
    pub col: usize,
}

impl<'code> Source<'code> {
    pub fn new(code: &'code str) -> Self {
        Self {
            code,
            newlines: vec![0],
        }
    }

    pub fn index_to_location(&self, index: usize) -> SourceLocation {
        let line = match self.newlines.binary_search(&index) {
            Ok(prev_line) => prev_line + 1,
            Err(line) => line,
        };

        let line_start = self
            .newlines
            .get(line - 1)
            .expect("line - 1 should always be in the vec");
        let col = index - line_start + 1;

        SourceLocation { line, col }
    }
}

fn main() {
    let mut args = args();
    println!("{}", args.nth(1).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location() {
        let mut source = Source::new("");
        source.newlines.extend([2, 3, 6, 8, 10]);

        assert_eq!(
            source.index_to_location(0),
            SourceLocation { line: 1, col: 1 }
        );
        assert_eq!(
            source.index_to_location(1),
            SourceLocation { line: 1, col: 2 }
        );
        assert_eq!(
            source.index_to_location(2),
            SourceLocation { line: 2, col: 1 }
        );
        assert_eq!(
            source.index_to_location(5),
            SourceLocation { line: 3, col: 3 }
        );
        assert_eq!(
            source.index_to_location(10),
            SourceLocation { line: 6, col: 1 }
        );
        assert_eq!(
            source.index_to_location(11),
            SourceLocation { line: 6, col: 2 }
        );
        assert_eq!(
            source.index_to_location(20),
            SourceLocation { line: 6, col: 11 }
        );
    }
}
