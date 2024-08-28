use std::path::Path;
use anyhow::{Context, Result};
use std::fs;

#[derive(Debug, PartialEq)]
enum TokenType {
    LeftParanthesis,
    RightParanthesis,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    Identifier(String),
    StringLiteral(String),
    Number(f64),
    And,
    Class,
    Else,
    False,
    Fun,
    For,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,
    EndOfFile,
}

#[derive(Debug)]
struct Location {
    line: i32,
    column: i32,
}

#[derive(Debug)]
struct Token {
    token_type: TokenType,
    location: Location,
    lexeme: String,
}

struct CharSequence {
    chars: Vec<char>,
    pos: usize,
    line: i64,
    column: i64,
}

struct Location {
    line: i64,
    column: i64,
}

impl CharSequence {
    fn new (chars: &str) -> Self {
        Self {
            chars: chars.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    fn consume (&mut self) -> Option<(char, Location)> {
        if self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            let loc = Location {
                line: self.line,
                column: self.column,
            };
            self.advance(ch);
            Some((ch, loc))
        }
        else {
            None
        }
    }

    fn advance(&mut self, ch: char) {
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        }
        else {
            self.column += 1;
        }
    }

    fn has_reached_end (&self) -> bool {
        return self.pos >= self.chars.len()
    }
}


fn tokenize (content: &String) -> Vec<Token> {
    let mut seq = CharSequence::new(content.as_str());
    let mut tokens: Vec<Token> = Vec::new();
    let mut line = 0i64;
    let mut column: = 0i64;
    while seq.has_reached_end() == false {
        let next_char = seq.consume();
        match next_char {
            Some(ch) => {
                match ch {
                    '(' => {
                        tokens.push(
                            Token { 
                                token_type: TokenType::And,
                                location: Location { 
                                    line: 1,
                                    column: 1,
                                },
                                lexeme: String::from("and"),
                            }
                        )
                    }
                    _ => {
                        panic!("Unknown character")
                    }
                }
            }
            None => {
                break;
            }
        }
    }
    tokens    
}

#[cfg(test)]
mod tests {
    use crate::TokenType;

    use super::tokenize;
    use super::CharSequence;

    #[test]
    fn test_tokenizer_single_token () {
        let tokens = tokenize(&String::from("("));
        assert_eq!(tokens.len(), 1);
        let first_token = &tokens[0];
        assert_eq!(first_token.token_type, TokenType::And);
    }

    // fn construct_char_sequence (input: &str) -> CharSequence {
    //     let binding = String::from(input);
    //     let seq = CharSequence::new(&binding);
    //     seq
    // }

    // #[test]
    // fn test_char_sequence_empty_sequence () {
    //     let mut seq = construct_char_sequence("");
    //     assert_eq!(seq.consume(), None);
    //     assert_eq!(seq.has_reached_end(), true);
    // }

    // #[test]
    // fn test_char_sequene_single_char() {
    //     let mut seq = construct_char_sequence("a");
    //     assert_eq!(seq.consume(), Some('a'));
    //     assert_eq!(seq.consume(), None);
    //     assert_eq!(seq.has_reached_end(), true);
    // }

    // #[test]
    // fn test_char_sequence_multiple_chars() {
    //     let mut seq = construct_char_sequence("ab");
    //     assert_eq!(seq.consume(), Some('a'));
    //     assert_eq!(seq.consume(), Some('b'));
    //     assert_eq!(seq.consume(), None);
    //     assert_eq!(seq.has_reached_end(), true);
    // }
}


pub fn execute(file_path: &Path) -> Result<()> {
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file {}", file_path.display()))?;
    let tokens = tokenize(&content);
    dbg!(tokens);
    Ok(())
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
