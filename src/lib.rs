// use std::path::Path;
// use anyhow::{Context, Result};
// use std::fs;

mod scanner;

// #[derive(Debug, PartialEq)]
// enum TokenType {
//     LeftParanthesis,
//     RightParanthesis,
//     LeftBrace,
//     RightBrace,
//     Comma,
//     Dot,
//     Minus,
//     Plus,
//     Semicolon,
//     Slash,
//     Star,
//     Bang,
//     BangEqual,
//     Equal,
//     EqualEqual,
//     Greater,
//     GreaterOrEqual,
//     Less,
//     LessOrEqual,
//     Identifier,
//     StringLiteral,
//     Number,
//     And,
//     Class,
//     Else,
//     False,
//     Fun,
//     For,
//     If,
//     Nil,
//     Or,
//     Print,
//     Return,
//     Super,
//     This,
//     True,
//     Var,
//     While,
//     EndOfFile,
// }

// #[derive(Debug)]
// struct Location {
//     line: i32,
//     column: i32,
// }

// #[derive(Debug)]
// struct Token {
//     token_type: TokenType,
//     location: Location,
//     lexeme: String,
// }







// fn tokenize (content: &String) -> Vec<Token> {
//     let mut seq = CharSequence::new(content.as_str());
//     let mut tokens: Vec<Token> = Vec::new();
//     while seq.has_reached_end() == false {
//         let next_char = seq.consume();
//         match next_char {
//             Some(ch) => {
//                 match ch {
//                     '(' => {
//                         tokens.push(
//                             Token { 
//                                 token_type: TokenType::And,
//                                 location: Location { 
//                                     line: 1,
//                                     column: 1,
//                                 },
//                                 lexeme: String::from("and"),
//                             }
//                         )
//                     }
//                     _ => {
//                         panic!("Unknown character")
//                     }
//                 }
//             }
//             None => {
//                 break;
//             }
//         }
//     }
//     tokens    
// }

// pub fn execute(file_path: &Path) -> Result<()> {
//     let content = fs::read_to_string(file_path)
//         .with_context(|| format!("Failed to read file {}", file_path.display()))?;
//     let tokens = tokenize(&content);
//     dbg!(tokens);
//     Ok(())
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
