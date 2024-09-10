use std::path::PathBuf;

use anyhow::Result;

use crate::diagnostics::{Location, DiagnosticError};

/// Wrapper around a sequence of characters providing convenience functions:
/// 
///   o Look at characters from the current position in the sequence
///   o Advance position while updating the location information
///   o End-of-sequence check
struct CharSequence {
    chars: Vec<char>,
    pos: usize,
    location: Location,
}

impl CharSequence {
    fn new (chars: &str) -> Self {
        Self {
            chars: chars.chars().collect(),
            pos: 0,
            location: Location { 
                line: 1,
                column: 1,
            }
        }
    }

    fn look_at (&self, num: usize) -> Option<char> {
        let index = self.pos + num;
        if index < self.chars.len() {
            Some(self.chars[index])
        }
        else {
            None
        }
    }

    fn advance (&mut self, num: usize) {
        for _ in 0..num {
            if self.pos < self.chars.len() {
               let ch = self.chars[self.pos]; 
               if ch == '\n' {
                    self.location.line += 1;
                    self.location.column = 1;
                }
                else if ch != '\r' {
                    self.location.column += 1;
                }
                self.pos += 1;
            }    
        }
    }

    fn has_reached_end (&self) -> bool {
        return self.pos >= self.chars.len()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TokenType {
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
    Identifier,
    StringLiteral,
    Number,
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
}


/// Token as an atomic element of the programming language
#[derive(Debug, PartialEq, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub location: Location,
    pub lexeme: String,
}

impl Token {
    fn new (token_type: TokenType, location: Location, lexeme: String) -> Self {
        Token {
            token_type,
            location,
            lexeme,
        }
    }
}

/// Utility class to split input code sequence into tokens
struct Tokenizer {
    content: CharSequence,
    tokens: Vec<Token>,
    source_file: PathBuf,
}

impl Tokenizer {
    pub fn new (content: &str, source_file: PathBuf) -> Self {
        let tokenizer = Tokenizer {
            content: CharSequence::new(content),
            tokens: Vec::new(),
            source_file: source_file,
        };
        tokenizer
    }

    pub fn tokenize (mut self) -> Result<Vec<Token>> {
        while self.content.has_reached_end() == false {
            if let Some(ch) = self.content.look_at(0) {
                match ch {
                    '(' => self.process_single_char_token(ch, TokenType::LeftParanthesis),
                    ')' => self.process_single_char_token(ch, TokenType::RightParanthesis),
                    '{' => self.process_single_char_token(ch, TokenType::LeftBrace),
                    '}' => self.process_single_char_token(ch, TokenType::RightBrace),
                    ',' => self.process_single_char_token(ch, TokenType::Comma),
                    '.' => self.process_single_char_token(ch, TokenType::Dot),
                    '-' => self.process_single_char_token(ch, TokenType::Minus),
                    '+' => self.process_single_char_token(ch, TokenType::Plus),
                    ';' => self.process_single_char_token(ch, TokenType::Semicolon),
                    '*' => self.process_single_char_token(ch, TokenType::Star),
                    '!' => self.process_comparison_token(ch, TokenType::Bang, TokenType::BangEqual),
                    '=' => self.process_comparison_token(ch, TokenType::Equal, TokenType::EqualEqual),
                    '>' => self.process_comparison_token(ch, TokenType::Greater, TokenType::GreaterOrEqual),
                    '<' => self.process_comparison_token(ch, TokenType::Less, TokenType::LessOrEqual),
                    '/' => self.process_slash_token_or_ignore_line_comment(),
                    def_char => {
                        if def_char.is_digit(10) {
                            self.process_number_token();
                        }
                        else if def_char.is_ascii_alphabetic() {
                            self.process_identifier_or_keyword_token();
                        }
                        else if def_char == '"' {
                            self.process_string_literal_token()?;
                        }
                        else if def_char.is_whitespace() {
                            self.content.advance(1);
                        }
                        else {
                            return Err(DiagnosticError::new(
                                format!("Invalid character detected: '{}'", def_char),
                                self.content.location,
                                self.source_file,
                            ).into());
                        }
                    }
                }
            }
        }
        Ok(self.tokens)
    }

    fn add_token (&mut self, token_type: TokenType, location: Location, lexeme: String) {
        self.tokens.push(Token::new(token_type, location, lexeme));
    }

    fn process_single_char_token (&mut self, ch: char, token_type: TokenType) {
        self.add_token(token_type, self.content.location, String::from(ch));
        self.content.advance(1);
    }

    fn process_comparison_token(&mut self, ch: char, token_type_no_equal: TokenType, token_type_equal: TokenType) {
        let mut token_type = token_type_no_equal;
        let mut lexeme = String::from(ch);
        let mut token_length = 1;
        
        if let Some('=') = self.content.look_at(1) {
            token_type = token_type_equal;
            lexeme.push('=');
            token_length = 2;
        }
        
        self.add_token(token_type, self.content.location, lexeme);
        self.content.advance(token_length);
    }

    fn process_slash_token_or_ignore_line_comment(&mut self) {
        if let Some('/') = self.content.look_at(1) {
            let mut pos = 2usize;
            while let Some(ch) = self.content.look_at(pos) {
                if ch == '\n' {
                    break
                }
                pos += 1;
            }
            self.content.advance(pos + 1);
        }
        else {
            self.add_token(TokenType::Slash, self.content.location, String::from('/'));
            self.content.advance(1);
        }
    }

    fn process_number_token(&mut self) {
        let mut lexeme = String::from(self.content.look_at(0).unwrap());
        let mut pos = 1usize;
        while let Some(ch) = self.content.look_at(pos) {
            if !(ch.is_digit(10) || ch == '.') {
                break;
            }
            lexeme.push(ch);
            pos += 1;
        }
        self.add_token(TokenType::Number, self.content.location, lexeme);
        self.content.advance(pos);
    }

    fn process_identifier_or_keyword_token(&mut self) {
        let mut lexeme = String::from(self.content.look_at(0).unwrap());
        let mut pos = 1usize;
        while let Some(ch) = self.content.look_at(pos) {
            if !(ch.is_alphanumeric() || ch == '_') {
                break;
            }
            lexeme.push(ch);
            pos += 1;
        }
        let token_type = match lexeme.as_str() {
            "and" => TokenType::And,
            "class" => TokenType::Class,
            "else" => TokenType::Else,
            "false" => TokenType::False,
            "fun" => TokenType::Fun,
            "for" => TokenType::For,
            "if" => TokenType::If,
            "nil" => TokenType::Nil,
            "or" => TokenType::Or,
            "print" => TokenType::Print,
            "return" => TokenType::Return,
            "super" => TokenType::Super,
            "this" => TokenType::This,
            "true" => TokenType::True,
            "var" => TokenType::Var,
            "while" => TokenType::While,
            _ => TokenType::Identifier,
        };
        self.add_token(token_type, self.content.location, lexeme);
        self.content.advance(pos);
    }

    fn process_string_literal_token (&mut self) -> Result<()> {
        let mut lexeme = String::new();
        let mut pos = 1usize;
        while let Some(ch) = self.content.look_at(pos) {
            if ch == '"' {
                break;
            }
            else if ch == '\n' {
                return Err(DiagnosticError::new(
                    format!("Unfinished string literal: '{}'", lexeme),
                    self.content.location,
                    self.source_file.clone(),
                ).into());
            }
            lexeme.push(ch);
            pos += 1;
        }
        self.tokens.push(
            Token{
                token_type: TokenType::StringLiteral,
                location: self.content.location,
                lexeme,
            }
        );
        self.content.advance(pos + 1);
        Ok(())
    }
}

pub fn tokenize (input: &str, source_file: PathBuf) -> Result<Vec<Token>> {
    let tokenizer = Tokenizer::new(input, source_file);
    let tokens = tokenizer.tokenize()?;
    Ok(tokens)
}


#[cfg(test)]
mod tests {
    use std::{path::PathBuf, str::FromStr};

    use super::{CharSequence, Location, tokenize, Token, TokenType};
    use textwrap::dedent;

    fn construct_char_sequence (input: &str) -> CharSequence {
        CharSequence::new(input)
    }

    fn new_loc (line: i64, column: i64) -> Location {
        Location {
            line,
            column,
        }
    }

    #[test]
    fn test_char_sequence_empty_sequence () {
        let seq = construct_char_sequence("");
        assert_eq!(seq.look_at(0), None);
        assert_eq!(seq.has_reached_end(), true);
    }

    #[test]
    fn test_char_sequence_single_char() {
        let mut seq = construct_char_sequence("a");
        assert_eq!(seq.look_at(0), Some('a'));
        assert_eq!(seq.look_at(0), Some('a'));
        seq.advance(1);
        assert_eq!(seq.location, new_loc(1, 2));
        assert_eq!(seq.look_at(0), None);
        assert_eq!(seq.has_reached_end(), true);
    }

    #[test]
    fn test_char_sequence_multiple_chars() {
        let mut seq = construct_char_sequence("ab");
        assert_eq!(seq.look_at(0), Some('a'));
        assert_eq!(seq.look_at(1), Some('b'));
        seq.advance(2);
        assert_eq!(seq.location, new_loc(1, 3));
        assert_eq!(seq.has_reached_end(), true);
    }

    #[test]
    fn test_char_sequence_with_newline() {
        let mut seq = construct_char_sequence("abcdef\nhij");
        assert_eq!(seq.look_at(0), Some('a'));
        seq.advance(5);
        assert_eq!(seq.location, new_loc(1, 6));
        assert_eq!(seq.look_at(0), Some('f'));
        seq.advance(2);
        assert_eq!(seq.location, new_loc(2, 1));
        assert_eq!(seq.look_at(0), Some('h'));
    }

    fn assert_tokens(input: &str, expected_tokens: Vec<Token>) {
        let tokens = tokenize(input, PathBuf::from_str("in-memory").unwrap()).unwrap();
        assert_eq!(tokens.len(), expected_tokens.len());
        for (index, token) in tokens.iter().enumerate() {
            assert_eq!(token, &expected_tokens[index]);
        }
    }

    fn new_token(token_type: TokenType, location: Location, lexeme: String) -> Token {
        Token::new(
            token_type,
            location,
            lexeme,
        )
    }

    #[test]
    fn test_scanner_elementary_tokens() {
        let test_data = vec![
            ("(", TokenType::LeftParanthesis),
            (")", TokenType::RightParanthesis),
            ("{", TokenType::LeftBrace),
            ("}", TokenType::RightBrace),
            (",", TokenType::Comma),
            (".", TokenType::Dot),
            ("-", TokenType::Minus),
            ("+", TokenType::Plus),
            (";", TokenType::Semicolon),
            ("*", TokenType::Star),
            ("!", TokenType::Bang),
            ("!=", TokenType::BangEqual),
            ("=", TokenType::Equal),
            ("==", TokenType::EqualEqual),
            (">", TokenType::Greater),
            (">=", TokenType::GreaterOrEqual),
            ("<", TokenType::Less),
            ("<=", TokenType::LessOrEqual),
            ("/", TokenType::Slash),
        ];

        for (input, token_type) in test_data {
            assert_tokens(input, vec![
                new_token(token_type, new_loc(1, 1), input.to_string())
            ]);
        }
    }

    #[test]
    fn test_number_tokens () {
        let test_data = vec![
            "1",
            "23",
            "456",
            "1.2",
            "4.",
            "0.25",
        ];
        for input in test_data {
            assert_tokens(input, vec![
                new_token(TokenType::Number, new_loc(1, 1), input.to_string())
            ]);
        }
    }

    #[test]
    fn test_numbers_with_leading_dot_as_two_tokens () {
        assert_tokens(".25", vec![
            new_token(TokenType::Dot, new_loc(1, 1), String::from(".")),
            new_token(TokenType::Number, new_loc(1, 2), String::from("25")),
        ]); 
    }

    #[test]
    fn test_string_literal () {
        assert_tokens("\"literal\"", vec![
            new_token(TokenType::StringLiteral, new_loc(1, 1), String::from("literal")),
        ]);
    }

    #[test]
    fn test_string_literal_uncompleted () {
        let tokens = tokenize("\"literal\n\"", PathBuf::from_str("in-memory").unwrap());
        assert!(tokens.is_err());
    }

    #[test]
    fn test_comment () {
        assert_tokens("12 // comment", vec![
            new_token(TokenType::Number, new_loc(1, 1), String::from("12")),
        ])
    }

    #[test]
    fn test_keywords () {
        let testdata = vec![
            ("and", TokenType::And),
            ("class", TokenType::Class),
            ("else", TokenType::Else),
            ("false", TokenType::False),
            ("fun", TokenType::Fun),
            ("for", TokenType::For),
            ("if", TokenType::If),
            ("nil", TokenType::Nil),
            ("or", TokenType::Or),
            ("print", TokenType::Print),
            ("return", TokenType::Return),
            ("super", TokenType::Super),
            ("this", TokenType::This),
            ("true", TokenType::True),
            ("var", TokenType::Var),
            ("while", TokenType::While),
        ];

        for (input, token_type) in testdata {
            assert_tokens(input, vec![
                new_token(token_type, new_loc(1, 1), input.to_string()),
            ]);
        }
    }

    #[test]
    fn test_identifier () {
        let testdata = vec![
            "a",
            "ab",
            "abc",
            "a1b",
            "a_1",
            "a_",
        ];

        for input in testdata {
            assert_tokens(input, vec![
                new_token(TokenType::Identifier, new_loc(1, 1), input.to_string()),
            ]);
        }
    }

    #[test]
    fn test_multiple_tokens_with_multiple_lines () {
        let input = dedent(r#"
            12 + 4; // comment
            {
                print "abc";
            }
            var b_ = nil;
            "#);
        let testdata = vec![
            new_token(TokenType::Number, new_loc(2, 1), String::from("12")),
            new_token(TokenType::Plus, new_loc(2, 4), String::from("+")),
            new_token(TokenType::Number, new_loc(2, 6), String::from("4")),
            new_token(TokenType::Semicolon, new_loc(2, 7), String::from(";")),
            new_token(TokenType::LeftBrace, new_loc(3, 1), String::from("{")),
            new_token(TokenType::Print, new_loc(4, 5), String::from("print")),
            new_token(TokenType::StringLiteral, new_loc(4, 11), String::from("abc")),
            new_token(TokenType::Semicolon, new_loc(4, 16), String::from(";")),
            new_token(TokenType::RightBrace, new_loc(5, 1), String::from("}")),
            new_token(TokenType::Var, new_loc(6, 1), String::from("var")),
            new_token(TokenType::Identifier, new_loc(6, 5), String::from("b_")),
            new_token(TokenType::Equal, new_loc(6, 8), String::from("=")),
            new_token(TokenType::Nil, new_loc(6, 10), String::from("nil")),
            new_token(TokenType::Semicolon, new_loc(6, 13), String::from(";")),
        ];
        assert_tokens(input.as_str(), testdata);
    }
    
}