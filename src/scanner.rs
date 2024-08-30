use anyhow::{Result, bail};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Location {
    line: i64,
    column: i64,
}

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

#[derive(Debug, PartialEq)]
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
}



#[derive(Debug)]
pub struct Token {
    token_type: TokenType,
    location: Location,
    lexeme: String,
}

pub struct Tokenizer {
    content: CharSequence,
    tokens: Vec<Token>,
}

impl Tokenizer {
    pub fn new (content: &str) -> Self {
        let tokenizer = Tokenizer {
            content: CharSequence::new(content),
            tokens: Vec::new(),
        };
        tokenizer
    }

    pub fn tokenize (&mut self) -> Result<&Vec<Token>> {
        if self.tokens.len() == 0 {
            while self.content.has_reached_end() == false {
                if let Some(ch) = self.content.look_at(0) {
                    match ch {
                        '(' => self.add_single_char_token(ch, TokenType::LeftParanthesis),
                        ')' => self.add_single_char_token(ch, TokenType::RightParanthesis),
                        '{' => self.add_single_char_token(ch, TokenType::LeftBrace),
                        '}' => self.add_single_char_token(ch, TokenType::RightBrace),
                        ',' => self.add_single_char_token(ch, TokenType::Comma),
                        '.' => self.add_single_char_token(ch, TokenType::Dot),
                        '-' => self.add_single_char_token(ch, TokenType::Minus),
                        '+' => self.add_single_char_token(ch, TokenType::Plus),
                        ';' => self.add_single_char_token(ch, TokenType::Semicolon),
                        '*' => self.add_single_char_token(ch, TokenType::Star),
                        '!' => self.add_comparison_token(ch, TokenType::Bang, TokenType::BangEqual),
                        '=' => self.add_comparison_token(ch, TokenType::Equal, TokenType::EqualEqual),
                        '>' => self.add_comparison_token(ch, TokenType::Greater, TokenType::GreaterOrEqual),
                        '<' => self.add_comparison_token(ch, TokenType::Less, TokenType::LessOrEqual),
                        '/' => self.add_slash_token_or_ignore_line_comment(),
                        def_char => {
                            if def_char.is_digit(10) {
                                self.add_number_token();
                            }
                            else if def_char.is_ascii_alphabetic() {
                                self.add_identifier_or_keyword_token();
                            }
                            else if def_char == '"' {
                                self.add_string_literal_token()?;
                            }
                            else if def_char.is_whitespace() {
                                self.content.advance(1);
                            }
                            else {
                                bail!("Invalid character detected: '{}' [{}:{}]", def_char, self.content.location.line, self.content.location.column,);
                            }
                        }
                    }
                }
            }
        }
        Ok(&self.tokens)
    }

    fn add_single_char_token (&mut self, ch: char, token_type: TokenType) {
        self.tokens.push(
            Token{
                token_type,
                location: self.content.location,
                lexeme: String::from(ch),
            }
        );
        self.content.advance(1);
    }

    fn add_comparison_token(&mut self, ch: char, token_type_no_equal: TokenType, token_type_equal: TokenType) {
        let mut token_type = token_type_no_equal;
        let mut lexeme = String::from(ch);
        let mut token_length = 1;
        
        if let Some('=') = self.content.look_at(1) {
            token_type = token_type_equal;
            lexeme.push('=');
            token_length = 2;
        }

        self.tokens.push(
            Token{
                token_type: token_type,
                location: self.content.location,
                lexeme,
            }
        );
        self.content.advance(token_length);
    }

    fn add_slash_token_or_ignore_line_comment(&mut self) {
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
            self.tokens.push(
                Token {
                    token_type: TokenType::Slash,
                    location: self.content.location,
                    lexeme: String::from('/'),
                }
            );
            self.content.advance(1);
        }
    }

    fn add_number_token(&mut self) {
        let mut lexeme = String::from(self.content.look_at(0).unwrap());
        let mut pos = 1usize;
        while let Some(ch) = self.content.look_at(pos) {
            if !(ch.is_digit(10) || ch == '.') {
                break;
            }
            lexeme.push(ch);
            pos += 1;
        }
        self.tokens.push(
            Token{
                token_type: TokenType::Number,
                location: self.content.location,
                lexeme,
            }
        );
        self.content.advance(pos);
    }

    fn add_identifier_or_keyword_token(&mut self) {
        let mut lexeme = String::from(self.content.look_at(0).unwrap());
        let mut pos = 1usize;
        while let Some(ch) = self.content.look_at(pos) {
            if !(ch.is_alphanumeric() || ch == '_') {
                break;
            }
            lexeme.push(ch);
            pos += 1;
        }
        self.tokens.push(
            Token{
                token_type: TokenType::Identifier,
                location: self.content.location,
                lexeme,
            }
        );
        self.content.advance(pos);
    }

    fn add_string_literal_token (&mut self) -> Result<()> {
        let mut lexeme = String::new();
        let mut pos = 1usize;
        while let Some(ch) = self.content.look_at(pos) {
            if ch == '"' {
                break;
            }
            else if ch == '\n' {
                bail!("Unfinished string literal [{},{}]", self.content.location.line, self.content.location.column);
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


#[cfg(test)]
mod tests {
    use super::{CharSequence, Location};

    fn construct_char_sequence (input: &str) -> CharSequence {
        CharSequence::new(input)
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
        assert_eq!(seq.location, Location{line: 1, column: 2});
        assert_eq!(seq.look_at(0), None);
        assert_eq!(seq.has_reached_end(), true);
    }

    #[test]
    fn test_char_sequence_multiple_chars() {
        let mut seq = construct_char_sequence("ab");
        assert_eq!(seq.look_at(0), Some('a'));
        assert_eq!(seq.look_at(1), Some('b'));
        seq.advance(2);
        assert_eq!(seq.location, Location{line: 1, column: 3});
        assert_eq!(seq.has_reached_end(), true);
    }

    #[test]
    fn test_char_sequence_with_newline() {
        let mut seq = construct_char_sequence("abcdef\nhij");
        assert_eq!(seq.look_at(0), Some('a'));
        seq.advance(5);
        assert_eq!(seq.location, Location{line: 1, column: 6});
        assert_eq!(seq.look_at(0), Some('f'));
        seq.advance(2);
        assert_eq!(seq.location, Location{line: 2, column: 1});
        assert_eq!(seq.look_at(0), Some('h'));
    }
}