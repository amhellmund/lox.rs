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



#[derive(Debug, PartialEq)]
pub struct Token {
    token_type: TokenType,
    location: Location,
    lexeme: String,
}

impl Token {
    fn new (token_type: TokenType, line: i64, column: i64, lexeme: String) -> Self {
        Token {
            token_type,
            location: Location {
                line,
                column,
            },
            lexeme,
        }
    }
}

struct Tokenizer {
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
        Ok(self.tokens)
    }

    fn add_token (&mut self, token_type: TokenType, location: Location, lexeme: String) {
        self.tokens.push(
            Token { token_type, location, lexeme }
        )
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
        self.add_token(TokenType::Number, self.content.location, lexeme);
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
        self.add_token(TokenType::Identifier, self.content.location, lexeme);
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

pub fn tokenize (input: &str) -> Result<Vec<Token>> {
    let tokenizer = Tokenizer::new(input);
    let tokens = tokenizer.tokenize()?;
    Ok(tokens)
}


#[cfg(test)]
mod tests {
    use super::{CharSequence, Location, tokenize, Token, TokenType};
    use textwrap::dedent;

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
            let tokens = tokenize(input).unwrap();
            assert_eq!(tokens.len(), 1);

            let token = &tokens[0];
            assert_eq!(token.token_type, token_type);
            assert_eq!(token.location, Location { line: 1, column: 1});
            assert_eq!(token.lexeme, input);
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
            let tokens = tokenize(input).unwrap();
            assert_eq!(tokens.len(), 1);

            let token = &tokens[0];
            assert_eq!(token.token_type, TokenType::Number);
            assert_eq!(token.location, Location { line: 1, column: 1});
            assert_eq!(token.lexeme, input);
        }
    }

    #[test]
    fn test_numbers_with_leading_dot_as_two_tokens () {
        let tokens = tokenize(".25").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].token_type, TokenType::Dot);
        assert_eq!(tokens[1].token_type, TokenType::Number); 
    }

    #[test]
    fn test_string_literal () {
        let tokens = tokenize("\"literal\"").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].token_type, TokenType::StringLiteral);
        assert_eq!(tokens[0].lexeme, "literal");
    }

    #[test]
    fn test_string_literal_uncompleted () {
        let tokens = tokenize("\"literal\n\"");
        assert!(tokens.is_err());
    }

    #[test]
    fn test_comment () {
        let tokens = tokenize("12 // comment").unwrap();
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn test_multiple_tokens_with_multiple_lines () {
        let input = dedent(r#"
            12 + 4; // comment
            {
                "abc";
            }"#);
        let testdata = vec![
            Token::new(TokenType::Number, 2, 1, String::from("12")),
            Token::new(TokenType::Plus, 2, 4, String::from("+")),
            Token::new(TokenType::Number, 2, 6, String::from("4")),
            Token::new(TokenType::Semicolon, 2, 7, String::from(";")),
            Token::new(TokenType::LeftBrace, 3, 1, String::from("{")),
            Token::new(TokenType::StringLiteral, 4, 5, String::from("abc")),
            Token::new(TokenType::Semicolon, 4, 10, String::from(";")),
            Token::new(TokenType::RightBrace, 5, 1, String::from("}")),
        ];

        let tokens = tokenize(&input).unwrap();

        assert_eq!(tokens.len(), testdata.len());

        for (i, token) in tokens.iter().enumerate() {
            assert_eq!(token, &testdata[i]);
        }
    }
    
}