#[derive(Clone, Copy, PartialEq, Debug)]
struct Location {
    line: i64,
    column: i64,
}

struct CharSequence {
    chars: Vec<char>,
    pos: usize,
    loc: Location,
}

impl CharSequence {
    fn new (chars: &str) -> Self {
        Self {
            chars: chars.chars().collect(),
            pos: 0,
            loc: Location {
                line: 1,
                column: 1,
            }
        }
    }

    fn get_loc (&self) -> Location {
        self.loc
    }

    fn look_ahead (&self) -> Option<char> {
        if self.pos < self.chars.len() {
            Some(self.chars[self.pos])
        }
        else {
            None
        }
    }

    fn consume (&mut self) -> Option<char> {
        if let Some(next) = self.look_ahead() {
            self.advance_single_char(next);
            Some(next)
        }
        else {
            None
        }
    }

    fn advance_single_char(&mut self, ch: char) {
        if ch == '\n' {
            self.loc.line += 1;
            self.loc.column = 1;
        }
        else {
            self.loc.column += 1;
        }
        self.pos += 1;
    }

    fn has_reached_end (&self) -> bool {
        return self.pos >= self.chars.len()
    }
}

#[derive(Debug, PartialEq)]
enum TokenType {
    LeftParanthesis,
}

#[derive(Debug)]
struct Token {
    token_type: TokenType,
    location: Location,
    lexeme: String,
}


fn tokenize (content: &String) -> Vec<Token> {
    let mut seq = CharSequence::new(content.as_str());
    let mut tokens: Vec<Token> = Vec::new();

    let mut add_token = |token_type: TokenType, location: Location, lexeme: String| tokens.push(
        Token { 
            token_type,
            location,
            lexeme,
        }
    );

    while seq.has_reached_end() == false {
        let next_char = seq.look_ahead();
        if let Some(ch) = seq.look_ahead() {
            match ch {
                '(' => add_token(TokenType::LeftParanthesis, seq.loc, String::from("(")),
                ')' => add_token(TokenType::RightParanthesis, )
            }
        }
    }
    tokens    
}


#[cfg(test)]
mod tests {
    use super::{Location, CharSequence};

    // #[test]
    // fn test_tokenizer_single_token () {
    //     let tokens = tokenize(&String::from("("));
    //     assert_eq!(tokens.len(), 1);
    //     let first_token = &tokens[0];
    //     assert_eq!(first_token.token_type, TokenType::And);
    // }

    fn construct_char_sequence (input: &str) -> CharSequence {
        CharSequence::new(input)
    }

    #[test]
    fn test_char_sequence_empty_sequence () {
        let mut seq = construct_char_sequence("");
        let loc = seq.get_loc();
        assert_eq!(loc, Location{line: 1, column: 1});
        assert_eq!(seq.look_ahead(), None);
        assert_eq!(seq.consume(), None);
        assert_eq!(seq.has_reached_end(), true);
    }

    #[test]
    fn test_char_sequence_single_char() {
        let mut seq = construct_char_sequence("a");
        assert_eq!(seq.look_ahead(), Some('a'));
        assert_eq!(seq.look_ahead(), Some('a'));
        assert_eq!(seq.consume(), Some('a'));

        assert_eq!(seq.loc, Location{line: 1, column: 2});
        assert_eq!(seq.look_ahead(), None);
        assert_eq!(seq.consume(), None);
        assert_eq!(seq.has_reached_end(), true);

        assert_eq!(seq.consume(), None);
    }

    #[test]
    fn test_char_sequence_multiple_chars() {
        let mut seq = construct_char_sequence("ab");
        assert_eq!(seq.consume(), Some('a'));
        assert_eq!(seq.loc, Location{line: 1, column: 2});
        assert_eq!(seq.consume(), Some('b'));
        assert_eq!(seq.loc, Location{line: 1, column: 3});
        assert_eq!(seq.consume(), None);
        assert_eq!(seq.has_reached_end(), true);
    }

    #[test]
    fn test_char_sequence_with_newline() {
        let mut seq = construct_char_sequence("a\nb");
        assert_eq!(seq.consume(), Some('a'));

        assert_eq!(seq.look_ahead(), Some('\n'));
        assert_eq!(seq.consume(), Some('\n'));

        assert_eq!(seq.loc, Location{line: 2, column: 1});

        assert_eq!(seq.look_ahead(), Some('b'));
        assert_eq!(seq.consume(), Some('b'));

        assert_eq!(seq.has_reached_end(), true);
    }
}