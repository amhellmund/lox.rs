use crate::scanner::{Token, TokenType};

/// Utility around a sequence of tokens.
///
/// Precondition: The token sequence must always end with an `EndOfFile` token.
///
/// The token sequence will always return a valid token. In case the `EndOfFile` token
/// has been reached, it will continuously return this token.
///
/// The utility class provides the following convenience:
///
///   o consuming (advance+return) a token if the token at the current position has a given type.
///   o consuming (advance+return) a token at the current position in the sequence.
pub struct TokenSequence {
    tokens: Vec<Token>,
    pos: usize,
}

impl TokenSequence {
    pub fn new(tokens: Vec<Token>) -> Self {
        assert!(tokens[tokens.len() - 1].token_type == TokenType::EndOfFile);
        TokenSequence { tokens, pos: 0 }
    }

    pub fn current(&self) -> Token {
        self.tokens[self.pos].clone()
    }

    pub fn advance(&mut self) {
        self.pos += 1;
        if self.pos >= self.tokens.len() {
            self.pos = self.tokens.len() - 1;
        }
    }

    /// Consumes the current token if it has one of the given types types.
    ///
    /// In case the current token has one of the given token types, the current
    /// token is returned and the position in the token stream is advanced to
    /// the next token.
    ///
    /// In case the current token does not have the specified token, None is returned.
    pub fn consume_if_has_token_type(&mut self, token_types: &[TokenType]) -> Option<Token> {
        let current_token = self.current();
        if self.current_has_token_type(token_types) {
            self.advance();
            return Some(current_token);
        } else {
            return None;
        }
    }

    /// Consumes the current token, that is the current token gets returned. Beforehand the position
    /// gets advanced to the next position in the input sequence.
    ///
    /// Example: Token Sequence
    ///  | T01 | T02 | T03 | T04 | EOF |
    ///           ^
    /// Token `TO2` gets returned, and the position gets advanced to `T03` at the end of the function.
    ///
    /// Note: This function does always return a valid token. In case the `EndOfFile` token has been reached
    /// in the input sequence, it continuously returns this token.
    pub fn consume(&mut self) -> Token {
        let current_token = self.current();
        if !self.has_reached_end() {
            self.advance();
        }
        current_token
    }

    /// Returns if the end of the sequence has been reached.
    pub fn has_reached_end(&self) -> bool {
        self.pos == self.tokens.len() - 1
    }

    fn current_has_token_type(&self, token_types: &[TokenType]) -> bool {
        let token = self.current();
        token_types.contains(&token.token_type)
    }
}

#[cfg(test)]
mod tests {
    use super::TokenSequence;
    use crate::{
        diagnostics::Location,
        scanner::{Token, TokenType},
    };

    fn create_token_sequence(token_types: &[TokenType]) -> TokenSequence {
        let tokens: Vec<Token> = token_types
            .iter()
            .map(|token_type| Token::new(*token_type, Location::new(1, 1), String::default()))
            .collect();
        TokenSequence::new(tokens)
    }

    #[test]
    #[should_panic]
    fn test_missing_end_of_file_marker() {
        create_token_sequence(&[TokenType::And]);
    }

    #[test]
    fn test_current_and_advance() {
        let mut seq = create_token_sequence(&[TokenType::And, TokenType::EndOfFile]);
        assert_eq!(seq.current().token_type, TokenType::And);
        seq.advance();
        assert_eq!(seq.current().token_type, TokenType::EndOfFile);
    }

    #[test]
    fn test_advance_past_end_of_file() {
        let mut seq = create_token_sequence(&[TokenType::EndOfFile]);
        assert_eq!(seq.current().token_type, TokenType::EndOfFile);
        seq.advance();
        assert_eq!(seq.current().token_type, TokenType::EndOfFile);
    }

    #[test]
    fn test_consume_if_has_token_type() {
        let mut seq = create_token_sequence(&[TokenType::And, TokenType::EndOfFile]);
        let result = seq.consume_if_has_token_type(&[TokenType::And]);
        assert!(result.is_some());
        let token = result.unwrap();
        assert_eq!(token.token_type, TokenType::And);
    }

    #[test]
    fn test_consume_if_has_token_type_not_contained() {
        let mut seq = create_token_sequence(&[TokenType::And, TokenType::EndOfFile]);
        let result = seq.consume_if_has_token_type(&[TokenType::Bang]);
        assert!(result.is_none());
    }

    #[test]
    fn test_consume() {
        let mut seq = create_token_sequence(&[TokenType::And, TokenType::EndOfFile]);
        assert_eq!(seq.consume().token_type, TokenType::And);
        assert_eq!(seq.consume().token_type, TokenType::EndOfFile);
        assert_eq!(seq.consume().token_type, TokenType::EndOfFile);
    }

    #[test]
    fn test_token_values_preserved() {
        let tokens = vec![
            Token::new(TokenType::And, Location::new(1, 1), String::from("and")),
            Token::new(TokenType::Bang, Location::new(2, 1), String::from("!")),
            Token::new(TokenType::EndOfFile, Location::new(2, 2), String::default()),
        ];
        let mut seq = TokenSequence::new(tokens.clone());
        assert_eq!(seq.consume(), tokens[0]);
        assert_eq!(seq.consume(), tokens[1]);
        assert_eq!(seq.consume(), tokens[2]);
    }
}
