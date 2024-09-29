// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

//! Utility around a sequence of tokens.
//!
//! Precondition: The token sequence must always end with an `EndOfFile` token.
//!
//! The token sequence will always return a valid token. In case the `EndOfFile` token
//! has been reached, it will continuously return this token.
//!
//! The utility class provides the following convenience:
//!
//!   o consuming (advance+return) a token if the token at the current position has a given type.
//!   o consuming (advance+return) a token at the current position in the sequence.

use crate::scanner::{Token, TokenType};

pub struct TokenSequence {
    tokens: Vec<Token>,
    pos: usize,
}

impl TokenSequence {
    pub fn new(tokens: Vec<Token>) -> Self {
        assert!(tokens[tokens.len() - 1].token_type == TokenType::EndOfFile);
        TokenSequence { tokens, pos: 0 }
    }

    /// Returns the token at the current position of the token sequence.
    pub fn current(&self) -> Token {
        self.tokens[self.pos].clone()
    }

    /// Advances the position in the token sequence to the next position.
    ///
    /// In case the position is already at the end of the sequence, it remains at this position.
    ///
    /// Post-Condition: the sequence points to a valid token, e.g. EndOfFile.
    pub fn advance(&mut self) {
        self.pos += 1;
        if self.pos >= self.tokens.len() {
            self.pos = self.tokens.len() - 1;
        }
    }

    /// Returns if the end of the sequence has been reached.
    pub fn has_reached_end(&self) -> bool {
        self.pos == self.tokens.len() - 1
    }

    /// Checks if the current token has one of the specified token types.
    pub fn current_has_token_type(&self, token_types: &[TokenType]) -> bool {
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
}
