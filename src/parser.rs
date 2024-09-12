use std::path::PathBuf;

use crate::ast::{BinaryOperator, Expr, Literal, UnaryOperator};
use crate::diagnostics::{DiagnosticError, Location, LocationSpan};
use crate::scanner::{Token, TokenType};
use anyhow::Result;

pub fn parse (tokens: &[Token], source_file: PathBuf) -> Result<Expr> {
    let mut parser = Parser::new(tokens, source_file);
    Ok(parser.parse()?)
}

fn get_binary_operator_from_token_type (token_type: &TokenType) -> BinaryOperator {
    match token_type {
        TokenType::EqualEqual => BinaryOperator::Equal,
        TokenType::BangEqual => BinaryOperator::NotEqual,
        TokenType::Greater => BinaryOperator::GreaterThan,
        TokenType::GreaterOrEqual => BinaryOperator::GreaterThanOrEqual,
        TokenType::Less => BinaryOperator::LessThan,
        TokenType::LessOrEqual => BinaryOperator::LessThanOrEqual,
        TokenType::Plus => BinaryOperator::Add,
        TokenType::Minus => BinaryOperator::Substract,
        TokenType::Star => BinaryOperator::Multiply,
        TokenType::Slash => BinaryOperator::Divide,
        _ => panic!("Invalid token type for binary operator: {:?}", token_type),
    }
}

fn get_unary_operator_from_token_type (token_type: &TokenType) -> UnaryOperator {
    match token_type {
        TokenType::Bang => UnaryOperator::Not,
        TokenType::Minus => UnaryOperator::Minus,
        _ => panic!("Invalid token type for unary operator: {:?}", token_type),
    }
}

struct TokenSequence<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> TokenSequence<'a> {
    fn new (tokens: &'a [Token]) -> Self {
        TokenSequence {
            tokens,
            pos: 0,
        }
    }

    fn current (&self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            return Some(self.tokens[self.pos].clone());
        }
        return None;
    }

    fn current_has_token_type (&self, token_types: &[TokenType]) -> bool {
        if let Some(token) = self.current() {
            token_types.contains(&token.token_type)
        }
        else {
            false
        }
    }

    fn advance (&mut self) {
        self.pos += 1;
    }
}

struct Parser<'a> {
    tokens: TokenSequence<'a>,
    source_file: PathBuf,
}

impl<'a> Parser<'a> {
    fn new (tokens: &'a [Token], source_file: PathBuf) -> Self {
        Parser { 
            tokens: TokenSequence::new(tokens),
            source_file
        }
    }

    fn parse (&mut self) -> Result<Expr> {
        Ok(self.parse_expression()?)
    }

    fn parse_binary_expr (&mut self, token_types: &[TokenType], parse_fn: impl Fn(&mut Self) -> Result<Expr>) -> Result<Expr> {
        let mut expr = parse_fn(self)?;

        while self.tokens.current_has_token_type(&token_types) {
            let token = self.tokens.current().unwrap();
            let op = get_binary_operator_from_token_type(&token.token_type);
            
            self.tokens.advance();
            let rhs = parse_fn(self)?;

            expr = Expr::Binary {
                lhs: Box::new(expr),
                op,
                rhs: Box::new(rhs),
                loc: LocationSpan {
                    start: 
                    end_inclusive: Location {
                        line: 0,
                        column: 0,
                    }
                }
            }
        }
        Ok(expr)
    }

    fn parse_expression (&mut self) -> Result<Expr> {
        Ok(self.parse_equality()?)
    }

    fn parse_equality (&mut self) -> Result<Expr> {
        let token_types = [
            TokenType::EqualEqual,
            TokenType::BangEqual,
        ];
        self.parse_binary_expr(&token_types, Self::parse_comparison)
    }

    fn parse_comparison (&mut self) -> Result<Expr> {
        let token_types = [
            TokenType::Greater,
            TokenType::GreaterOrEqual,
            TokenType::Less,
            TokenType::LessOrEqual,
            TokenType::EqualEqual,
            TokenType::BangEqual,
        ];
        self.parse_binary_expr(&token_types, Self::parse_term)
    }

    fn parse_term (&mut self) -> Result<Expr> {
        let token_types = [
            TokenType::Plus,
            TokenType::Minus,
        ];
        self.parse_binary_expr(&token_types, Self::parse_factor)
    }

    fn parse_factor (&mut self) -> Result<Expr> {
        let token_types = [
            TokenType::Star,
            TokenType::Slash,
        ];
        self.parse_binary_expr(&token_types, Self::parse_unary)
    }

    fn parse_unary (&mut self) -> Result<Expr> {
        let token_types = [
            TokenType::Minus,
            TokenType::Bang,
        ];
        if self.tokens.current_has_token_type(&token_types) {
            let token = self.tokens.current().unwrap();
            let op = get_unary_operator_from_token_type(&token.token_type);
            
            self.tokens.advance();
            let expr = self.parse_primary()?;
            Ok(
                Expr::Unary { 
                    op,
                    expr: Box::new(expr),
                }
            )
        }
        else {
            self.parse_primary()
        }
    }

    fn parse_primary (&mut self) -> Result<Expr> {
        if let Some(token) = self.tokens.current() {
            self.tokens.advance();
            let expr: Result<Expr> = match token.token_type {
                TokenType::True => Ok(Expr::Literal(Literal::Boolean(true))),
                TokenType::False => Ok(Expr::Literal(Literal::Boolean(false))),
                TokenType::Nil => Ok(Expr::Literal(Literal::Nil)),
                TokenType::StringLiteral => Ok(Expr::Literal(Literal::String(String::from(&token.lexeme)))),
                TokenType::Number => Ok(Expr::Literal(Literal::Number(token.lexeme.parse()?))),
                TokenType::LeftParanthesis => {
                    let expr = self.parse_expression()?;
                    match self.tokens.current() {
                        Some(closing_token) => {
                            if closing_token.token_type == TokenType::RightParanthesis {
                                self.tokens.advance();
                                return Ok(
                                    Expr::Grouping {
                                        expr: Box::new(expr),
                                    }
                                );
                            }
                            else {
                                return Err(DiagnosticError::new(
                                    format!("Expected closing paranthesis, but got: '{}'", &closing_token.lexeme),
                                    closing_token.location,
                                    self.source_file.clone(),
                                ).into());
                            }
                        },
                        None => {
                            Err(DiagnosticError::new(
                                format!("Expected closing paranthesis, but got end-of-file"),
                                token.location,
                                self.source_file.clone(),
                            ).into())
                        },
                    }
                },
                _ => {
                    Err(DiagnosticError::new(
                        format!("Unexpected token: '{}'", token.lexeme),
                        token.location,
                        self.source_file.clone(),
                    ).into())
                },
            };
            expr
        }
        else {
            return Err(DiagnosticError::new(
                format!("Expected a primary expression, but got end-of-file"),
                Location{ line: 0, column: 0},
                self.source_file.clone(),
            ).into());
        }
    }
}

#[cfg(test)]
mod tests {
    
}