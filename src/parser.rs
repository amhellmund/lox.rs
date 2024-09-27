mod token_sequence;

use std::path::PathBuf;

use crate::ast::{BinaryOperator, Expr, Literal, UnaryOperator};
use crate::diagnostics::{DiagnosticError, FileLocation, Location, LocationSpan};
use crate::scanner::{Token, TokenType};
use anyhow::Result;

use token_sequence::TokenSequence;

pub fn parse(tokens: Vec<Token>, source_file: PathBuf) -> Result<Expr> {
    let mut parser = Parser::new(tokens, source_file);
    Ok(parser.parse()?)
}

fn get_binary_operator_from_token_type(token_type: &TokenType) -> BinaryOperator {
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

fn get_unary_operator_from_token_type(token_type: &TokenType) -> UnaryOperator {
    match token_type {
        TokenType::Bang => UnaryOperator::Not,
        TokenType::Minus => UnaryOperator::Minus,
        _ => panic!("Invalid token type for unary operator: {:?}", token_type),
    }
}

struct Parser {
    tokens: TokenSequence,
    source_file: PathBuf,
}

impl Parser {
    fn new(tokens: Vec<Token>, source_file: PathBuf) -> Self {
        Parser {
            tokens: TokenSequence::new(tokens),
            source_file,
        }
    }

    fn merge_token_locations(expr_first: &Expr, expr_second: &Expr) -> LocationSpan {
        LocationSpan::new(
            expr_first.get_loc().start,
            expr_second.get_loc().end_inclusive,
        )
    }

    fn parse(&mut self) -> Result<Expr> {
        Ok(self.parse_expression()?)
    }

    // fn parse_program(&mut self) -> Result<Vec<Stmt>> {
    //     let mut statements: Vec<Stmt> = Vec::new();
    //     while !self.tokens.has_reached_end() {
    //         statements.push(self.parse_declaration()?);
    //     }
    //     Ok(statements)
    // }

    // fn parse_declaration(&mut self) -> Result<Stmt> {
    //     if self.tokens.current_has_token_type(&[TokenType::Var]) {
    //         self.tokens.advance();
    //         self.parse_variable_declaration()
    //     } else {
    //         self.parse_statement()
    //     }
    // }

    // fn parse_variable_declaration(&mut self) -> Result<Stmt> {
    //     if self.tokens.current_has_token_type(&[TokenType::Identifier]) {
    //         self.tokens.consume();
    //     } else {
    //     }
    // }

    // fn parse_statement(&mut self) -> Result<Stmt> {}

    fn parse_expression(&mut self) -> Result<Expr> {
        Ok(self.parse_equality()?)
    }

    fn parse_binary_expr(
        &mut self,
        token_types: &[TokenType],
        parse_fn: impl Fn(&mut Self) -> Result<Expr>,
    ) -> Result<Expr> {
        let mut expr = parse_fn(self)?;

        while let Some(token) = self.tokens.consume_if_has_token_type(&token_types) {
            let op = get_binary_operator_from_token_type(&token.token_type);

            let rhs = parse_fn(self)?;

            // the locations must get merged before the rhs expression gets moved into the new binary expression
            let loc = Self::merge_token_locations(&expr, &rhs);
            expr = Expr::Binary {
                lhs: Box::new(expr),
                op,
                rhs: Box::new(rhs),
                loc,
            }
        }
        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expr> {
        let token_types = [TokenType::EqualEqual, TokenType::BangEqual];
        self.parse_binary_expr(&token_types, Self::parse_comparison)
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
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

    fn parse_term(&mut self) -> Result<Expr> {
        let token_types = [TokenType::Plus, TokenType::Minus];
        self.parse_binary_expr(&token_types, Self::parse_factor)
    }

    fn parse_factor(&mut self) -> Result<Expr> {
        let token_types = [TokenType::Star, TokenType::Slash];
        self.parse_binary_expr(&token_types, Self::parse_unary)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        let token_types = [TokenType::Minus, TokenType::Bang];
        if let Some(token) = self.tokens.consume_if_has_token_type(&token_types) {
            let op = get_unary_operator_from_token_type(&token.token_type);

            let expr = self.parse_primary()?;
            // The location must get preserved before the expression gets moved into the new unary expression
            let loc = LocationSpan::new(token.location, expr.get_loc().end_inclusive);
            Ok(Expr::Unary {
                op,
                expr: Box::new(expr),
                loc,
            })
        } else {
            self.parse_primary()
        }
    }

    fn create_expr_from_literal_and_token(literal: Literal, token: &Token) -> Result<Expr> {
        Ok(Expr::Literal {
            literal,
            loc: LocationSpan::new(
                token.location,
                Location {
                    line: token.location.line,
                    column: token.location.column + token.lexeme.len() as i64 - 1,
                },
            ),
        })
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        if let Some(token) = self.tokens.consume_if_has_not_reached_end() {
            let expr: Result<Expr> = match token.token_type {
                TokenType::True => {
                    Self::create_expr_from_literal_and_token(Literal::Boolean(true), &token)
                }
                TokenType::False => {
                    Self::create_expr_from_literal_and_token(Literal::Boolean(false), &token)
                }
                TokenType::Nil => Self::create_expr_from_literal_and_token(Literal::Nil, &token),
                TokenType::StringLiteral => Self::create_expr_from_literal_and_token(
                    Literal::String(String::from(&token.lexeme)),
                    &token,
                ),
                TokenType::Number => Self::create_expr_from_literal_and_token(
                    Literal::Number(token.lexeme.parse()?),
                    &token,
                ),
                TokenType::LeftParanthesis => {
                    let expr = self.parse_expression()?;
                    if let Some(closing_token) = self
                        .tokens
                        .consume_if_has_token_type(&[TokenType::RightParanthesis])
                    {
                        return Ok(Expr::Grouping {
                            expr: Box::new(expr),
                            loc: LocationSpan::new(token.location, closing_token.location),
                        });
                    } else {
                        match self.tokens.current() {
                            Some(token) => {
                                return Err(DiagnosticError::new(
                                    format!(
                                        "Expected closing paranthesis, but got: '{}'",
                                        &token.lexeme
                                    ),
                                    FileLocation::SinglePoint(token.location),
                                    self.source_file.clone(),
                                )
                                .into());
                            }
                            None => {
                                return Err(DiagnosticError::new(
                                    format!("Unexpected token: '{}'", token.lexeme),
                                    FileLocation::SinglePoint(token.location),
                                    self.source_file.clone(),
                                )
                                .into());
                            }
                        }
                    }
                }
                _ => {
                    return Err(DiagnosticError::new(
                        format!("Unexpected token: '{}'", token.lexeme),
                        FileLocation::SinglePoint(token.location),
                        self.source_file.clone(),
                    )
                    .into());
                }
            };
            expr
        } else {
            return Err(DiagnosticError::new(
                format!("Expected a primary expression, but got end-of-file"),
                FileLocation::EndOfFile,
                self.source_file.clone(),
            )
            .into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    use crate::{
        ast::{BinaryOperator, UnaryOperator},
        diagnostics::{Location, LocationSpan},
        scanner::{Token, TokenType},
    };

    use crate::ast::{Expr, Literal};

    // token sequence: ToDo

    fn parse_and_check_literal(
        tokens: &[Token],
        expected_literal: Literal,
        expected_end_inclusive_loc: Location,
    ) {
        let ast = parse(&tokens, "in-memory".into());
        assert!(ast.is_ok());

        if let Expr::Literal { literal, loc } = ast.unwrap() {
            assert_eq!(literal, expected_literal);
            assert_eq!(
                loc,
                LocationSpan::new(Location::new(1, 1), expected_end_inclusive_loc)
            );
        } else {
            panic!("Invalid expression type returned")
        }
    }

    #[test]
    fn test_primary_expr_from_number() {
        let tokens = vec![Token::new(
            TokenType::Number,
            Location::new(1, 1),
            String::from("12.0"),
        )];
        parse_and_check_literal(&tokens, Literal::Number(12.0), Location::new(1, 4));
    }

    #[test]
    fn test_primary_expr_from_string() {
        let tokens = vec![Token::new(
            TokenType::StringLiteral,
            Location::new(1, 1),
            String::from("abc"),
        )];
        parse_and_check_literal(
            &tokens,
            Literal::String(String::from("abc")),
            Location::new(1, 3),
        );
    }

    #[test]
    fn test_primary_expr_from_boolean_true() {
        let tokens = vec![Token::new(
            TokenType::True,
            Location::new(1, 1),
            String::from("true"),
        )];
        parse_and_check_literal(&tokens, Literal::Boolean(true), Location::new(1, 4));
    }

    #[test]
    fn test_primary_expr_from_boolean_false() {
        let tokens = vec![Token::new(
            TokenType::False,
            Location::new(1, 1),
            String::from("false"),
        )];
        parse_and_check_literal(&tokens, Literal::Boolean(false), Location::new(1, 5));
    }

    #[test]
    fn test_primary_expr_from_boolean_nil() {
        let tokens = vec![Token::new(
            TokenType::Nil,
            Location::new(1, 1),
            String::from("nil"),
        )];
        parse_and_check_literal(&tokens, Literal::Nil, Location::new(1, 3));
    }

    #[test]
    fn test_grouping_expr() {
        let tokens = vec![
            Token::new(
                TokenType::LeftParanthesis,
                Location::new(1, 1),
                String::from("("),
            ),
            Token::new(TokenType::Number, Location::new(1, 2), String::from("0")),
            Token::new(
                TokenType::RightParanthesis,
                Location::new(1, 3),
                String::from(")"),
            ),
        ];
        let expected_ast = Expr::Grouping {
            expr: Box::new(Expr::Literal {
                literal: Literal::Number(0.0),
                loc: LocationSpan::new(Location::new(1, 2), Location::new(1, 2)),
            }),
            loc: LocationSpan::new(Location::new(1, 1), Location::new(1, 3)),
        };

        let ast = parse(&tokens, "in-memory".into()).unwrap();
        assert_eq!(ast, expected_ast);
    }

    #[test]
    fn test_binary_expr() {
        let test_data = vec![
            (TokenType::Plus, String::from("+"), BinaryOperator::Add),
            (
                TokenType::Minus,
                String::from("-"),
                BinaryOperator::Substract,
            ),
            (TokenType::Star, String::from("*"), BinaryOperator::Multiply),
            (TokenType::Slash, String::from("/"), BinaryOperator::Divide),
            (
                TokenType::Greater,
                String::from(">"),
                BinaryOperator::GreaterThan,
            ),
            (
                TokenType::GreaterOrEqual,
                String::from(">="),
                BinaryOperator::GreaterThanOrEqual,
            ),
            (TokenType::Less, String::from("<"), BinaryOperator::LessThan),
            (
                TokenType::LessOrEqual,
                String::from("<="),
                BinaryOperator::LessThanOrEqual,
            ),
            (
                TokenType::EqualEqual,
                String::from("=="),
                BinaryOperator::Equal,
            ),
            (
                TokenType::BangEqual,
                String::from("!="),
                BinaryOperator::NotEqual,
            ),
        ];
        for (token_type, lexeme, binary_op) in test_data {
            let lexeme_length = lexeme.len() as i64;
            let tokens = vec![
                Token::new(
                    TokenType::StringLiteral,
                    Location::new(1, 1),
                    String::from("a"),
                ),
                Token::new(token_type, Location::new(1, 2), lexeme),
                Token::new(
                    TokenType::StringLiteral,
                    Location::new(1, 2 + lexeme_length),
                    String::from("b"),
                ),
            ];
            let expected_ast = Expr::Binary {
                lhs: Box::new(Expr::Literal {
                    literal: Literal::String(String::from("a")),
                    loc: LocationSpan::new(Location::new(1, 1), Location::new(1, 1)),
                }),
                op: binary_op,
                rhs: Box::new(Expr::Literal {
                    literal: Literal::String(String::from("b")),
                    loc: LocationSpan::new(
                        Location::new(1, 2 + lexeme_length),
                        Location::new(1, 2 + lexeme_length),
                    ),
                }),
                loc: LocationSpan::new(Location::new(1, 1), Location::new(1, 2 + lexeme_length)),
            };
            let ast = parse(&tokens, "in-memory".into()).unwrap();
            assert_eq!(ast, expected_ast);
        }
    }

    #[test]
    fn test_unary_expr() {
        let test_data = vec![
            (TokenType::Minus, String::from("-"), UnaryOperator::Minus),
            (TokenType::Bang, String::from("!"), UnaryOperator::Not),
        ];
        for (token_type, lexeme, unary_op) in test_data {
            let tokens = vec![
                Token::new(token_type, Location::new(1, 1), lexeme),
                Token::new(TokenType::Number, Location::new(1, 2), String::from("1")),
            ];
            let expected_ast = Expr::Unary {
                op: unary_op,
                expr: Box::new(Expr::Literal {
                    literal: Literal::Number(1.0),
                    loc: LocationSpan::new(Location::new(1, 2), Location::new(1, 2)),
                }),
                loc: LocationSpan::new(Location::new(1, 1), Location::new(1, 2)),
            };
            let ast: Expr = parse(&tokens, "in-memory".into()).unwrap();
            assert_eq!(ast, expected_ast);
        }
    }

    #[test]
    fn test_multiline_expr() {
        let tokens = vec![
            Token::new(
                TokenType::StringLiteral,
                Location::new(1, 1),
                String::from("a"),
            ),
            Token::new(TokenType::Plus, Location::new(2, 2), String::from("+")),
            Token::new(
                TokenType::StringLiteral,
                Location::new(3, 4),
                String::from("b"),
            ),
        ];
        let expected_ast = Expr::Binary {
            lhs: Box::new(Expr::Literal {
                literal: Literal::String(String::from("a")),
                loc: LocationSpan::new(Location::new(1, 1), Location::new(1, 1)),
            }),
            op: BinaryOperator::Add,
            rhs: Box::new(Expr::Literal {
                literal: Literal::String(String::from("b")),
                loc: LocationSpan::new(Location::new(3, 4), Location::new(3, 4)),
            }),
            loc: LocationSpan::new(Location::new(1, 1), Location::new(3, 4)),
        };
        let ast = parse(&tokens, "in-memory".into()).unwrap();
        assert_eq!(ast, expected_ast);
    }
}
