// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

//! Recursive-descent Parser for the Lox programming language.
//!
//! The syntax and reference of the Lox programming langauge is documented here:
//!
//!   <https://craftinginterpreters.com/the-lox-language.html>
//!
//! The terms and notations will be re-used in the code and documentation of the
//! individual functions to better map them.

mod token_sequence;

use std::path::PathBuf;

use crate::ast::{BinaryOperator, Expr, ExprData, Literal, Stmt, StmtData, UnaryOperator};
use crate::diagnostics::{emit_diagnostic, FileLocation, Location, LocationSpan};
use crate::scanner::{Token, TokenType};
use anyhow::Result;

use token_sequence::TokenSequence;

/// Converts a sequence of tokens into an Abstract Syntax Tree (AST).
pub fn parse(tokens: Vec<Token>, source_file: PathBuf) -> Result<Stmt> {
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

/// Implementation class to convert the sequence of tokens into the AST.
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

    fn create_location_span_from_stmts(statements: &Vec<Stmt>) -> LocationSpan {
        if statements.len() == 0 {
            LocationSpan::new(Location::new(0, 0), Location::new(0, 0))
        } else {
            LocationSpan::new(
                statements.first().unwrap().get_loc().start,
                statements.last().unwrap().get_loc().end_inclusive,
            )
        }
    }

    fn create_location_span_from_token(token: &Token) -> LocationSpan {
        LocationSpan::new(
            token.location,
            Location::new(
                token.location.line,
                token.location.column + token.lexeme.len() as i64 - 1,
            ),
        )
    }

    /// Consumes the current token if it has one of the given types types.
    ///
    /// In case the current token has one of the given token types, the current
    /// token is returned and the position in the token stream is advanced to
    /// the next token.
    ///
    /// In case the current token does not have the specified token, None is returned.
    pub fn consume_if_has_token_types(&mut self, token_types: &[TokenType]) -> Option<Token> {
        let current_token = self.tokens.current();
        if self.tokens.current_has_token_type(token_types) {
            self.tokens.advance();
            return Some(current_token);
        } else {
            return None;
        }
    }

    /// Consumes the current token if it has the given token type.
    pub fn consume_if_has_token_type(&mut self, token_type: TokenType) -> Option<Token> {
        self.consume_if_has_token_types(&[token_type])
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
        let current_token = self.tokens.current();
        self.tokens.advance();
        current_token
    }

    /// Consumes the current token if it has the specified type, otherwise an error is returned.
    fn consume_or_error(&mut self, token_type: TokenType) -> Result<Token> {
        let token = self.tokens.current();
        if token.token_type == token_type {
            self.tokens.advance();
            Ok(token)
        } else {
            Err(emit_diagnostic(
                format!(
                    "Expected token '{}', but got: '{}'",
                    token_type.to_string(),
                    token.token_type.to_string(),
                ),
                FileLocation::SinglePoint(token.location),
                &self.source_file,
            ))
        }
    }

    fn parse(mut self) -> Result<Stmt> {
        Ok(self.parse_program()?)
    }

    /// Parses the whole program (or parts of it in case of REPL input).
    ///
    /// Grammar rule:
    ///
    ///   program: statement* EndOfFile
    fn parse_program(&mut self) -> Result<Stmt> {
        let mut declarations: Vec<Stmt> = Vec::new();
        while !self.tokens.has_reached_end() {
            declarations.push(self.parse_declaration()?);
        }
        let loc = Self::create_location_span_from_stmts(&declarations);
        Ok(Stmt::new(
            StmtData::List {
                statements: declarations,
            },
            loc,
        ))
    }

    /// Parses a declaration.
    ///
    /// Grammar rule:
    ///
    ///   declaration: variable_declaration
    ///              | statement
    fn parse_declaration(&mut self) -> Result<Stmt> {
        if self.tokens.current_has_token_type(&[TokenType::Var]) {
            self.parse_variable_declaration()
        } else {
            self.parse_statement()
        }
    }

    /// Parses variable declaration.
    ///
    /// Grammar rule:
    ///   variable_declaration: 'var' 'identifer'
    ///                       | 'var' 'identifier' '=' expression ';'
    ///
    /// This function assumes that the `Var` token has not yet been consumed.
    fn parse_variable_declaration(&mut self) -> Result<Stmt> {
        let var_token = self.consume_or_error(TokenType::Var)?;
        let identifier = self.consume_or_error(TokenType::Identifier)?;

        let init_expr;
        if let Some(_) = self.consume_if_has_token_type(TokenType::Equal) {
            init_expr = self.parse_expression()?;
        } else {
            // The init expression is set to 'nil' in case there is no initializer.
            init_expr = Expr::new_literal(
                Literal::Nil,
                Self::create_location_span_from_token(&identifier),
            );
        }

        let semicolon_token = self.consume_or_error(TokenType::Semicolon)?;

        Ok(Stmt::new(
            StmtData::VarDecl {
                identifier: identifier.lexeme,
                init_expr: init_expr.as_box(),
            },
            LocationSpan::new(var_token.location, semicolon_token.location),
        ))
    }

    /// Parses a statement.
    ///
    /// Grammar rule:
    ///
    ///   statement: print_statement
    ///            | expression_statement
    ///            | block_statement
    ///            | if_statement
    ///            | while_statement
    fn parse_statement(&mut self) -> Result<Stmt> {
        match self.tokens.current().token_type {
            TokenType::Print => self.parse_print_statement(),
            TokenType::LeftBrace => self.parse_block_statement(),
            TokenType::If => self.parse_if_statement(),
            TokenType::While => self.parse_while_statement(),
            _ => self.parse_expression_statement(),
        }
    }

    /// Parses a print statement.
    ///
    /// Grammar rule:
    ///
    ///   print_statement: 'print' expression ';'
    ///
    /// This function assumes that the 'print' token has not yet been consumed.
    fn parse_print_statement(&mut self) -> Result<Stmt> {
        let print_token = self.consume_or_error(TokenType::Print)?;
        let expr = self.parse_expression()?;
        let semicolon_token = self.consume_or_error(TokenType::Semicolon)?;

        Ok(Stmt::new(
            StmtData::Print {
                expr: Box::new(expr),
            },
            LocationSpan::new(print_token.location, semicolon_token.location),
        ))
    }

    /// Parses an expression statement.
    ///
    /// Grammar rule:
    ///
    ///   expression_statement: expression ';'
    fn parse_expression_statement(&mut self) -> Result<Stmt> {
        let expr = self.parse_expression()?;
        let semicolon_token = self.consume_or_error(TokenType::Semicolon)?;

        // The start location of the expression gets preserved because the expr gets moved into the statement.
        let start_loc = expr.get_loc().start;
        Ok(Stmt::new(
            StmtData::Expr {
                expr: Box::new(expr),
            },
            LocationSpan::new(start_loc, semicolon_token.location),
        ))
    }

    /// Parses a block statement.
    ///
    /// Grammar rule:
    ///
    ///   block_statement: '{' statement* '}'
    ///
    /// This function assumes that the '{' token has not yet been consumed.
    fn parse_block_statement(&mut self) -> Result<Stmt> {
        let (statements, loc) = self.parse_block()?;
        Ok(Stmt::new(StmtData::Block { statements }, loc))
    }

    /// Helper function to parse a block with statements.
    ///
    /// This function assumes that the `{` token has not yet been consumed.
    fn parse_block(&mut self) -> Result<(Vec<Stmt>, LocationSpan)> {
        let left_brace = self.consume_or_error(TokenType::LeftBrace)?;
        let mut statements = Vec::<Stmt>::new();
        while !self.tokens.current_has_token_type(&[TokenType::RightBrace])
            && !self.tokens.has_reached_end()
        {
            statements.push(self.parse_declaration()?);
        }
        let right_brace = self.consume_or_error(TokenType::RightBrace)?;

        Ok((
            statements,
            LocationSpan::new(left_brace.location, right_brace.location),
        ))
    }

    /// Parses an If statement.
    ///
    /// Grammar rule:
    ///
    ///   if_statement: 'if' '(' expression ')' statement ( 'else' statement )?
    ///
    /// This function assumes that the 'if' token has not yet been consumed.
    fn parse_if_statement(&mut self) -> Result<Stmt> {
        let if_token = self.consume_or_error(TokenType::If)?;
        self.consume_or_error(TokenType::LeftParanthesis)?;
        let condition = Box::new(self.parse_expression()?);
        self.consume_or_error(TokenType::RightParanthesis)?;
        let if_statement = Box::new(self.parse_statement()?);

        let mut else_statement: Option<Box<Stmt>> = None;
        let mut loc = LocationSpan::new(if_token.location, if_statement.get_loc().end_inclusive);
        if self.consume_if_has_token_type(TokenType::Else).is_some() {
            else_statement = Some(Box::new(self.parse_statement()?));
            loc.end_inclusive = else_statement.as_ref().unwrap().get_loc().end_inclusive;
        }

        Ok(Stmt::new(
            StmtData::If {
                condition,
                if_statement,
                else_statement,
            },
            loc,
        ))
    }

    /// Parses a While statement.
    ///
    /// Grammar rule:
    ///
    ///   while_statement: 'while '(' expression ')' statement
    ///
    /// This function assumes that the 'while' token has not yet been consumed.s
    fn parse_while_statement(&mut self) -> Result<Stmt> {
        let while_token = self.consume_or_error(TokenType::While)?;
        self.consume_or_error(TokenType::LeftParanthesis)?;
        let condition = Box::new(self.parse_expression()?);
        self.consume_or_error(TokenType::RightParanthesis)?;
        let body = Box::new(self.parse_statement()?);

        let loc = LocationSpan::new(while_token.location, body.get_loc().end_inclusive);
        Ok(Stmt::new(StmtData::While { condition, body }, loc))
    }

    /// Parses an expression (lowest precedence -> highest in AST (sub)hierarchy).
    ///
    /// Grammar rule:
    ///
    ///   expression: equality
    fn parse_expression(&mut self) -> Result<Expr> {
        Ok(self.parse_assignment()?)
    }

    /// Parses an assignment expression.
    ///
    /// Grammer rule:
    ///   assignment: 'identifier' '=' assignment
    ///             | equality
    fn parse_assignment(&mut self) -> Result<Expr> {
        let lvalue_expr = self.parse_equality()?;
        if let Some(_) = self.consume_if_has_token_type(TokenType::Equal) {
            let rvalue_expr = self.parse_assignment()?;

            if let ExprData::Variable { name, .. } = &lvalue_expr.get_data() {
                let loc = LocationSpan::new(
                    lvalue_expr.get_loc().start,
                    rvalue_expr.get_loc().end_inclusive,
                );
                Ok(Expr::new(
                    ExprData::Assign {
                        name: name.clone(),
                        expr: Box::new(rvalue_expr),
                    },
                    loc,
                ))
            } else {
                Err(emit_diagnostic(
                    format!("Invalid lvalue for assignment"),
                    FileLocation::Span(*lvalue_expr.get_loc()),
                    &self.source_file,
                ))
            }
        } else {
            Ok(lvalue_expr)
        }
    }

    /// Parses generic binary expressions.
    ///
    /// The token types valid for the binary expressions as well as the parsing function
    /// for the next entity (with higher precedence in the grammar) get passed into this function.
    ///
    /// Grammar rule:
    ///
    ///   <non-terminal>: [parse_fn -> Expr] ( <binary_op> [parse_fn -> Expr] )*
    fn parse_binary_expr(
        &mut self,
        token_types: &[TokenType],
        parse_fn: impl Fn(&mut Self) -> Result<Expr>,
    ) -> Result<Expr> {
        let mut expr = parse_fn(self)?;

        while let Some(token) = self.consume_if_has_token_types(&token_types) {
            let op = get_binary_operator_from_token_type(&token.token_type);

            let rhs = parse_fn(self)?;

            // The locations must get merged before the rhs expression gets moved into the new binary expression.
            let loc = LocationSpan::new(expr.get_loc().start, rhs.get_loc().end_inclusive);
            expr = Expr::new(
                ExprData::Binary {
                    lhs: Box::new(expr),
                    op,
                    rhs: Box::new(rhs),
                },
                loc,
            )
        }
        Ok(expr)
    }

    /// Parses an equality expression.
    ///
    /// Grammar rule:
    ///
    ///   equality: comparison ( ( '!=' | '==' ) comparison )*
    fn parse_equality(&mut self) -> Result<Expr> {
        let token_types = [TokenType::EqualEqual, TokenType::BangEqual];
        self.parse_binary_expr(&token_types, Self::parse_comparison)
    }

    /// Parses a comparison expression.
    ///
    /// Grammar rule:
    ///
    ///   comparison: term ( ( '>' | '>=' | '<' | '<=' ) term )*
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

    /// Parses a term expression.
    ///
    /// Grammar rule:
    ///
    ///   term: factor ( ( '-' | '+' ) factor )*
    fn parse_term(&mut self) -> Result<Expr> {
        let token_types = [TokenType::Plus, TokenType::Minus];
        self.parse_binary_expr(&token_types, Self::parse_factor)
    }

    /// Parses a factor expression.
    ///
    /// Grammar rule:
    ///
    ///   factor: unary ( ( '*' | '/' ) unary )*
    fn parse_factor(&mut self) -> Result<Expr> {
        let token_types = [TokenType::Star, TokenType::Slash];
        self.parse_binary_expr(&token_types, Self::parse_unary)
    }

    /// Parses a unary expression.
    ///
    /// Grammar rule:
    ///
    ///   unary: ( '!' | '-' ) unary
    ///        | primary
    fn parse_unary(&mut self) -> Result<Expr> {
        let token_types = [TokenType::Minus, TokenType::Bang];
        if let Some(token) = self.consume_if_has_token_types(&token_types) {
            let op = get_unary_operator_from_token_type(&token.token_type);

            let expr = self.parse_primary()?;
            // The location must get preserved before the expression gets moved into the new unary expression
            let loc = LocationSpan::new(token.location, expr.get_loc().end_inclusive);
            Ok(Expr::new(
                ExprData::Unary {
                    op,
                    expr: Box::new(expr),
                },
                loc,
            ))
        } else {
            self.parse_primary()
        }
    }

    fn create_expr_from_literal_and_token(literal: Literal, token: &Token) -> Result<Expr> {
        Ok(Expr::new(
            ExprData::Literal { literal },
            Self::create_location_span_from_token(token),
        ))
    }

    fn create_variable_from_token(token: &Token) -> Result<Expr> {
        Ok(Expr::new(
            ExprData::Variable {
                name: token.lexeme.clone(),
            },
            Self::create_location_span_from_token(token),
        ))
    }

    /// Parses a primary expression.
    ///
    /// Grammar rule:
    ///
    ///   primary: 'number'
    ///          | 'string'
    ///          | 'identifier'
    ///          | 'true'
    ///          | 'false'
    ///          | 'nil'
    ///          | '(' expression ')'
    fn parse_primary(&mut self) -> Result<Expr> {
        let token = self.consume();
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
            TokenType::Identifier => Self::create_variable_from_token(&token),
            TokenType::LeftParanthesis => {
                let expr = self.parse_expression()?;
                if let Some(closing_token) =
                    self.consume_if_has_token_type(TokenType::RightParanthesis)
                {
                    return Ok(Expr::new(
                        ExprData::Grouping {
                            expr: Box::new(expr),
                        },
                        LocationSpan::new(token.location, closing_token.location),
                    ));
                } else {
                    let cur_token = self.tokens.current();
                    return Err(emit_diagnostic(
                        format!(
                            "Expected closing paranthesis, but got: '{}'",
                            &cur_token.lexeme
                        ),
                        FileLocation::SinglePoint(token.location),
                        &self.source_file,
                    ));
                }
            }
            _ => {
                return Err(emit_diagnostic(
                    format!("Unexpected token: '{}'", token.lexeme),
                    FileLocation::SinglePoint(token.location),
                    &self.source_file,
                ))
            }
        };
        expr
    }
}

/// General Note: the tests for the parsers sub-divide into two categories:
///
///   o Full-Feature Tests: The generated AST from a token sequence gets fully tested including
///      location information (line, column).
///
///   o Structural Tests: The generated AST gets tested for structure only with the location information
///     getting ignored. To ease the testing, the equivalence of two ASTs, i.e. the generated and expected
///     ones are tested by using the AST `serializer` module.
///                   
#[cfg(test)]
mod tests {
    use core::panic;
    use std::path::PathBuf;

    use super::Parser;
    use crate::{
        ast::{
            serializer::tests::serialize_expr,
            tests::{new_grouping_expr, new_number_literal_expr},
            BinaryOperator, ExprData, UnaryOperator,
        },
        diagnostics::{Location, LocationSpan},
        scanner::{Token, TokenType},
    };
    use anyhow::Result;
    use colored::Colorize;

    use crate::ast::{Expr, Literal, Stmt};

    fn build_token_sequence(token_types: Vec<TokenType>) -> Vec<Token> {
        let mut loc = Location::new(1, 1);
        let mut tokens = Vec::<Token>::new();
        for token_type in token_types {
            let lexeme = match token_type {
                TokenType::Number => String::from("1.0"),
                TokenType::Identifier => String::from("id"),
                TokenType::StringLiteral => String::from("value"),
                _ => token_type.to_string(),
            };
            tokens.push(Token::new(token_type, loc, lexeme));
            loc.line += 1;
        }
        tokens.push(Token::new(TokenType::EndOfFile, loc, String::default()));
        tokens
    }

    macro_rules! token_seq {
        ($($token:expr),*) => {
            build_token_sequence(vec![$($token),*])
        }
    }

    #[test]
    fn test_build_token_sequence() {
        let tokens = token_seq![TokenType::And, TokenType::Identifier];
        assert_eq!(tokens.len(), 3);
        assert_eq!(
            tokens[0],
            Token::new(TokenType::And, Location::new(1, 1), String::from("and"))
        );
        assert_eq!(
            tokens[1],
            Token::new(
                TokenType::Identifier,
                Location::new(2, 1),
                String::from("id")
            )
        );
        assert_eq!(
            tokens[2],
            Token::new(TokenType::EndOfFile, Location::new(3, 1), String::default())
        );
    }

    fn new_parser(tokens: Vec<Token>) -> Parser {
        Parser::new(tokens, PathBuf::from("test"))
    }

    //     fn parse_stmt(tokens: Vec<Token>) -> Result<Stmt> {
    //         let mut parser = Parser::new(tokens, "in-memory".into());
    //         parser.parse_declaration()
    //     }

    ///////////////////////////////////
    /// Tests for Utility Functions ///
    ///////////////////////////////////

    #[test]
    fn test_consume() {
        let mut parser = new_parser(token_seq!(TokenType::And));

        let token = parser.consume();
        assert_eq!(token.token_type, TokenType::And);
        assert_eq!(token.location, Location::new(1, 1));
        assert_eq!(token.lexeme, String::from("and"));

        let eof = parser.consume();
        assert_eq!(eof.token_type, TokenType::EndOfFile);
    }

    #[test]
    fn test_consume_if_has_token_type() {
        let mut parser = new_parser(token_seq!(TokenType::And));

        assert!(parser.consume_if_has_token_type(TokenType::Bang).is_none());
        let token = parser
            .consume_if_has_token_types(&[TokenType::And])
            .unwrap();
        assert_eq!(token.token_type, TokenType::And);
        assert_eq!(token.location, Location::new(1, 1));
        assert_eq!(token.lexeme, String::from("and"));
    }

    #[test]
    fn test_consume_or_error() {
        let mut parser = new_parser(token_seq!(TokenType::And));

        let result = parser.consume_or_error(TokenType::Bang);
        assert!(result.is_err());

        let token = parser.consume_or_error(TokenType::And).unwrap();
        assert_eq!(token.token_type, TokenType::And);
        assert_eq!(token.location, Location::new(1, 1));
        assert_eq!(token.lexeme, String::from("and"));
    }

    ////////////////////////////////////////
    /// Structural Tests for Expressions ///
    ////////////////////////////////////////

    fn parse_expr(tokens: Vec<Token>) -> Result<Expr> {
        let mut parser = new_parser(tokens);
        parser.parse_expression()
    }

    fn parse_expr_and_check_literal(tokens: Vec<Token>, expected_literal: Literal) {
        let ast = parse_expr(tokens);
        assert!(ast.is_ok());

        if let ExprData::Literal { literal } = ast.unwrap().get_data() {
            assert_eq!(*literal, expected_literal);
        } else {
            panic!("Invalid expression type returned")
        }
    }

    fn print_diff(lhs: &Vec<String>, rhs: &Vec<String>) {
        assert!(lhs.len() > 0);

        const EXTRA_PADDING: usize = 3;
        // The maximum line length of the `lhs` input is chosen to appropriately pad the `rhs` input.
        let max_line_length = lhs.iter().map(|line| line.len()).max().unwrap() + EXTRA_PADDING;
        // Closure to generate the padding
        let gen_pad = |indent: usize| {
            std::iter::repeat(" ")
                .take(indent as usize)
                .collect::<String>()
        };
        // Closure to generate the status of the line comparison
        let gen_status = |lhs: &str, rhs: &str| {
            if lhs == rhs {
                String::from("[O]").green()
            } else {
                String::from("[F]").red()
            }
        };
        println!("Diff for AST:");
        println!("=============");
        for i in 0..lhs.len() {
            println!(
                "[line {:02}]   {}{}{}{}{}",
                i,
                lhs[i],
                gen_pad(max_line_length - lhs[i].len()),
                gen_status(&lhs[i], &rhs[i]),
                gen_pad(EXTRA_PADDING),
                rhs[i],
            );
        }
        println!("=============");
    }

    fn assert_serialized_output(lhs: Vec<String>, rhs: Vec<String>) {
        let mut panic_message: Option<String> = None;
        if lhs.len() != rhs.len() {
            panic_message = Some(String::from("Lengths of serialized output do not match"));
        } else {
            for i in 0..lhs.len() {
                if lhs[i] != rhs[i] {
                    panic_message = Some(format!(
                        "Content of serialized output does not match at line: {}",
                        i
                    ));
                }
            }
        }
        if let Some(message) = panic_message {
            print_diff(&lhs, &rhs);
            panic!("Test for serialized output failed: {}", message);
        }
    }

    fn parse_expr_and_check(tokens: Vec<Token>, expected_expr_ast: Expr) {
        let ast = parse_expr(tokens).unwrap();

        let ast_serialized = serialize_expr(&ast);
        let expected_ast_serialized = serialize_expr(&expected_expr_ast);

        assert_serialized_output(ast_serialized, expected_ast_serialized);
    }

    #[test]
    fn test_primary_expr() {
        let test_data = vec![
            (token_seq!(TokenType::Number), Literal::Number(1.0)),
            (
                token_seq!(TokenType::StringLiteral),
                Literal::String(String::from("value")),
            ),
            (token_seq!(TokenType::True), Literal::Boolean(true)),
            (token_seq!(TokenType::False), Literal::Boolean(false)),
            (token_seq!(TokenType::Nil), Literal::Nil),
        ];

        for (seq, expected) in test_data {
            parse_expr_and_check_literal(seq, expected);
        }
    }

    #[test]
    fn test_grouping_expr() {
        parse_expr_and_check(
            token_seq!(
                TokenType::LeftParanthesis,
                TokenType::Number,
                TokenType::RightParanthesis
            ),
            new_grouping_expr(new_number_literal_expr(0)),
        )
    }

    //     #[test]
    //     fn test_primary_expr_from_identifier() {
    //         let tokens = build_token_sequence(vec![TokenType::Identifier]);
    //         let expected_ast = Expr::Variable {
    //             name: String::from("identifier"),
    //             loc: loc_span((1, 1), (1, 10)),
    //         };

    //         let ast = parse_expr(tokens).unwrap();
    //         assert_eq!(ast, expected_ast);
    //     }

    //     #[test]
    //     fn test_binary_expr() {
    //         let test_data = vec![
    //             (TokenType::Plus, BinaryOperator::Add),
    //             (TokenType::Minus, BinaryOperator::Substract),
    //             (TokenType::Star, BinaryOperator::Multiply),
    //             (TokenType::Slash, BinaryOperator::Divide),
    //             (TokenType::Greater, BinaryOperator::GreaterThan),
    //             (
    //                 TokenType::GreaterOrEqual,
    //                 BinaryOperator::GreaterThanOrEqual,
    //             ),
    //             (TokenType::Less, BinaryOperator::LessThan),
    //             (TokenType::LessOrEqual, BinaryOperator::LessThanOrEqual),
    //             (TokenType::EqualEqual, BinaryOperator::Equal),
    //             (TokenType::BangEqual, BinaryOperator::NotEqual),
    //         ];
    //         for (token_type, binary_op) in test_data {
    //             let tokens =
    //                 build_token_sequence(vec![TokenType::Number, token_type, TokenType::Number]);
    //             let expected_ast = Expr::Binary {
    //                 lhs: Box::new(Expr::Literal {
    //                     literal: Literal::Number(0.0),
    //                     loc: loc_span((1, 1), (1, 3)),
    //                 }),
    //                 op: binary_op,
    //                 rhs: Box::new(Expr::Literal {
    //                     literal: Literal::Number(0.0),
    //                     loc: loc_span((3, 1), (3, 3)),
    //                 }),
    //                 loc: loc_span((1, 1), (3, 3)),
    //             };
    //             let ast = parse_expr(tokens).unwrap();
    //             assert_eq!(ast, expected_ast);
    //         }
    //     }

    //     #[test]
    //     fn test_unary_expr() {
    //         let test_data = vec![
    //             (TokenType::Minus, UnaryOperator::Minus),
    //             (TokenType::Bang, UnaryOperator::Not),
    //         ];
    //         for (token_type, unary_op) in test_data {
    //             let tokens = build_token_sequence(vec![token_type, TokenType::Number]);
    //             let expected_ast = Expr::Unary {
    //                 op: unary_op,
    //                 expr: Box::new(Expr::Literal {
    //                     literal: Literal::Number(0.0),
    //                     loc: loc_span((2, 1), (2, 3)),
    //                 }),
    //                 loc: loc_span((1, 1), (2, 3)),
    //             };
    //             let ast: Expr = parse_expr(tokens).unwrap();
    //             assert_eq!(ast, expected_ast);
    //         }
    //     }

    //     #[test]
    //     fn test_assignment() {
    //         let tokens = build_token_sequence(vec![
    //             TokenType::Identifier,
    //             TokenType::Equal,
    //             TokenType::StringLiteral,
    //         ]);
    //         let expected_ast = Expr::Assign {
    //             name: String::from("identifier"),
    //             expr: Box::new(Expr::Literal {
    //                 literal: Literal::String("string-literal".into()),
    //                 loc: loc_span((3, 1), (3, 14)),
    //             }),
    //             loc: loc_span((1, 1), (3, 14)),
    //         };
    //         let ast = parse_expr(tokens).unwrap();
    //         assert_eq!(ast, expected_ast);
    //     }

    //     #[test]
    //     fn test_assignment_error() {
    //         let tokens = build_token_sequence(vec![
    //             TokenType::Identifier,
    //             TokenType::Plus,
    //             TokenType::Number,
    //             TokenType::Equal,
    //             TokenType::StringLiteral,
    //         ]);
    //         assert!(parse_expr(tokens).is_err());
    //     }

    //     #[test]
    //     fn test_variable_declaration_statement_no_initializer() {
    //         let tokens = build_token_sequence(vec![
    //             TokenType::Var,
    //             TokenType::Identifier,
    //             TokenType::Semicolon,
    //         ]);
    //         let expected_ast = Stmt::VarDecl {
    //             identifier: String::from("identifier"),
    //             init_expr: Box::new(Expr::Literal {
    //                 literal: Literal::Nil,
    //                 loc: loc_span((2, 1), (2, 10)),
    //             }),
    //             loc: loc_span((1, 1), (3, 1)),
    //         };
    //         let ast = parse_stmt(tokens).unwrap();
    //         assert_eq!(ast, expected_ast);
    //     }

    //     #[test]
    //     fn test_variable_declaration_statement_with_initializer() {
    //         let tokens = build_token_sequence(vec![
    //             TokenType::Var,
    //             TokenType::Identifier,
    //             TokenType::Equal,
    //             TokenType::Number,
    //             TokenType::Semicolon,
    //         ]);
    //         let expected_ast = Stmt::VarDecl {
    //             identifier: String::from("identifier"),
    //             init_expr: Box::new(Expr::Literal {
    //                 literal: Literal::Number(0.0),
    //                 loc: loc_span((4, 1), (4, 3)),
    //             }),
    //             loc: loc_span((1, 1), (5, 1)),
    //         };
    //         let ast = parse_stmt(tokens).unwrap();
    //         assert_eq!(ast, expected_ast);
    //     }

    //     #[test]
    //     fn test_variable_declaration_errors() {
    //         let test_data = vec![
    //             build_token_sequence(vec![TokenType::Var, TokenType::Identifier]),
    //             build_token_sequence(vec![
    //                 TokenType::Var,
    //                 TokenType::Identifier,
    //                 TokenType::Equal,
    //                 TokenType::Semicolon,
    //             ]),
    //         ];
    //         for tokens in test_data {
    //             assert!(parse_stmt(tokens).is_err());
    //         }
    //     }

    //     #[test]
    //     fn test_print_statement() {
    //         let tokens = build_token_sequence(vec![
    //             TokenType::Print,
    //             TokenType::Number,
    //             TokenType::Semicolon,
    //         ]);
    //         let expected_ast = Stmt::Print {
    //             expr: Box::new(Expr::Literal {
    //                 literal: Literal::Number(0.0),
    //                 loc: loc_span((2, 1), (2, 3)),
    //             }),
    //             loc: loc_span((1, 1), (3, 1)),
    //         };
    //         let ast = parse_stmt(tokens).unwrap();
    //         assert_eq!(ast, expected_ast);
    //     }

    //     #[test]
    //     fn test_print_statement_errors() {
    //         let test_data = vec![
    //             build_token_sequence(vec![TokenType::Print, TokenType::Semicolon]),
    //             build_token_sequence(vec![TokenType::Print, TokenType::Number]),
    //         ];
    //         for tokens in test_data {
    //             assert!(parse_stmt(tokens).is_err());
    //         }
    //     }

    //     #[test]
    //     fn test_expr_statement() {
    //         let tokens = build_token_sequence(vec![TokenType::Number, TokenType::Semicolon]);
    //         let expected_ast = Stmt::Expr {
    //             expr: Box::new(Expr::Literal {
    //                 literal: Literal::Number(0.0),
    //                 loc: loc_span((1, 1), (1, 3)),
    //             }),
    //             loc: loc_span((1, 1), (2, 1)),
    //         };
    //         let ast = parse_stmt(tokens).unwrap();
    //         assert_eq!(ast, expected_ast);
    //     }

    //     #[test]
    //     fn test_block_statement() {
    //         let tokens = build_token_sequence(vec![
    //             TokenType::LeftBrace,
    //             TokenType::Print,
    //             TokenType::Number,
    //             TokenType::Semicolon,
    //             TokenType::RightBrace,
    //         ]);
    //         let expected_ast = Stmt::Block {
    //             statements: vec![Stmt::Print {
    //                 expr: Box::new(Expr::Literal {
    //                     literal: Literal::Number(0.0),
    //                     loc: loc_span((3, 1), (3, 3)),
    //                 }),
    //                 loc: loc_span((2, 1), (4, 1)),
    //             }],
    //             loc: loc_span((1, 1), (5, 1)),
    //         };
    //         let ast = parse_stmt(tokens).unwrap();
    //         assert_eq!(ast, expected_ast);
    //     }

    //     #[test]
    //     fn test_block_statement_with_inner_variable_declaration() {
    //         let tokens = build_token_sequence(vec![
    //             TokenType::LeftBrace,
    //             TokenType::Var,
    //             TokenType::Identifier,
    //             TokenType::Equal,
    //             TokenType::Number,
    //             TokenType::Semicolon,
    //             TokenType::RightBrace,
    //         ]);
    //         let expected_ast = Stmt::Block {
    //             statements: vec![Stmt::VarDecl {
    //                 identifier: String::from("identifier"),
    //                 init_expr: Box::new(Expr::Literal {
    //                     literal: Literal::Number(0.0),
    //                     loc: loc_span((5, 1), (5, 3)),
    //                 }),
    //                 loc: loc_span((2, 1), (6, 1)),
    //             }],
    //             loc: loc_span((1, 1), (7, 1)),
    //         };
    //         let ast = parse_stmt(tokens).unwrap();
    //         assert_eq!(ast, expected_ast);
    //     }

    //     #[test]
    //     fn test_if_statement_with_if_only() {
    //         let tokens = build_token_sequence(vec![
    //             TokenType::If,
    //             TokenType::LeftParanthesis,
    //             TokenType::Number,
    //             TokenType::RightParanthesis,
    //             TokenType::Identifier,
    //             TokenType::Equal,
    //             TokenType::Number,
    //             TokenType::Semicolon,
    //         ]);
    //         let expected_ast = Stmt::If {
    //             condition: Box::new(Expr::Literal {
    //                 literal: Literal::Number(0.0),
    //                 loc: loc_span((3, 1), (3, 3)),
    //             }),
    //             if_statement: Box::new(Stmt::Expr {
    //                 expr: Box::new(Expr::Assign {
    //                     name: "identifier".into(),
    //                     expr: Box::new(Expr::Literal {
    //                         literal: Literal::Number(0.0),
    //                         loc: loc_span((7, 1), (7, 3)),
    //                     }),
    //                     loc: loc_span((5, 1), (7, 3)),
    //                 }),
    //                 loc: loc_span((5, 1), (8, 1)),
    //             }),
    //             else_statement: None,
    //             loc: loc_span((1, 1), (8, 1)),
    //         };
    //         let ast = parse_stmt(tokens).unwrap();
    //         assert_eq!(ast, expected_ast);
    //     }

    //     #[test]
    //     fn test_if_statement_with_if_and_else() {
    //         let tokens = build_token_sequence(vec![
    //             TokenType::If,
    //             TokenType::LeftParanthesis,
    //             TokenType::Number,
    //             TokenType::RightParanthesis,
    //             TokenType::Identifier,
    //             TokenType::Equal,
    //             TokenType::Number,
    //             TokenType::Semicolon,
    //             TokenType::Else,
    //             TokenType::Identifier,
    //             TokenType::Equal,
    //             TokenType::Nil,
    //             TokenType::Semicolon,
    //         ]);
    //         let expected_ast = Stmt::If {
    //             condition: Box::new(Expr::Literal {
    //                 literal: Literal::Number(0.0),
    //                 loc: loc_span((3, 1), (3, 3)),
    //             }),
    //             if_statement: Box::new(Stmt::Expr {
    //                 expr: Box::new(Expr::Assign {
    //                     name: "identifier".into(),
    //                     expr: Box::new(Expr::Literal {
    //                         literal: Literal::Number(0.0),
    //                         loc: loc_span((7, 1), (7, 3)),
    //                     }),
    //                     loc: loc_span((5, 1), (7, 3)),
    //                 }),
    //                 loc: loc_span((5, 1), (8, 1)),
    //             }),
    //             else_statement: Some(Box::new(Stmt::Expr {
    //                 expr: Box::new(Expr::Assign {
    //                     name: "identifier".into(),
    //                     expr: Box::new(Expr::Literal {
    //                         literal: Literal::Nil,
    //                         loc: loc_span((12, 1), (12, 3)),
    //                     }),
    //                     loc: loc_span((10, 1), (12, 3)),
    //                 }),
    //                 loc: loc_span((10, 1), (13, 1)),
    //             })),
    //             loc: loc_span((1, 1), (13, 1)),
    //         };
    //         let ast = parse_stmt(tokens).unwrap();
    //         assert_eq!(ast, expected_ast);
    //     }

    //     #[test]
    //     fn test_while_statment() {
    //         let tokens = build_token_sequence(vec![
    //             TokenType::While,
    //             TokenType::LeftParanthesis,
    //             TokenType::Number,
    //             TokenType::RightParanthesis,
    //             TokenType::Identifier,
    //             TokenType::Equal,
    //             TokenType::Number,
    //             TokenType::Semicolon,
    //         ]);
    //         let expected_ast = Stmt::While {
    //             condition: Box::new(Expr::Literal {
    //                 literal: Literal::Number(0.0),
    //                 loc: loc_span((3, 1), (3, 3)),
    //             }),
    //             body: Box::new(Stmt::Expr {
    //                 expr: Box::new(Expr::Assign {
    //                     name: "identifier".into(),
    //                     expr: Box::new(Expr::Literal {
    //                         literal: Literal::Number(0.0),
    //                         loc: loc_span((7, 1), (7, 3)),
    //                     }),
    //                     loc: loc_span((5, 1), (7, 3)),
    //                 }),
    //                 loc: loc_span((5, 1), (8, 1)),
    //             }),
    //             loc: loc_span((1, 1), (8, 1)),
    //         };
    //         let ast = parse_stmt(tokens).unwrap();
    //         assert_eq!(ast, expected_ast);
    //     }
}
