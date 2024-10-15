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
use anyhow::{Ok, Result};

use token_sequence::TokenSequence;

/// Converts a sequence of tokens into an Abstract Syntax Tree (AST).
pub fn parse(tokens: Vec<Token>, source_file: PathBuf) -> Result<Stmt> {
    let parser = Parser::new(tokens, source_file);
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

    fn current_has_token_type(&self, token_type: TokenType) -> bool {
        return self.tokens.current_has_token_type(&vec![token_type]);
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
                expr: expr.as_box(),
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
                expr: expr.as_box(),
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
        let condition = self.parse_expression()?.as_box();
        self.consume_or_error(TokenType::RightParanthesis)?;
        let if_statement = self.parse_statement()?.as_box();

        let mut else_statement: Option<Box<Stmt>> = None;
        let mut loc = LocationSpan::new(if_token.location, if_statement.get_loc().end_inclusive);
        if self.consume_if_has_token_type(TokenType::Else).is_some() {
            else_statement = Some(self.parse_statement()?.as_box());
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
        let condition = self.parse_expression()?.as_box();
        self.consume_or_error(TokenType::RightParanthesis)?;
        let body = self.parse_statement()?.as_box();

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
                        expr: rvalue_expr.as_box(),
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
                    lhs: expr.as_box(),
                    op,
                    rhs: rhs.as_box(),
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
    ///        | call
    fn parse_unary(&mut self) -> Result<Expr> {
        let token_types = [TokenType::Minus, TokenType::Bang];
        if let Some(token) = self.consume_if_has_token_types(&token_types) {
            let op = get_unary_operator_from_token_type(&token.token_type);

            let expr = self.parse_unary()?;
            // The location must get preserved before the expression gets moved into the new unary expression
            let loc = LocationSpan::new(token.location, expr.get_loc().end_inclusive);
            Ok(Expr::new(
                ExprData::Unary {
                    op,
                    expr: expr.as_box(),
                },
                loc,
            ))
        } else {
            self.parse_call_expression()
        }
    }

    /// Parses a call expression.
    ///
    /// Grammar rule:
    ///
    ///   call: primary ( '(' arguments? ')' )*
    fn parse_call_expression(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_primary()?;

        while let Some(_) = self.consume_if_has_token_type(TokenType::LeftParanthesis) {
            let loc = lhs.get_loc().to_owned();
            lhs = self.finish_call_expression(lhs, loc.start)?;
        }

        Ok(lhs)
    }

    /// Parses the arguments and the closing right parenthesis of a function call expression.
    ///
    /// This function assumes that the opening paranthesis has already been parsed.
    fn finish_call_expression(&mut self, callee: Expr, start_loc: Location) -> Result<Expr> {
        let mut arguments = Vec::<Expr>::new();

        if !self.current_has_token_type(TokenType::RightParanthesis) {
            loop {
                arguments.push(self.parse_expression()?);
                if !self.current_has_token_type(TokenType::Comma) {
                    break;
                }
                self.consume_or_error(TokenType::Comma)?;
            }
        }

        let closing_token = self.consume_or_error(TokenType::RightParanthesis)?;

        let loc = LocationSpan::new(start_loc, closing_token.location);
        Ok(Expr::new(
            ExprData::Call {
                callee: callee.as_box(),
                arguments,
            },
            loc,
        ))
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
                            expr: expr.as_box(),
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
    use std::path::PathBuf;

    use super::Parser;
    use crate::{
        ast::{
            serializer::tests::{serialize_expr, serialize_stmt},
            tests::{
                new_assign_expr, new_binary_expr, new_block_stmt, new_expr_stmt,
                new_function_call_expr, new_grouping_expr, new_if_else_stmt, new_if_stmt,
                new_literal_expr, new_number_literal_expr, new_print_stmt, new_string_literal_expr,
                new_unary_expr, new_var_decl_stmt, new_variable_expr, new_while_stmt,
            },
            BinaryOperator, ExprData, StmtData, UnaryOperator,
        },
        diagnostics::{Location, LocationSpan},
        scanner::{Token, TokenType},
    };
    use anyhow::Result;

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
        ($($token:expr),* $(,)?) => {
            build_token_sequence(vec![$($token),*])
        }
    }

    #[test]
    fn test_build_token_sequence() {
        let tokens = token_seq![TokenType::And, TokenType::Identifier,];
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

    fn parse_decl(tokens: Vec<Token>) -> Result<Stmt> {
        let mut parser = new_parser(tokens);
        parser.parse_declaration()
    }

    /// Macro to parse an statement and check for an expected `Stmt` AST.
    macro_rules! parse_decl_and_check {
        ($arg1:expr, $arg2:expr) => {
            let ast = parse_decl($arg1).unwrap();

            let ast_serialized = serialize_stmt(&ast);
            let expected_ast_serialized = serialize_stmt(&$arg2);

            pretty_assertions::assert_eq!(ast_serialized, expected_ast_serialized);
        };
    }

    fn parse_expr(tokens: Vec<Token>) -> Result<Expr> {
        let mut parser = new_parser(tokens);
        parser.parse_expression()
    }

    /// Macro to parse an expression and check for an expected `Expr` AST.
    macro_rules! parse_expr_and_check {
        ($arg1:expr, $arg2:expr) => {
            let ast = parse_expr($arg1).unwrap();

            let ast_serialized = serialize_expr(&ast);
            let expected_ast_serialized = serialize_expr(&$arg2);

            pretty_assertions::assert_eq!(ast_serialized, expected_ast_serialized);
        };
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
            let ast = parse_expr(seq).unwrap();
            assert_eq!(ast.get_data(), &ExprData::Literal { literal: expected });
        }
    }

    #[test]
    fn test_grouping_expr() {
        parse_expr_and_check!(
            token_seq!(
                TokenType::LeftParanthesis,
                TokenType::Number,
                TokenType::RightParanthesis,
            ),
            new_grouping_expr(new_number_literal_expr(1))
        );
    }

    #[test]
    fn test_primary_expr_from_identifier() {
        parse_expr_and_check!(token_seq!(TokenType::Identifier), new_variable_expr("id"));
    }

    #[test]
    fn test_binary_expr() {
        let test_data = vec![
            (TokenType::Plus, BinaryOperator::Add),
            (TokenType::Minus, BinaryOperator::Substract),
            (TokenType::Star, BinaryOperator::Multiply),
            (TokenType::Slash, BinaryOperator::Divide),
            (TokenType::Greater, BinaryOperator::GreaterThan),
            (
                TokenType::GreaterOrEqual,
                BinaryOperator::GreaterThanOrEqual,
            ),
            (TokenType::Less, BinaryOperator::LessThan),
            (TokenType::LessOrEqual, BinaryOperator::LessThanOrEqual),
            (TokenType::EqualEqual, BinaryOperator::Equal),
            (TokenType::BangEqual, BinaryOperator::NotEqual),
        ];
        for (token_type, binary_op) in test_data {
            parse_expr_and_check!(
                token_seq!(TokenType::Number, token_type, TokenType::Number),
                new_binary_expr(
                    binary_op,
                    new_number_literal_expr(1),
                    new_number_literal_expr(1)
                )
            );
        }
    }

    #[test]
    fn test_unary_expr() {
        let test_data = vec![
            (TokenType::Minus, UnaryOperator::Minus),
            (TokenType::Bang, UnaryOperator::Not),
        ];
        for (token_type, unary_op) in test_data {
            parse_expr_and_check!(
                token_seq!(token_type, TokenType::Number),
                new_unary_expr(unary_op, new_number_literal_expr(1))
            );
        }
    }

    #[test]
    fn test_assignment() {
        parse_expr_and_check!(
            token_seq!(
                TokenType::Identifier,
                TokenType::Equal,
                TokenType::StringLiteral,
            ),
            new_assign_expr("id", new_string_literal_expr("value"))
        );
    }

    #[test]
    fn test_assignment_error() {
        let tokens = token_seq!(
            TokenType::Identifier,
            TokenType::Plus,
            TokenType::Number,
            TokenType::Equal,
            TokenType::StringLiteral,
        );
        assert!(parse_expr(tokens).is_err());
    }

    #[test]
    fn test_function_call_expr_no_arguments() {
        parse_expr_and_check!(
            token_seq!(
                TokenType::Identifier,
                TokenType::LeftParanthesis,
                TokenType::RightParanthesis,
            ),
            new_function_call_expr(new_variable_expr("id"), vec![])
        );
    }

    #[test]
    fn test_function_call_expr_with_arguments() {
        parse_expr_and_check!(
            token_seq!(
                TokenType::Identifier,
                TokenType::LeftParanthesis,
                TokenType::Number,
                TokenType::Comma,
                TokenType::StringLiteral,
                TokenType::Comma,
                TokenType::Nil,
                TokenType::RightParanthesis,
            ),
            new_function_call_expr(
                new_variable_expr("id"),
                vec![
                    new_number_literal_expr(1.0),
                    new_string_literal_expr("value"),
                    new_literal_expr(Literal::Nil),
                ]
            )
        );
    }

    #[test]
    fn test_function_call_expr_with_nested_call_expr() {
        parse_expr_and_check!(
            token_seq!(
                TokenType::Identifier,
                TokenType::LeftParanthesis,
                TokenType::Identifier,
                TokenType::LeftParanthesis,
                TokenType::RightParanthesis,
                TokenType::RightParanthesis,
            ),
            new_function_call_expr(
                new_variable_expr("id"),
                vec![new_function_call_expr(new_variable_expr("id"), vec![])]
            )
        );
    }

    ///////////////////////////////////////
    /// Structural Tests for Statements ///
    ///////////////////////////////////////
    ///
    /// Note: the tests here are white-box tests that directly test the parsing on the grammar-level of
    /// declarations instead of the whole program. The reason is that using the `parse` or `parse_program`
    /// function would embed the resulting AST into a `StmtData::List` enum type which is not wanted for these
    /// finer-grained tests.

    #[test]
    fn test_variable_declaration_statement_no_initializer() {
        parse_decl_and_check!(
            token_seq!(TokenType::Var, TokenType::Identifier, TokenType::Semicolon),
            new_var_decl_stmt("id", new_literal_expr(Literal::Nil))
        );
    }

    #[test]
    fn test_variable_declaration_statement_with_initializer() {
        parse_decl_and_check!(
            token_seq!(
                TokenType::Var,
                TokenType::Identifier,
                TokenType::Equal,
                TokenType::Number,
                TokenType::Semicolon,
            ),
            new_var_decl_stmt("id", new_number_literal_expr(1))
        );
    }

    #[test]
    fn test_variable_declaration_errors() {
        let test_data = vec![
            token_seq!(TokenType::Var, TokenType::Identifier),
            token_seq!(
                TokenType::Var,
                TokenType::Identifier,
                TokenType::Equal,
                TokenType::Semicolon,
            ),
        ];
        for tokens in test_data {
            assert!(parse_decl(tokens).is_err());
        }
    }

    #[test]
    fn test_print_statement() {
        parse_decl_and_check!(
            token_seq!(TokenType::Print, TokenType::Number, TokenType::Semicolon),
            new_print_stmt(new_number_literal_expr(1))
        );
    }

    #[test]
    fn test_print_statement_errors() {
        let test_data = vec![
            token_seq!(TokenType::Print, TokenType::Semicolon),
            token_seq!(TokenType::Print, TokenType::Number),
        ];
        for tokens in test_data {
            assert!(parse_decl(tokens).is_err());
        }
    }

    #[test]
    fn test_expr_statement() {
        parse_decl_and_check!(
            token_seq!(TokenType::Number, TokenType::Semicolon),
            new_expr_stmt(new_number_literal_expr(1))
        );
    }

    #[test]
    fn test_block_statement() {
        parse_decl_and_check!(
            token_seq!(
                TokenType::LeftBrace,
                TokenType::Print,
                TokenType::Number,
                TokenType::Semicolon,
                TokenType::RightBrace,
            ),
            new_block_stmt(vec![new_print_stmt(new_number_literal_expr(1))])
        );
    }

    #[test]
    fn test_block_statement_with_inner_variable_declaration() {
        parse_decl_and_check!(
            token_seq!(
                TokenType::LeftBrace,
                TokenType::Var,
                TokenType::Identifier,
                TokenType::Equal,
                TokenType::Number,
                TokenType::Semicolon,
                TokenType::RightBrace,
            ),
            new_block_stmt(vec![new_var_decl_stmt("id", new_number_literal_expr(1))])
        );
    }

    #[test]
    fn test_if_statement_with_if_only() {
        parse_decl_and_check!(
            token_seq!(
                TokenType::If,
                TokenType::LeftParanthesis,
                TokenType::Number,
                TokenType::RightParanthesis,
                TokenType::Identifier,
                TokenType::Equal,
                TokenType::Number,
                TokenType::Semicolon,
            ),
            new_if_stmt(
                new_number_literal_expr(1),
                new_expr_stmt(new_assign_expr("id", new_number_literal_expr(1)))
            )
        );
    }

    #[test]
    fn test_if_statement_with_if_and_else() {
        parse_decl_and_check!(
            token_seq!(
                TokenType::If,
                TokenType::LeftParanthesis,
                TokenType::Number,
                TokenType::RightParanthesis,
                TokenType::Identifier,
                TokenType::Equal,
                TokenType::Number,
                TokenType::Semicolon,
                TokenType::Else,
                TokenType::Identifier,
                TokenType::Equal,
                TokenType::Nil,
                TokenType::Semicolon,
            ),
            new_if_else_stmt(
                new_number_literal_expr(1),
                new_expr_stmt(new_assign_expr("id", new_number_literal_expr(1))),
                new_expr_stmt(new_assign_expr("id", new_literal_expr(Literal::Nil)))
            )
        );
    }

    #[test]
    fn test_while_statment() {
        parse_decl_and_check!(
            token_seq!(
                TokenType::While,
                TokenType::LeftParanthesis,
                TokenType::Number,
                TokenType::RightParanthesis,
                TokenType::Identifier,
                TokenType::Equal,
                TokenType::Number,
                TokenType::Semicolon,
            ),
            new_while_stmt(
                new_number_literal_expr(1),
                new_expr_stmt(new_assign_expr("id", new_number_literal_expr(1)))
            )
        );
    }

    fn loc_span(start: (i64, i64), end_inclusive: (i64, i64)) -> LocationSpan {
        LocationSpan {
            start: Location {
                line: start.0,
                column: start.1,
            },
            end_inclusive: Location {
                line: end_inclusive.0,
                column: end_inclusive.1,
            },
        }
    }

    #[test]
    fn test_ast_with_locations() {
        let tokens = token_seq!(
            TokenType::If,
            TokenType::LeftParanthesis,
            TokenType::Identifier,
            TokenType::RightParanthesis,
            TokenType::LeftBrace,
            TokenType::Identifier,
            TokenType::Equal,
            TokenType::Number,
            TokenType::Semicolon,
            TokenType::RightBrace,
        );
        let ast = parse_decl(tokens).unwrap();
        let expected_ast = Stmt::new(
            StmtData::If {
                condition: Expr::new(
                    ExprData::Variable {
                        name: String::from("id"),
                    },
                    loc_span((3, 1), (3, 2)),
                )
                .as_box(),
                if_statement: Stmt::new(
                    StmtData::Block {
                        statements: vec![Stmt::new(
                            StmtData::Expr {
                                expr: Expr::new(
                                    ExprData::Assign {
                                        name: String::from("id"),
                                        expr: Expr::new(
                                            ExprData::Literal {
                                                literal: Literal::Number(1.0),
                                            },
                                            loc_span((8, 1), (8, 3)),
                                        )
                                        .as_box(),
                                    },
                                    loc_span((6, 1), (8, 3)),
                                )
                                .as_box(),
                            },
                            loc_span((6, 1), (9, 1)),
                        )],
                    },
                    loc_span((5, 1), (10, 1)),
                )
                .as_box(),
                else_statement: None,
            },
            loc_span((1, 1), (10, 1)),
        );

        pretty_assertions::assert_eq!(ast, expected_ast);
    }
}
