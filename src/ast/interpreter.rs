// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

//! Evaluator of Abstract Syntax Tree (AST) nodes.
//!
//! This module provides functions to execute (and thereby) eval nodes
//! of the AST.

mod environment;

use std::{io::Write, path::PathBuf};

use anyhow::Result;
use environment::ExecutionEnvironment;

use crate::{
    ast::{Expr, Literal},
    diagnostics::{emit_diagnostic, DiagnosticError, FileLocation, LocationSpan},
};

use super::{BinaryOperator, ExprData, Stmt, StmtData, UnaryOperator};

/// Result of evaluating an expression node of the AST.
#[derive(PartialEq, Debug, Clone)]
pub enum ExprValue {
    Number(f64),
    String(String),
    Boolean(bool),
    Nil,
}

impl ExprValue {
    fn type_to_string(&self) -> String {
        match self {
            ExprValue::Boolean(_) => String::from("bool"),
            ExprValue::Nil => String::from("nil"),
            ExprValue::Number(_) => String::from("number"),
            ExprValue::String(_) => String::from("string"),
        }
    }
}

impl ToString for ExprValue {
    fn to_string(&self) -> String {
        match self {
            ExprValue::Boolean(value) => value.to_string(),
            ExprValue::Nil => String::from("nil"),
            ExprValue::Number(value) => value.to_string(),
            ExprValue::String(value) => format!("\"{}\"", value.to_string()),
        }
    }
}

/// Interprets the statement AST.
///
/// The output of the interpretation, e.g. from `print` statements get written to the `output_writer`.
pub fn interpret<'a, W: Write>(
    stmt: &Stmt,
    source_file: PathBuf,
    output_writer: Option<&'a mut W>,
) -> Result<()> {
    let mut interpreter = Interpreter::new(source_file, output_writer);
    interpreter.interpret(stmt)
}

/// Runtime interpreter for Lox.
pub struct Interpreter<'a, W: Write> {
    source_file: PathBuf,
    env: ExecutionEnvironment,
    output_writer: Option<&'a mut W>,
}

impl<'a, W: Write> Interpreter<'a, W> {
    pub fn new(source_file: PathBuf, output_writer: Option<&'a mut W>) -> Self {
        Self {
            source_file,
            env: ExecutionEnvironment::new(),
            output_writer,
        }
    }

    fn is_truthy(expr_value: &ExprValue) -> bool {
        match expr_value {
            ExprValue::Boolean(false) => false,
            ExprValue::Nil => false,
            _ => true,
        }
    }

    pub fn interpret(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt.get_data() {
            StmtData::List { statements, .. } => self.interpret_stmts(statements)?,
            StmtData::Block { statements, .. } => {
                // Errors when executing statements shall not be propagated as is, because
                // the scope must get cleaned up properly at the end of the block.
                self.env.create_scope();
                let result = self.interpret_stmts(statements);
                self.env.drop_innermost_scope();
                return result;
            }
            StmtData::VarDecl {
                identifier,
                init_expr,
                ..
            } => {
                let expr_value = self.interpret_expr(init_expr)?;
                self.env.define_variable(identifier, expr_value);
            }
            StmtData::Expr { expr, .. } => {
                self.interpret_expr(&expr)?;
            }
            StmtData::Print { expr, .. } => {
                let expr_value = self.interpret_expr(expr)?;
                if let Some(writer) = &mut self.output_writer {
                    writer.write_fmt(format_args!("{}\n", expr_value.to_string()))?;
                }
            }
            StmtData::If {
                condition,
                if_statement,
                else_statement,
                ..
            } => {
                let cond_value = self.interpret_expr(condition)?;
                if Self::is_truthy(&cond_value) {
                    self.interpret(if_statement)?;
                } else if let Some(else_statement) = else_statement {
                    self.interpret(&else_statement)?;
                }
            }
            StmtData::While {
                condition, body, ..
            } => {
                while Self::is_truthy(&self.interpret_expr(condition)?) {
                    self.interpret(body)?;
                }
            }
        }
        Ok(())
    }
    fn interpret_stmts(&mut self, statements: &Vec<Stmt>) -> Result<()> {
        for stmt in statements {
            self.interpret(stmt)?
        }
        Ok(())
    }

    /// Evaluates the `Expr` node of the AST into an `ExprValue`.
    fn interpret_expr(&mut self, expr: &Expr) -> Result<ExprValue> {
        let loc = expr.get_loc();
        match expr.get_data() {
            ExprData::Binary { lhs, op, rhs } => self.interpret_binary_expr(op, lhs, rhs, loc),
            ExprData::Grouping { expr, .. } => self.interpret_expr(expr),
            ExprData::Literal { literal, .. } => self.interpret_literal(&literal),
            ExprData::Unary { op, expr } => self.interpret_unary_expr(op, expr, loc),
            ExprData::Variable { name } => {
                if let Some(value) = self.env.get_variable(name) {
                    Ok(value.clone())
                } else {
                    Err(emit_diagnostic(
                        format!("Undefined variable '{}'", name),
                        FileLocation::Span(*loc),
                        &self.source_file,
                    ))
                }
            }
            ExprData::Assign { name, expr } => {
                let value = self.interpret_expr(expr)?;
                if self.env.assign_variable(name, value.clone()) {
                    Ok(value)
                } else {
                    Err(emit_diagnostic(
                        format!("Undefined variable '{}'", name),
                        FileLocation::Span(*loc),
                        &self.source_file,
                    ))
                }
            }
        }
    }

    fn interpret_literal(&self, literal: &Literal) -> Result<ExprValue> {
        Ok(match literal {
            Literal::Boolean(value) => ExprValue::Boolean(*value),
            Literal::Nil => ExprValue::Nil,
            Literal::Number(value) => ExprValue::Number(*value),
            Literal::String(value) => ExprValue::String(String::from(value)),
        })
    }

    fn interpret_unary_expr(
        &mut self,
        op: &UnaryOperator,
        expr: &Expr,
        loc: &LocationSpan,
    ) -> Result<ExprValue> {
        let result = self.interpret_expr(expr)?;
        match op {
            UnaryOperator::Minus => {
                if let ExprValue::Number(value) = &result {
                    Ok(ExprValue::Number(-value))
                } else {
                    Err(self.emit_eval_error(
                        format!(
                            "Unary operator '{}' only applicable to numbers: given {}",
                            op.to_string(),
                            result.type_to_string(),
                        ),
                        loc,
                    ))
                }
            }
            UnaryOperator::Not => {
                if let ExprValue::Boolean(value) = &result {
                    Ok(ExprValue::Boolean(!value))
                } else {
                    Err(self.emit_eval_error(
                        format!(
                            "Unary operator '{}' only applicable to boolean values, given: {}",
                            op.to_string(),
                            result.type_to_string(),
                        ),
                        loc,
                    ))
                }
            }
        }
    }

    fn interpret_binary_expr(
        &mut self,
        op: &BinaryOperator,
        lhs: &Expr,
        rhs: &Expr,
        loc: &LocationSpan,
    ) -> Result<ExprValue> {
        let lhs_value = self.interpret_expr(lhs)?;
        let rhs_value = self.interpret_expr(rhs)?;

        match lhs_value {
            ExprValue::Number(_) => {
                self.interpret_binary_op_for_number(op, &lhs_value, &rhs_value, loc)
            }
            ExprValue::String(_) => {
                self.interpret_binary_op_for_string(op, &lhs_value, &rhs_value, loc)
            }
            ExprValue::Nil => Err(self.emit_eval_error(
                format!("Expression of type 'nil' is not supported for binary operators"),
                loc,
            )),
            ExprValue::Boolean(_) => Err(self.emit_eval_error(
                format!("Expression of type 'bool' is not supported for binary operators"),
                loc,
            )),
        }
    }

    fn interpret_binary_op_for_number(
        &self,
        op: &BinaryOperator,
        lhs: &ExprValue,
        rhs: &ExprValue,
        loc: &LocationSpan,
    ) -> Result<ExprValue> {
        if let ExprValue::Number(lhs_value) = lhs {
            if let ExprValue::Number(rhs_value) = rhs {
                Ok(match op {
                    BinaryOperator::Add => ExprValue::Number(lhs_value + rhs_value),
                    BinaryOperator::Substract => ExprValue::Number(lhs_value - rhs_value),
                    BinaryOperator::Multiply => ExprValue::Number(lhs_value * rhs_value),
                    BinaryOperator::Divide => ExprValue::Number(lhs_value / rhs_value),
                    BinaryOperator::Equal => ExprValue::Boolean(lhs_value == rhs_value),
                    BinaryOperator::NotEqual => ExprValue::Boolean(lhs_value != rhs_value),
                    BinaryOperator::GreaterThan => ExprValue::Boolean(lhs_value > rhs_value),
                    BinaryOperator::GreaterThanOrEqual => {
                        ExprValue::Boolean(lhs_value >= rhs_value)
                    }
                    BinaryOperator::LessThan => ExprValue::Boolean(lhs_value < rhs_value),
                    BinaryOperator::LessThanOrEqual => ExprValue::Boolean(lhs_value <= rhs_value),
                })
            } else {
                Err(self.emit_mismatch_type_error_for_binary_op(op, &lhs, &rhs, &loc, "number"))
            }
        } else {
            Err(self.emit_mismatch_type_error_for_binary_op(op, &lhs, &rhs, &loc, "number"))
        }
    }

    fn interpret_binary_op_for_string(
        &self,
        op: &BinaryOperator,
        lhs: &ExprValue,
        rhs: &ExprValue,
        loc: &LocationSpan,
    ) -> Result<ExprValue> {
        if let ExprValue::String(lhs_value) = lhs {
            if let ExprValue::String(rhs_value) = rhs {
                match op {
                    BinaryOperator::Add => Ok(ExprValue::String(lhs_value.to_owned() + rhs_value)),
                    BinaryOperator::Substract => {
                        Err(self.emit_unsupported_binary_operator(op, loc, "string"))
                    }
                    BinaryOperator::Multiply => {
                        Err(self.emit_unsupported_binary_operator(op, loc, "string"))
                    }
                    BinaryOperator::Divide => {
                        Err(self.emit_unsupported_binary_operator(op, loc, "string"))
                    }
                    BinaryOperator::Equal => Ok(ExprValue::Boolean(lhs_value == rhs_value)),
                    BinaryOperator::NotEqual => Ok(ExprValue::Boolean(lhs_value != rhs_value)),
                    BinaryOperator::GreaterThan => Ok(ExprValue::Boolean(lhs_value > rhs_value)),
                    BinaryOperator::GreaterThanOrEqual => {
                        Ok(ExprValue::Boolean(lhs_value >= rhs_value))
                    }
                    BinaryOperator::LessThan => Ok(ExprValue::Boolean(lhs_value < rhs_value)),
                    BinaryOperator::LessThanOrEqual => {
                        Ok(ExprValue::Boolean(lhs_value <= rhs_value))
                    }
                }
            } else {
                Err(self.emit_mismatch_type_error_for_binary_op(op, &lhs, &rhs, &loc, "number"))
            }
        } else {
            Err(self.emit_mismatch_type_error_for_binary_op(op, &lhs, &rhs, &loc, "number"))
        }
    }

    fn emit_unsupported_binary_operator(
        &self,
        op: &BinaryOperator,
        loc: &LocationSpan,
        operand_type: &str,
    ) -> anyhow::Error {
        self.emit_eval_error(
            format!(
                "Binary operator '{}' not supported for {}",
                op.to_string(),
                operand_type,
            ),
            loc,
        )
    }

    fn emit_mismatch_type_error_for_binary_op(
        &self,
        op: &BinaryOperator,
        lhs: &ExprValue,
        rhs: &ExprValue,
        loc: &LocationSpan,
        expected_type: &str,
    ) -> anyhow::Error {
        self.emit_eval_error(
            format!(
                "Binary operator '{}' only supported for {} operands: given {} and {}",
                op.to_string(),
                expected_type,
                lhs.type_to_string(),
                rhs.type_to_string(),
            ),
            loc,
        )
    }

    fn emit_eval_error(&self, message: String, loc: &LocationSpan) -> anyhow::Error {
        return DiagnosticError::new(
            message,
            FileLocation::Span(loc.to_owned()),
            self.source_file.clone(),
        )
        .into();
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, Write};
    use std::path::PathBuf;

    use super::ExprValue;
    use crate::ast::interpreter::{interpret, Interpreter};
    use crate::ast::tests::{
        new_assign_expr, new_binary_expr, new_boolean_literal_expr, new_expr_stmt,
        new_grouping_expr, new_if_else_stmt, new_if_stmt, new_list_stmt, new_literal_expr,
        new_number_literal_expr, new_print_stmt, new_string_literal_expr, new_unary_expr,
        new_var_decl_stmt, new_variable_expr, new_while_stmt,
    };
    use crate::ast::{BinaryOperator, Expr, Literal, UnaryOperator};

    struct TestCursor {}

    impl Write for TestCursor {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Ok(0)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Creates a new interpreter for tests without an `output_writer` defined.
    fn new_test_interpreter() -> Interpreter<'static, TestCursor> {
        Interpreter::<TestCursor>::new("in-memory".into(), None)
    }

    /// Creates a new interpreter with a pre-defined set of variables in the global scope
    fn new_test_interpreter_with_variables(
        vars: Vec<(&str, ExprValue)>,
    ) -> Interpreter<'static, TestCursor> {
        let mut interpreter = new_test_interpreter();
        for (name, value) in vars {
            interpreter.env.define_variable(name, value);
        }
        interpreter
    }

    /////////////////////////////
    /// Tests for Expressions ///
    /////////////////////////////

    macro_rules! interpret_expr_and_check {
        ($test_data:expr) => {
            for (expr, expected_value) in $test_data {
                let mut interpreter = new_test_interpreter();
                let value = interpreter.interpret_expr(&expr).unwrap();
                assert_eq!(value, expected_value);
            }
        };
    }

    macro_rules! interpret_expr_and_expect_error {
        ($test_data:expr) => {
            for (expr, expected_error) in $test_data {
                let mut interpreter = new_test_interpreter();
                let value = interpreter.interpret_expr(&expr);
                assert!(value.is_err_and(|err| err.to_string().contains(expected_error)));
            }
        };
    }

    #[test]
    fn test_literal() {
        let test_data = vec![
            (new_number_literal_expr(10.0), ExprValue::Number(10.0)),
            (
                new_string_literal_expr("value"),
                ExprValue::String(String::from("value")),
            ),
            (new_boolean_literal_expr(true), ExprValue::Boolean(true)),
            (new_boolean_literal_expr(false), ExprValue::Boolean(false)),
            (new_literal_expr(Literal::Nil), ExprValue::Nil),
        ];
        interpret_expr_and_check!(test_data);
    }

    fn new_binary_expr_from_numbers<NumType1, NumType2>(
        op: BinaryOperator,
        lhs: NumType1,
        rhs: NumType2,
    ) -> Expr
    where
        NumType1: Into<f64>,
        NumType2: Into<f64>,
    {
        new_binary_expr(
            op,
            new_number_literal_expr(lhs),
            new_number_literal_expr(rhs),
        )
    }

    fn new_binary_expr_from_str(op: BinaryOperator, lhs: &str, rhs: &str) -> Expr {
        new_binary_expr(
            op,
            new_string_literal_expr(lhs),
            new_string_literal_expr(rhs),
        )
    }

    #[test]
    fn test_binary_operator() {
        let test_data = vec![
            (
                new_binary_expr_from_numbers(BinaryOperator::Add, 10, 20),
                ExprValue::Number(30.0),
            ),
            (
                new_binary_expr_from_numbers(BinaryOperator::Substract, 1, 2),
                ExprValue::Number(-1.0),
            ),
            (
                new_binary_expr_from_numbers(BinaryOperator::Multiply, 1.5, 8.0),
                ExprValue::Number(12.0),
            ),
            (
                new_binary_expr_from_numbers(BinaryOperator::Divide, 42.0, 6.0),
                ExprValue::Number(7.0),
            ),
            (
                new_binary_expr_from_numbers(BinaryOperator::Equal, 1.0, 1.2),
                ExprValue::Boolean(false),
            ),
            (
                new_binary_expr_from_numbers(BinaryOperator::NotEqual, 1.0, 1.2),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_from_numbers(BinaryOperator::GreaterThan, 2.5, 2.4),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_from_numbers(BinaryOperator::GreaterThanOrEqual, 2.5, 2.6),
                ExprValue::Boolean(false),
            ),
            (
                new_binary_expr_from_numbers(BinaryOperator::LessThan, 2.0, 2.4),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_from_numbers(BinaryOperator::LessThanOrEqual, 2.4, 2.4),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_from_str(BinaryOperator::Add, "a", "b"),
                ExprValue::String(String::from("ab")),
            ),
            (
                new_binary_expr_from_str(BinaryOperator::Equal, "a", "a"),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_from_str(BinaryOperator::NotEqual, "a", "b"),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_from_str(BinaryOperator::GreaterThan, "a", "b"),
                ExprValue::Boolean(false),
            ),
            (
                new_binary_expr_from_str(BinaryOperator::GreaterThanOrEqual, "a", "a"),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_from_str(BinaryOperator::LessThan, "a", "b"),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_from_str(BinaryOperator::Equal, "c", "b"),
                ExprValue::Boolean(false),
            ),
        ];

        interpret_expr_and_check!(test_data);
    }

    #[test]
    fn test_unary_operator() {
        let test_data = vec![
            (
                new_unary_expr(UnaryOperator::Minus, new_number_literal_expr(1)),
                ExprValue::Number(-1.0),
            ),
            (
                new_unary_expr(UnaryOperator::Minus, new_number_literal_expr(-1)),
                ExprValue::Number(1.0),
            ),
            (
                new_unary_expr(UnaryOperator::Not, new_boolean_literal_expr(true)),
                ExprValue::Boolean(false),
            ),
            (
                new_unary_expr(UnaryOperator::Not, new_boolean_literal_expr(false)),
                ExprValue::Boolean(true),
            ),
        ];

        interpret_expr_and_check!(test_data);
    }

    #[test]
    fn test_grouping_expr() {
        let test_data = vec![(
            new_grouping_expr(new_number_literal_expr(10)),
            ExprValue::Number(10.0),
        )];

        interpret_expr_and_check!(test_data);
    }

    #[test]
    fn test_interpret_complex_ast() {
        let test_data = vec![
            (
                new_binary_expr(
                    BinaryOperator::Add,
                    new_binary_expr_from_numbers(BinaryOperator::Multiply, 10.0, 2.0),
                    new_binary_expr_from_numbers(BinaryOperator::Divide, 20.0, 4.0),
                ),
                ExprValue::Number(25.0),
            ),
            (
                new_grouping_expr(new_unary_expr(
                    UnaryOperator::Minus,
                    new_grouping_expr(new_binary_expr_from_numbers(BinaryOperator::Add, 1.0, 0.5)),
                )),
                ExprValue::Number(-1.5),
            ),
            (
                new_binary_expr(
                    BinaryOperator::GreaterThanOrEqual,
                    new_grouping_expr(new_binary_expr_from_numbers(
                        BinaryOperator::Substract,
                        42.0,
                        20.0,
                    )),
                    new_grouping_expr(new_binary_expr_from_numbers(
                        BinaryOperator::Multiply,
                        4.0,
                        6.0,
                    )),
                ),
                ExprValue::Boolean(false),
            ),
        ];

        interpret_expr_and_check!(test_data);
    }

    #[test]
    fn test_interpret_with_errors() {
        let test_data = vec![
            (
                new_binary_expr(
                    BinaryOperator::Add,
                    new_number_literal_expr(10),
                    new_string_literal_expr("value"),
                ),
                "Binary operator '+' only supported for",
            ),
            (
                new_binary_expr(
                    BinaryOperator::Add,
                    new_string_literal_expr("value"),
                    new_number_literal_expr(10),
                ),
                "Binary operator '+' only supported for",
            ),
            (
                new_binary_expr(
                    BinaryOperator::Substract,
                    new_string_literal_expr("value"),
                    new_string_literal_expr("value"),
                ),
                "Binary operator '-' not supported",
            ),
            (
                new_binary_expr(
                    BinaryOperator::Multiply,
                    new_string_literal_expr("value"),
                    new_string_literal_expr("value"),
                ),
                "Binary operator '*' not supported",
            ),
            (
                new_binary_expr(
                    BinaryOperator::Divide,
                    new_string_literal_expr("value"),
                    new_string_literal_expr("value"),
                ),
                "Binary operator '/' not supported",
            ),
        ];

        interpret_expr_and_expect_error!(test_data);
    }

    #[test]
    fn test_assignment_single() {
        let mut interpreter =
            new_test_interpreter_with_variables(vec![("id", ExprValue::Number(1.0))]);
        let ast = new_assign_expr("id", new_number_literal_expr(10));
        let value = interpreter.interpret_expr(&ast).unwrap();
        assert_eq!(value, ExprValue::Number(10.0));
    }

    #[test]
    fn test_assignment_single_in_lexical_scope() {
        let ast = new_assign_expr("id", new_number_literal_expr(10));
        let mut interpreter =
            new_test_interpreter_with_variables(vec![("id", ExprValue::Number(1.0))]);
        interpreter.env.create_scope();
        let value = interpreter.interpret_expr(&ast).unwrap();
        interpreter.env.drop_innermost_scope();
        assert_eq!(value, ExprValue::Number(10.0));
    }

    #[test]
    fn test_assignment_nested() {
        let ast = new_assign_expr("id", new_assign_expr("id1", new_boolean_literal_expr(true)));
        let mut interpreter = new_test_interpreter_with_variables(vec![
            ("id", ExprValue::Boolean(true)),
            ("id1", ExprValue::Boolean(false)),
        ]);
        let value = interpreter.interpret_expr(&ast).unwrap();
        assert_eq!(value, ExprValue::Boolean(true));
    }

    #[test]
    fn test_eval_print_statement() {
        let ast = new_print_stmt(new_binary_expr(
            BinaryOperator::Add,
            new_number_literal_expr(1),
            new_number_literal_expr(2),
        ));
        let mut output_writer = std::io::Cursor::new(Vec::<u8>::new());
        interpret(&ast, PathBuf::from("in-memory"), Some(&mut output_writer)).unwrap();

        // read the content from the output writer and trim the ending whitespaces
        output_writer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut string_output = String::new();
        let _ = output_writer.read_to_string(&mut string_output);
        let captured_output = String::from(string_output.trim_end());

        assert_eq!(captured_output, String::from("3"));
    }

    macro_rules! interpret_stmt_and_check_var_value {
        ($ast:expr, $target_var:expr, $target_value:expr) => {
            interpret_stmt_and_check_var_value!($ast, vec![], $target_var, $target_value)
        };
        ($ast:expr, $vars:expr, $target_var:expr, $target_value:expr) => {
            let mut interpreter = new_test_interpreter_with_variables($vars);
            interpreter.interpret(&$ast).unwrap();
            assert_eq!(
                interpreter.env.get_variable($target_var).unwrap(),
                $target_value
            )
        };
    }

    #[test]
    fn test_variable_declaration() {
        interpret_stmt_and_check_var_value!(
            new_var_decl_stmt("id", new_number_literal_expr(2)),
            "id",
            ExprValue::Number(2.0)
        );
    }

    #[test]
    fn test_variable_usage() {
        interpret_stmt_and_check_var_value!(
            new_list_stmt(vec![
                new_var_decl_stmt("id", new_number_literal_expr(2)),
                new_var_decl_stmt("id_copied", new_variable_expr("id"))
            ]),
            "id_copied",
            ExprValue::Number(2.0)
        );
    }

    #[test]
    fn test_expression_statement() {
        interpret_stmt_and_check_var_value!(
            new_expr_stmt(new_assign_expr("id", new_string_literal_expr("value"))),
            vec![("id", ExprValue::Nil)],
            "id",
            ExprValue::String(String::from("value"))
        );
    }

    #[test]
    fn test_if_statement() {
        interpret_stmt_and_check_var_value!(
            new_if_stmt(
                new_number_literal_expr(1),
                new_expr_stmt(new_assign_expr("id", new_string_literal_expr("value")))
            ),
            vec![("id", ExprValue::Nil)],
            "id",
            ExprValue::String(String::from("value"))
        );
    }

    #[test]
    fn test_if_else_statement() {
        interpret_stmt_and_check_var_value!(
            new_if_else_stmt(
                new_boolean_literal_expr(false),
                new_expr_stmt(new_assign_expr("id", new_string_literal_expr("value"))),
                new_expr_stmt(new_assign_expr("id", new_number_literal_expr(10)))
            ),
            vec![("id", ExprValue::Nil)],
            "id",
            ExprValue::Number(10.0)
        );
    }

    #[test]
    fn test_while_statement() {
        interpret_stmt_and_check_var_value!(
            new_while_stmt(
                new_binary_expr(
                    BinaryOperator::LessThan,
                    new_variable_expr("id"),
                    new_number_literal_expr(10)
                ),
                new_expr_stmt(new_assign_expr(
                    "id",
                    new_binary_expr(
                        BinaryOperator::Add,
                        new_variable_expr("id"),
                        new_number_literal_expr(1)
                    )
                ))
            ),
            vec![("id", ExprValue::Number(0.0))],
            "id",
            ExprValue::Number(10.0)
        );
    }
}
