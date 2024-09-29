// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

//! Evaluator of Abstract Syntax Tree (AST) nodes.
//!
//! This module provides functions to execute (and thereby) eval nodes
//! of the AST.

use std::path::PathBuf;

use anyhow::Result;

use crate::{
    ast::{Expr, Literal},
    diagnostics::{DiagnosticError, FileLocation, LocationSpan},
};

use super::{BinaryOperator, UnaryOperator};

/// Evaluates the `Expr` node of the AST into an `ExprValue`.
pub fn eval_expr(expr: &Expr, source_file: PathBuf) -> Result<ExprValue> {
    let evaluator = ExprEvaluator::new(source_file);
    evaluator.eval(expr)
}

/// Result of evaluating an expression node of the AST.
#[derive(PartialEq, Debug)]
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

struct ExprEvaluator {
    source_file: PathBuf,
}

impl ExprEvaluator {
    fn new(source_file: PathBuf) -> Self {
        ExprEvaluator { source_file }
    }

    fn eval(&self, expr: &Expr) -> Result<ExprValue> {
        match expr {
            Expr::Binary { lhs, op, rhs, loc } => self.eval_binary_expr(op, lhs, rhs, loc),
            Expr::Grouping { expr, .. } => self.eval(expr),
            Expr::Literal { literal, .. } => self.eval_literal(&literal),
            Expr::Unary { op, expr, loc } => self.eval_unary_expr(op, expr, loc),
        }
    }

    fn eval_literal(&self, literal: &Literal) -> Result<ExprValue> {
        Ok(match literal {
            Literal::Boolean(value) => ExprValue::Boolean(*value),
            Literal::Nil => ExprValue::Nil,
            Literal::Number(value) => ExprValue::Number(*value),
            Literal::String(value) => ExprValue::String(String::from(value)),
        })
    }

    fn eval_unary_expr(
        &self,
        op: &UnaryOperator,
        expr: &Expr,
        loc: &LocationSpan,
    ) -> Result<ExprValue> {
        let result = self.eval(expr)?;
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

    fn eval_binary_expr(
        &self,
        op: &BinaryOperator,
        lhs: &Expr,
        rhs: &Expr,
        loc: &LocationSpan,
    ) -> Result<ExprValue> {
        let lhs_value = self.eval(lhs)?;
        let rhs_value = self.eval(rhs)?;

        match lhs_value {
            ExprValue::Number(_) => self.eval_binary_op_for_number(op, &lhs_value, &rhs_value, loc),
            ExprValue::String(_) => self.eval_binary_op_for_string(op, &lhs_value, &rhs_value, loc),
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

    fn eval_binary_op_for_number(
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

    fn eval_binary_op_for_string(
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
    use std::path::PathBuf;

    use super::{eval_expr, ExprValue};
    use crate::ast::{BinaryOperator, Expr, Literal, UnaryOperator};
    use crate::diagnostics::{Location, LocationSpan};

    fn default_loc_span() -> LocationSpan {
        LocationSpan {
            start: Location::new(1, 1),
            end_inclusive: Location::new(1, 1),
        }
    }

    fn new_literal_expr(literal: Literal) -> Expr {
        Expr::Literal {
            literal: literal,
            loc: default_loc_span(),
        }
    }

    fn new_binary_expr(op: BinaryOperator, lhs: Expr, rhs: Expr) -> Expr {
        Expr::Binary {
            lhs: Box::new(lhs),
            op,
            rhs: Box::new(rhs),
            loc: default_loc_span(),
        }
    }

    fn new_binary_expr_number(op: BinaryOperator, lhs: f64, rhs: f64) -> Expr {
        new_binary_expr(
            op,
            new_literal_expr(Literal::Number(lhs)),
            new_literal_expr(Literal::Number(rhs)),
        )
    }

    fn new_binary_expr_str(op: BinaryOperator, lhs: &str, rhs: &str) -> Expr {
        new_binary_expr(
            op,
            new_literal_expr(Literal::String(lhs.to_string())),
            new_literal_expr(Literal::String(rhs.to_string())),
        )
    }

    fn new_unary_expr(op: UnaryOperator, expr: Expr) -> Expr {
        Expr::Unary {
            op,
            expr: Box::new(expr),
            loc: default_loc_span(),
        }
    }

    fn new_grouping_expr(expr: Expr) -> Expr {
        Expr::Grouping {
            expr: Box::new(expr),
            loc: default_loc_span(),
        }
    }

    fn run_test_eval(test_data: Vec<(Expr, ExprValue)>) {
        for (expr, expected_value) in test_data {
            let value = eval_expr(&expr, PathBuf::from("in-memory")).unwrap();
            assert_eq!(value, expected_value);
        }
    }

    fn run_test_eval_with_expected_errors(test_data: Vec<(Expr, &str)>) {
        for (expr, expected_error) in test_data {
            let value = eval_expr(&expr, PathBuf::from("in-memory"));
            assert!(value.is_err_and(|err| err.to_string().contains(expected_error)));
        }
    }

    #[test]
    fn test_eval_from_literal_ast() {
        let test_data = vec![
            (
                new_literal_expr(Literal::Number(10.0)),
                ExprValue::Number(10.0),
            ),
            (
                new_literal_expr(Literal::String("abc".into())),
                ExprValue::String("abc".into()),
            ),
            (
                new_literal_expr(Literal::Boolean(true)),
                ExprValue::Boolean(true),
            ),
            (
                new_literal_expr(Literal::Boolean(false)),
                ExprValue::Boolean(false),
            ),
            (new_literal_expr(Literal::Nil), ExprValue::Nil),
        ];
        run_test_eval(test_data);
    }

    #[test]
    fn test_eval_binary_operator() {
        let test_data = vec![
            (
                new_binary_expr_number(BinaryOperator::Add, 10.0, 20.0),
                ExprValue::Number(30.0),
            ),
            (
                new_binary_expr_number(BinaryOperator::Substract, 1.0, 2.0),
                ExprValue::Number(-1.0),
            ),
            (
                new_binary_expr_number(BinaryOperator::Multiply, 1.5, 8.0),
                ExprValue::Number(12.0),
            ),
            (
                new_binary_expr_number(BinaryOperator::Divide, 42.0, 6.0),
                ExprValue::Number(7.0),
            ),
            (
                new_binary_expr_number(BinaryOperator::Equal, 1.0, 1.2),
                ExprValue::Boolean(false),
            ),
            (
                new_binary_expr_number(BinaryOperator::NotEqual, 1.0, 1.2),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_number(BinaryOperator::GreaterThan, 2.5, 2.4),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_number(BinaryOperator::GreaterThanOrEqual, 2.5, 2.6),
                ExprValue::Boolean(false),
            ),
            (
                new_binary_expr_number(BinaryOperator::LessThan, 2.0, 2.4),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_number(BinaryOperator::LessThanOrEqual, 2.4, 2.4),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_str(BinaryOperator::Add, "a", "b"),
                ExprValue::String("ab".into()),
            ),
            (
                new_binary_expr_str(BinaryOperator::Equal, "a", "a"),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_str(BinaryOperator::NotEqual, "a", "b"),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_str(BinaryOperator::GreaterThan, "a", "b"),
                ExprValue::Boolean(false),
            ),
            (
                new_binary_expr_str(BinaryOperator::GreaterThanOrEqual, "a", "a"),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_str(BinaryOperator::LessThan, "a", "b"),
                ExprValue::Boolean(true),
            ),
            (
                new_binary_expr_str(BinaryOperator::Equal, "c", "b"),
                ExprValue::Boolean(false),
            ),
        ];

        run_test_eval(test_data);
    }

    #[test]
    fn test_eval_unary_operator() {
        let test_data = vec![
            (
                new_unary_expr(UnaryOperator::Minus, new_literal_expr(Literal::Number(1.0))),
                ExprValue::Number(-1.0),
            ),
            (
                new_unary_expr(
                    UnaryOperator::Minus,
                    new_literal_expr(Literal::Number(-1.0)),
                ),
                ExprValue::Number(1.0),
            ),
            (
                new_unary_expr(UnaryOperator::Not, new_literal_expr(Literal::Boolean(true))),
                ExprValue::Boolean(false),
            ),
            (
                new_unary_expr(
                    UnaryOperator::Not,
                    new_literal_expr(Literal::Boolean(false)),
                ),
                ExprValue::Boolean(true),
            ),
        ];

        run_test_eval(test_data);
    }

    #[test]
    fn test_eval_grouping_expr() {
        let test_data = vec![(
            new_grouping_expr(new_literal_expr(Literal::Number(10.0))),
            ExprValue::Number(10.0),
        )];

        run_test_eval(test_data);
    }

    #[test]
    fn test_eval_complex_ast() {
        let test_data = vec![
            (
                new_binary_expr(
                    BinaryOperator::Add,
                    new_binary_expr_number(BinaryOperator::Multiply, 10.0, 2.0),
                    new_binary_expr_number(BinaryOperator::Divide, 20.0, 4.0),
                ),
                ExprValue::Number(25.0),
            ),
            (
                new_grouping_expr(new_unary_expr(
                    UnaryOperator::Minus,
                    new_grouping_expr(new_binary_expr_number(BinaryOperator::Add, 1.0, 0.5)),
                )),
                ExprValue::Number(-1.5),
            ),
            (
                new_binary_expr(
                    BinaryOperator::GreaterThanOrEqual,
                    new_grouping_expr(new_binary_expr_number(
                        BinaryOperator::Substract,
                        42.0,
                        20.0,
                    )),
                    new_grouping_expr(new_binary_expr_number(BinaryOperator::Multiply, 4.0, 6.0)),
                ),
                ExprValue::Boolean(false),
            ),
        ];

        run_test_eval(test_data);
    }

    #[test]
    fn test_eval_with_errors() {
        let test_data = vec![
            (
                new_binary_expr(
                    BinaryOperator::Add,
                    new_literal_expr(Literal::Number(10.0)),
                    new_literal_expr(Literal::String("abc".into())),
                ),
                "Binary operator '+' only supported for",
            ),
            (
                new_binary_expr(
                    BinaryOperator::Add,
                    new_literal_expr(Literal::String("abc".into())),
                    new_literal_expr(Literal::Number(10.0)),
                ),
                "Binary operator '+' only supported for",
            ),
            (
                new_binary_expr(
                    BinaryOperator::Substract,
                    new_literal_expr(Literal::String("abc".into())),
                    new_literal_expr(Literal::String("abc".into())),
                ),
                "Binary operator '-' not supported",
            ),
            (
                new_binary_expr(
                    BinaryOperator::Multiply,
                    new_literal_expr(Literal::String("abc".into())),
                    new_literal_expr(Literal::String("abc".into())),
                ),
                "Binary operator '*' not supported",
            ),
            (
                new_binary_expr(
                    BinaryOperator::Divide,
                    new_literal_expr(Literal::String("abc".into())),
                    new_literal_expr(Literal::String("abc".into())),
                ),
                "Binary operator '/' not supported",
            ),
        ];

        run_test_eval_with_expected_errors(test_data);
    }
}
