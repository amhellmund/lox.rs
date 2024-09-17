use std::path::PathBuf;

use anyhow::Result;

use crate::{
    ast::{Expr, Literal},
    diagnostics::{DiagnosticError, FileLocation, LocationSpan},
};

use super::{BinaryOperator, UnaryOperator};

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
            ExprValue::String(value) => value.to_string(),
        }
    }
}

pub fn eval_expr(expr: &Expr, source_file: PathBuf) -> Result<ExprValue> {
    let evaluator = ExprEvaluator::new(source_file);
    evaluator.eval(expr)
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
