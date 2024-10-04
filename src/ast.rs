// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

//! Type definitions for Abstract Syntax Tree.
//!
//! An abstract syntax tree gets composed of individual nodes, e.g. a node for binary expressions.
//! The nodes get arranged in a (hierarchical) tree structure. For example, the Lox statement
//!
//!   var a = 12 + 4 * 1;
//!
//! gets represented as
//!
//!   VarDecl-Stmt
//!     < a >
//!     BinaryExpr
//!       < + >
//!       BinaryExpr
//!         < * >
//!         < 4 >
//!         < 1 >

// pub mod interpreter;
pub mod serializer;

use crate::diagnostics::LocationSpan;

/// Statements.
#[derive(PartialEq, Debug)]
pub struct Stmt {
    data: StmtData,
    loc: LocationSpan,
}

impl Stmt {
    pub fn new(data: StmtData, loc: LocationSpan) -> Self {
        Self { data, loc }
    }

    pub fn as_box(self) -> Box<Self> {
        Box::new(self)
    }

    pub fn get_data(&self) -> &StmtData {
        &self.data
    }

    pub fn get_loc(&self) -> &LocationSpan {
        &self.loc
    }
}

#[derive(PartialEq, Debug)]
pub enum StmtData {
    List {
        statements: Vec<Stmt>,
    },
    Block {
        statements: Vec<Stmt>,
    },
    VarDecl {
        identifier: String,
        init_expr: Box<Expr>,
    },
    Expr {
        expr: Box<Expr>,
    },
    Print {
        expr: Box<Expr>,
    },
    If {
        condition: Box<Expr>,
        if_statement: Box<Stmt>,
        else_statement: Option<Box<Stmt>>,
    },
    While {
        condition: Box<Expr>,
        body: Box<Stmt>,
    },
}

#[derive(PartialEq, Debug)]
pub enum UnaryOperator {
    Minus,
    Not,
}

impl ToString for UnaryOperator {
    fn to_string(&self) -> String {
        match self {
            UnaryOperator::Minus => String::from("-"),
            UnaryOperator::Not => String::from("!"),
        }
    }
}

#[derive(PartialEq, Debug)]
pub enum BinaryOperator {
    Add,
    Substract,
    Multiply,
    Divide,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
}

impl ToString for BinaryOperator {
    fn to_string(&self) -> String {
        match self {
            BinaryOperator::Add => String::from("+"),
            BinaryOperator::Substract => String::from("-"),
            BinaryOperator::Multiply => String::from("*"),
            BinaryOperator::Divide => String::from("/"),
            BinaryOperator::Equal => String::from("=="),
            BinaryOperator::NotEqual => String::from("!="),
            BinaryOperator::GreaterThan => String::from(">"),
            BinaryOperator::GreaterThanOrEqual => String::from(">="),
            BinaryOperator::LessThan => String::from("<"),
            BinaryOperator::LessThanOrEqual => String::from("<="),
        }
    }
}

/// Expression Literal.
#[derive(PartialEq, Debug)]
pub enum Literal {
    Number(f64),
    String(String),
    Boolean(bool),
    Nil,
}

/// Expression.
#[derive(PartialEq, Debug)]
pub struct Expr {
    data: ExprData,
    loc: LocationSpan,
}

impl Expr {
    pub fn new(data: ExprData, loc: LocationSpan) -> Self {
        Self { data, loc }
    }

    pub fn as_box(self) -> Box<Self> {
        Box::new(self)
    }

    pub fn new_literal(literal: Literal, loc: LocationSpan) -> Self {
        Self {
            data: ExprData::Literal { literal },
            loc,
        }
    }

    pub fn get_data(&self) -> &ExprData {
        &self.data
    }

    pub fn get_loc(&self) -> &LocationSpan {
        &self.loc
    }
}

#[derive(PartialEq, Debug)]
pub enum ExprData {
    Binary {
        lhs: Box<Expr>,
        op: BinaryOperator,
        rhs: Box<Expr>,
    },
    Unary {
        op: UnaryOperator,
        expr: Box<Expr>,
    },
    Literal {
        literal: Literal,
    },
    Grouping {
        expr: Box<Expr>,
    },
    Variable {
        name: String,
    },
    Assign {
        name: String,
        expr: Box<Expr>,
    },
}

/// Test utilities when working with the Abstract Syntax Tree.
#[cfg(test)]
pub mod tests {
    use crate::diagnostics::{Location, LocationSpan};

    use super::{BinaryOperator, Expr, ExprData, Literal, Stmt, UnaryOperator};

    /// Compares two statements for equality without into account the 'data' field only.
    ///
    /// This function may be used when the location information is irrelevant for a test case.
    pub fn compare_stmt_equal_data(lhs: &Stmt, rhs: &Stmt) -> bool {
        lhs.data == rhs.data
    }

    /// Compares two expressions for eqaulity taking into account the data only.
    ///
    /// This function may be used when the location information is irrelevant for a test case.
    pub fn compare_expr_equal_data(lhs: &Expr, rhs: &Expr) -> bool {
        lhs.data == rhs.data
    }

    /// Creates a default location span for testing purpose.
    pub fn default_loc_span() -> LocationSpan {
        LocationSpan {
            start: Location::new(1, 1),
            end_inclusive: Location::new(1, 1),
        }
    }

    /// Creates a new expression by setting the location to a default value.
    fn new_expr(expr_data: ExprData) -> Expr {
        Expr {
            data: expr_data,
            loc: default_loc_span(),
        }
    }

    /// Creates a new literal expression from a scalar literal value.
    fn new_literal_expr(literal: Literal) -> Expr {
        new_expr(ExprData::Literal { literal })
    }

    /// Creates a new number literal from any type convertible to the target type.
    fn new_number_literal<T: Into<f64>>(value: T) -> Expr {
        new_literal_expr(Literal::Number(value.into()))
    }

    /// Creates a new string literal.
    fn new_string_literal(value: &str) -> Expr {
        new_literal_expr(Literal::String(String::from(value)))
    }

    /// Creates a new binary expression.
    fn new_binary_expr(op: BinaryOperator, lhs: Expr, rhs: Expr) -> Expr {
        new_expr(ExprData::Binary {
            lhs: lhs.as_box(),
            op,
            rhs: rhs.as_box(),
        })
    }

    /// Creates a new unary expression.
    fn new_unary_expr(op: UnaryOperator, expr: Expr) -> Expr {
        new_expr(ExprData::Unary {
            op,
            expr: expr.as_box(),
        })
    }

    /// Creating a grouping expression.
    fn new_grouping_expr(expr: Expr) -> Expr {
        new_expr(ExprData::Grouping {
            expr: expr.as_box(),
        })
    }

    /// Create a new variable expression.
    fn new_variable_assignment(name: &str, expr: Expr) -> Expr {
        new_expr(ExprData::Assign {
            name: String::from(name),
            expr: expr.as_box(),
        })
    }
}
