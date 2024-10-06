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

pub mod interpreter;
pub mod serializer;

use crate::diagnostics::LocationSpan;
use strum_macros::EnumIter;

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
    Block {
        statements: Vec<Stmt>,
    },
    Expr {
        expr: Box<Expr>,
    },
    If {
        condition: Box<Expr>,
        if_statement: Box<Stmt>,
        else_statement: Option<Box<Stmt>>,
    },
    List {
        statements: Vec<Stmt>,
    },
    Print {
        expr: Box<Expr>,
    },
    VarDecl {
        identifier: String,
        init_expr: Box<Expr>,
    },
    While {
        condition: Box<Expr>,
        body: Box<Stmt>,
    },
}

#[derive(Debug, Clone, Copy, EnumIter, PartialEq)]
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

#[derive(Clone, Copy, Debug, EnumIter, PartialEq)]
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

    use super::{BinaryOperator, Expr, ExprData, Literal, Stmt, StmtData, UnaryOperator};

    /// Creates a default location span for testing purpose.
    pub fn default_loc_span() -> LocationSpan {
        LocationSpan {
            start: Location::new(1, 1),
            end_inclusive: Location::new(1, 1),
        }
    }

    // Creates a new statement by setting the location to a default value.
    pub fn new_stmt(stmt_data: StmtData) -> Stmt {
        Stmt {
            data: stmt_data,
            loc: default_loc_span(),
        }
    }

    /// Creates a new list statement.
    pub fn new_list_stmt(statements: Vec<Stmt>) -> Stmt {
        new_stmt(StmtData::List { statements })
    }

    /// Creates a new block statement.
    pub fn new_block_stmt(statements: Vec<Stmt>) -> Stmt {
        new_stmt(StmtData::Block {
            statements: statements,
        })
    }

    /// Creates a new expression statement.
    pub fn new_expr_stmt(expr: Expr) -> Stmt {
        new_stmt(StmtData::Expr {
            expr: expr.as_box(),
        })
    }

    /// Creates a new if-else statement.
    pub fn new_if_stmt(condition: Expr, if_stmt: Stmt) -> Stmt {
        new_stmt(StmtData::If {
            condition: condition.as_box(),
            if_statement: if_stmt.as_box(),
            else_statement: None,
        })
    }

    /// Creates a new if-else statement.
    pub fn new_if_else_stmt(condition: Expr, if_stmt: Stmt, else_stmt: Stmt) -> Stmt {
        new_stmt(StmtData::If {
            condition: condition.as_box(),
            if_statement: if_stmt.as_box(),
            else_statement: Some(else_stmt.as_box()),
        })
    }

    /// Creates a new print statement.
    pub fn new_print_stmt(expr: Expr) -> Stmt {
        new_stmt(StmtData::Print {
            expr: expr.as_box(),
        })
    }

    /// Creates a new variable declaration statement.
    pub fn new_var_decl_stmt(identifier: &str, init_expr: Expr) -> Stmt {
        new_stmt(StmtData::VarDecl {
            identifier: identifier.to_string(),
            init_expr: init_expr.as_box(),
        })
    }

    /// Creates a new while statement.
    pub fn new_while_stmt(condition: Expr, body: Stmt) -> Stmt {
        new_stmt(StmtData::While {
            condition: condition.as_box(),
            body: body.as_box(),
        })
    }

    /// Creates a new expression by setting the location to a default value.
    pub fn new_expr(expr_data: ExprData) -> Expr {
        Expr {
            data: expr_data,
            loc: default_loc_span(),
        }
    }

    /// Creates a new assig expression.
    pub fn new_assign_expr(name: &str, rvalue: Expr) -> Expr {
        new_expr(ExprData::Assign {
            name: name.into(),
            expr: rvalue.as_box(),
        })
    }

    /// Creates a new literal expression from a scalar literal value.
    pub fn new_literal_expr(literal: Literal) -> Expr {
        new_expr(ExprData::Literal { literal })
    }

    // Creates a new boolean literal.
    pub fn new_boolean_literal_expr(value: bool) -> Expr {
        new_literal_expr(Literal::Boolean(value))
    }

    /// Creates a new number literal from any type convertible to the target type.
    pub fn new_number_literal_expr<T: Into<f64>>(value: T) -> Expr {
        new_literal_expr(Literal::Number(value.into()))
    }

    /// Creates a new string literal.
    pub fn new_string_literal_expr(value: &str) -> Expr {
        new_literal_expr(Literal::String(String::from(value)))
    }

    /// Creates a new binary expression.
    pub fn new_binary_expr(op: BinaryOperator, lhs: Expr, rhs: Expr) -> Expr {
        new_expr(ExprData::Binary {
            lhs: lhs.as_box(),
            op,
            rhs: rhs.as_box(),
        })
    }

    /// Creating a grouping expression.
    pub fn new_grouping_expr(expr: Expr) -> Expr {
        new_expr(ExprData::Grouping {
            expr: expr.as_box(),
        })
    }

    /// Creates a new unary expression.
    pub fn new_unary_expr(op: UnaryOperator, expr: Expr) -> Expr {
        new_expr(ExprData::Unary {
            op,
            expr: expr.as_box(),
        })
    }

    /// Creates a new variable expression.
    pub fn new_variable_expr(name: &str) -> Expr {
        new_expr(ExprData::Variable { name: name.into() })
    }
}
