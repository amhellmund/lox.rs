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

pub mod eval;
pub mod printer;

use crate::diagnostics::LocationSpan;

/// Statements.
#[derive(PartialEq, Debug)]
struct Stmt {
    data: StmtData,
    loc: LocationSpan,
}

impl Stmt {
    pub fn new(data: StmtData, loc: LocationSpan) -> Self {
        Self { data, loc }
    }

    pub fn new_as_box(data: StmtData, loc: LocationSpan) -> Box<Self> {
        Box::new(Self::new(data, loc))
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

    pub fn new_as_box(data: ExprData, loc: LocationSpan) -> Box<Self> {
        Box::new(Self::new(data, loc))
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

#[cfg(test)]
pub mod tests {
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

    fn new_variable_assignment(name: &str, expr: Expr) -> Expr {
        Expr::Assign {
            name: name.into(),
            expr: Box::new(expr),
            loc: default_loc_span(),
        }
    }
}
