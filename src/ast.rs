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

use crate::diagnostics::LocationSpan;

pub mod eval;
pub mod printer;

#[derive(PartialEq, Debug)]
pub enum UnaryOperator {
    Minus,
    Not,
}

impl ToString for UnaryOperator {
    fn to_string(&self) -> String {
        match *self {
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
        match *self {
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

#[derive(PartialEq, Debug)]
pub enum Literal {
    Number(f64),
    String(String),
    Boolean(bool),
    Nil,
}

#[derive(PartialEq, Debug)]
pub enum Expr {
    Binary {
        lhs: Box<Expr>,
        op: BinaryOperator,
        rhs: Box<Expr>,
        loc: LocationSpan,
    },
    Unary {
        op: UnaryOperator,
        expr: Box<Expr>,
        loc: LocationSpan,
    },
    Literal {
        literal: Literal,
        loc: LocationSpan,
    },
    Grouping {
        expr: Box<Expr>,
        loc: LocationSpan,
    },
}

impl Expr {
    pub fn get_loc(&self) -> &LocationSpan {
        match self {
            Expr::Binary { loc, .. } => loc,
            Expr::Unary { loc, .. } => loc,
            Expr::Grouping { loc, .. } => loc,
            Expr::Literal { loc, .. } => loc,
        }
    }
}

// pub enum Stmt {
//     Expr(Expr),
//     Print(Expr),
// }
