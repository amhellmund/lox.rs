// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

//! Printer for the Abstract Syntax Tree (AST).
//!
//! Prints the structure of the AST hierarchically on the command line.

use crate::ast::{Expr, Literal};

use super::Stmt;

const INDENT_NEXT_LEVEL: i64 = 2;

/// Prints the AST to the command-line
pub fn print_ast(stmt: &Stmt) -> String {
    print_stmt(stmt, 0)
}

fn get_indent_as_string(indent: i64) -> String {
    std::iter::repeat(" ")
        .take(indent as usize)
        .collect::<String>()
}

fn stringify_statements(statements: &Vec<Stmt>, ast_tag: &str, indent: i64) -> String {
    let mut statement_string = format!("{}[{}]\n", get_indent_as_string(indent), ast_tag);
    for statement in statements {
        statement_string += &print_stmt(statement, indent + INDENT_NEXT_LEVEL);
    }
    statement_string
}

fn print_stmt(stmt: &Stmt, indent: i64) -> String {
    match stmt {
        Stmt::List(statements) => stringify_statements(statements, "Stmt", indent),
        Stmt::Block(statements) => stringify_statements(statements, "Block", indent),
        Stmt::VarDecl {
            identifier,
            init_expr,
            loc,
        } => {
            format!(
                "{}[VarDecl] [{}]\n{}\"{}\n{}\n",
                get_indent_as_string(indent),
                loc.to_string(),
                get_indent_as_string(indent + INDENT_NEXT_LEVEL),
                identifier,
                print_expr(init_expr, indent + INDENT_NEXT_LEVEL),
            )
        }
        Stmt::Expr { expr, loc } => {
            format!(
                "{}[Expr] [{}]\n{}\n",
                get_indent_as_string(indent),
                loc.to_string(),
                print_expr(expr, indent + INDENT_NEXT_LEVEL),
            )
        }
        Stmt::Print { expr, loc } => {
            format!(
                "{}[Print] [{}]\n{}\n",
                get_indent_as_string(indent),
                loc.to_string(),
                print_expr(expr, indent + INDENT_NEXT_LEVEL),
            )
        }
    }
}

fn print_expr(expr: &Expr, indent: i64) -> String {
    match expr {
        Expr::Binary { lhs, op, rhs, loc } => {
            let lhs_str = print_expr(lhs, indent + INDENT_NEXT_LEVEL);
            let op_str = op.to_string();
            let rhs_string = print_expr(rhs, indent + INDENT_NEXT_LEVEL);
            format!(
                "{}{} [{}]\n{}\n{}",
                get_indent_as_string(indent),
                op_str,
                loc.to_string(),
                lhs_str,
                rhs_string
            )
        }
        Expr::Unary { op, expr, loc } => {
            let op_str = op.to_string();
            let expr_str = print_expr(expr, indent + INDENT_NEXT_LEVEL);
            format!(
                "{}{} [{}]\n{}",
                get_indent_as_string(indent),
                op_str,
                loc.to_string(),
                expr_str
            )
        }
        Expr::Grouping { expr, loc } => {
            let expr_str = print_expr(expr, indent + INDENT_NEXT_LEVEL);
            format!(
                "{}() [{}]\n{}",
                get_indent_as_string(indent),
                loc.to_string(),
                expr_str
            )
        }
        Expr::Variable { name, loc } => {
            format!(
                "{}<var: {}> [{}]",
                get_indent_as_string(indent),
                name,
                loc.to_string()
            )
        }
        Expr::Assign { name, expr, loc } => {
            let expr_str = print_expr(expr, indent + INDENT_NEXT_LEVEL);
            format!(
                "{}= [{}]\n{}{}\n{}",
                get_indent_as_string(indent),
                loc.to_string(),
                get_indent_as_string(indent + INDENT_NEXT_LEVEL),
                name,
                expr_str,
            )
        }
        Expr::Literal {
            literal: Literal::Boolean(value),
            loc,
        } => format!(
            "{}{} [{}]",
            get_indent_as_string(indent),
            value,
            loc.to_string()
        ),
        Expr::Literal {
            literal: Literal::Nil,
            loc,
        } => format!("{}nil [{}]", get_indent_as_string(indent), loc.to_string()),
        Expr::Literal {
            literal: Literal::String(value),
            loc,
        } => format!(
            "{}\"{}\" [{}]",
            get_indent_as_string(indent),
            value,
            loc.to_string()
        ),
        Expr::Literal {
            literal: Literal::Number(value),
            loc,
        } => format!(
            "{}{} [{}]",
            get_indent_as_string(indent),
            value,
            loc.to_string()
        ),
    }
}
