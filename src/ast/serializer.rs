// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

//! Printer for the Abstract Syntax Tree (AST).
//!
//! Prints the structure of the AST hierarchically on the command line.

use crate::{
    ast::{Expr, Literal},
    diagnostics::LocationSpan,
};

use super::{Stmt, StmtData};

#[derive(Clone)]
pub struct AstSerializerOptions {
    /// Include the location information (lines and columns) for statements and expressions.
    include_location: bool,
}

/// Utility class to create a topological (String) serialization of an Abstract Syntax Tree (AST).
///
/// Given an AST of the structure:
///
/// ASSIGN:
/// |_ IDENTIFIER
/// |_ PLUS:
///    |_ IDENTIFIER
///    |_ NUMBER
///
/// the topological serializer creates a string representation using a depth-first search approach:
///
/// (ASSIGN IDENTIFIER (PLUS IDENTIFIER NUMBER))
///
/// This class may be used for different purposes, e.g. to store the AST to a file or use it for
/// testing purpose to easily compare two ASTs structurally.
///
struct AstTopologicalSerializer {
    options: AstSerializerOptions,
    indent: i64,
    content: String,
}

impl AstTopologicalSerializer {
    const INDENT_NEXT_LEVEL: i64 = 2;

    fn new(options: AstSerializerOptions, indent: i64) -> Self {
        Self {
            options,
            indent,
            content: String::default(),
        }
    }

    fn new_nested_serializer(&self) -> Self {
        Self::new(self.options.clone(), self.indent + Self::INDENT_NEXT_LEVEL)
    }

    fn serialize(mut self, stmt: &Stmt) -> String {
        self.serialize_stmt(stmt);
        self.content
    }

    fn serialize_stmt(&mut self, stmt: &Stmt) {
        match &stmt.data {
            StmtData::Block { statements } => {}
            StmtData::Expr { expr } => {}
            StmtData::If {
                condition,
                if_statement,
                else_statement,
            } => {}
            StmtData::List { statements } => {}
            StmtData::Print { expr } => {}
            StmtData::VarDecl {
                identifier,
                init_expr,
            } => {}
            StmtData::While { condition, body } => {}
        }
    }
}

pub fn serialize_ast(stmt: &Stmt, options: AstSerializerOptions) -> String {
    let serializer = AstTopologicalSerializer::new(options);
    serializer.serialize(stmt)
}

/// Prints the AST to the command-line
pub fn print_ast(stmt: &Stmt) -> String {
    print_stmt(stmt, 0)
}

fn get_indent_as_string(indent: i64) -> String {
    std::iter::repeat(" ")
        .take(indent as usize)
        .collect::<String>()
}

fn stringify_statements(
    statements: &Vec<Stmt>,
    ast_tag: &str,
    loc: &LocationSpan,
    indent: i64,
) -> String {
    let mut statement_string = format!(
        "{}[{}] [{}]\n",
        get_indent_as_string(indent),
        ast_tag,
        loc.to_string()
    );
    for statement in statements {
        statement_string += &print_stmt(statement, indent + INDENT_NEXT_LEVEL);
    }
    statement_string
}

fn print_stmt(stmt: &Stmt, indent: i64) -> String {
    match stmt {
        Stmt::List { statements, loc } => stringify_statements(statements, "List", loc, indent),
        Stmt::Block { statements, loc } => stringify_statements(statements, "Block", loc, indent),
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
        Stmt::If {
            condition,
            if_statement,
            else_statement,
            loc,
        } => {
            if else_statement.is_some() {
                format!(
                    "{}[If] [{}]\n{}\n{}\n{}",
                    get_indent_as_string(indent),
                    loc.to_string(),
                    print_expr(condition, indent + INDENT_NEXT_LEVEL),
                    print_stmt(if_statement, indent + INDENT_NEXT_LEVEL),
                    print_stmt(else_statement.as_ref().unwrap(), indent + INDENT_NEXT_LEVEL),
                )
            } else {
                format!(
                    "{}[If] [{}]\n{}\n{}",
                    get_indent_as_string(indent),
                    loc.to_string(),
                    print_expr(condition, indent + INDENT_NEXT_LEVEL),
                    print_stmt(if_statement, indent + INDENT_NEXT_LEVEL),
                )
            }
        }
        Stmt::While {
            condition,
            body,
            loc,
        } => {
            format!(
                "{}[While] [{}]\n{}\n{}",
                get_indent_as_string(indent),
                loc.to_string(),
                print_expr(&condition, indent + INDENT_NEXT_LEVEL),
                print_stmt(&body, indent + INDENT_NEXT_LEVEL),
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
