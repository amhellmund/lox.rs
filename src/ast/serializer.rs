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

use super::{ExprData, Stmt, StmtData};

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
/// Note on Performance: the performance of the serialization is sub-optimal due to many inefficient
/// string operations and recursions. For example: the indentation is achieved by "unwinding" the
/// call stack und repeatedly adding the next-level indentation. The design however has nevertheless
/// been chosen due to simplicity of the serialization code itself. The sub-optimal performance is accepted
/// as of now because the serialization is seldomly used in production and mostly for development, debugging
/// and testing purpose.
struct AstTopologicalSerializer {
    options: AstSerializerOptions,
}

impl AstTopologicalSerializer {
    const INDENT_NEXT_LEVEL: i64 = 2;

    pub fn new(options: AstSerializerOptions) -> Self {
        Self { options }
    }

    pub fn serialize(self, stmt: &Stmt) -> String {
        self.serialize_stmt(stmt)
    }

    fn generate_ast_entry(
        &self,
        tag: &str,
        loc: &LocationSpan,
        sub_entries: Vec<String>,
    ) -> String {
        let mut content = String::default();
        if self.options.include_location {
            content += &format!("({} [{}]\n", tag, loc.to_string());
        } else {
            content += &format!("({}\n", tag)
        }
        for sub_entry in sub_entries {
            content += &format!(
                "{}{}",
                Self::get_indent_as_string(Self::INDENT_NEXT_LEVEL),
                sub_entry,
            );
        }
        content += ")\n";
        content
    }

    fn serialize_stmt(&self, stmt: &Stmt) -> String {
        let loc = stmt.get_loc();
        match &stmt.data {
            StmtData::Block { statements } => self.serialize_stmt_list("block", loc, statements),
            StmtData::Expr { expr } => {
                self.generate_ast_entry("expr", loc, vec![self.serialize_expr(expr)])
            }
            StmtData::If {
                condition,
                if_statement,
                else_statement,
            } => {
                let mut sub_entries = vec![
                    self.serialize_expr(condition),
                    self.serialize_stmt(if_statement),
                ];
                if let Some(else_statement) = else_statement {
                    sub_entries.push(self.serialize_stmt(else_statement));
                }
                self.generate_ast_entry(tag, loc, sub_entries)
            }
            StmtData::List { statements } => self.serialize_stmt_list("list", loc, statements),
            StmtData::Print { expr } => {
                self.generate_ast_entry_single("print", loc, self.serialize_expr(expr))
            }
            StmtData::VarDecl {
                identifier,
                init_expr,
            } => self.generate_ast_entry(
                "var-decl",
                loc,
                vec![identifier.clone(), self.serialize_expr(init_expr)],
            ),
            StmtData::While { condition, body } => self.generate_ast_entry_single(
                "while",
                loc,
                vec![self.serialize_expr(condition), self.serialize_stmt(body)],
            ),
        }
    }

    fn serialize_expr(&self, expr: &Expr) -> String {
        let loc = expr.get_loc();
        match &expr.data {
            ExprData::Binary { lhs, op, rhs } => self.generate_ast_entry(
                &op.to_string(),
                loc,
                vec![self.serialize_expr(lhs), self.serialize_expr(rhs)],
            ),
            ExprData::Unary { op, expr } => {
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
}
