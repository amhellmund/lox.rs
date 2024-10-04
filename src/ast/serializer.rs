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
/// Note on Performance: the performance of the serialization code is sub-optimal due to many inefficient
/// string operations and recursive function calls. For example: the indentation of the sub entries is achieved
/// by "unwinding" call stack and repeatedly add the next-level indentation. The design has nevertheless been chosen
/// due to the simplicity of the actual code. Due to the fact that this code is seldomly used in production and mostly
/// for development, debugging and testing purpose, the disadvantages are accepted as of now.
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

    fn get_indent_as_string(indent: i64) -> String {
        std::iter::repeat(" ")
            .take(indent as usize)
            .collect::<String>()
    }

    fn get_loc_string(&self, loc: &LocationSpan) -> String {
        let mut loc_string = String::new();
        if self.options.include_location {
            loc_string = format!(" [{}]", loc.to_string());
        }
        loc_string
    }

    fn generate_ast_serialization(
        &self,
        tag: &str,
        loc: &LocationSpan,
        sub_entries: Vec<String>,
    ) -> String {
        let mut content = format!("({}{}\n", tag, self.get_loc_string(loc));
        for sub_entry in sub_entries {
            content += &format!(
                "{}{}\n",
                Self::get_indent_as_string(Self::INDENT_NEXT_LEVEL),
                sub_entry,
            );
        }
        content += ")\n";
        content
    }

    fn generate_ast_serialization_for_literal(
        &self,
        tag: &str,
        loc: &LocationSpan,
        value: Option<String>,
    ) -> String {
        let mut value_string = String::new();
        if let Some(value) = value {
            value_string = format!(" {}", value);
        }
        format!("({}{}){}", tag, value_string, self.get_loc_string(loc))
    }

    fn serialize_stmt(&self, stmt: &Stmt) -> String {
        let loc = stmt.get_loc();
        match &stmt.data {
            StmtData::Block { statements } => self.serialize_stmt_list("block", loc, statements),
            StmtData::Expr { expr } => {
                self.generate_ast_serialization("expr", loc, vec![self.serialize_expr(expr)])
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
                self.generate_ast_serialization("if", loc, sub_entries)
            }
            StmtData::List { statements } => self.serialize_stmt_list("list", loc, statements),
            StmtData::Print { expr } => {
                self.generate_ast_serialization("print", loc, vec![self.serialize_expr(expr)])
            }
            StmtData::VarDecl {
                identifier,
                init_expr,
            } => self.generate_ast_serialization(
                "var-decl",
                loc,
                vec![identifier.clone(), self.serialize_expr(init_expr)],
            ),
            StmtData::While { condition, body } => self.generate_ast_serialization(
                "while",
                loc,
                vec![self.serialize_expr(condition), self.serialize_stmt(body)],
            ),
        }
    }

    fn serialize_stmt_list(&self, tag: &str, loc: &LocationSpan, statements: &Vec<Stmt>) -> String {
        let serialized_statements: Vec<String> = statements
            .iter()
            .map(|stmt| self.serialize_stmt(stmt))
            .collect();
        self.generate_ast_serialization(tag, loc, serialized_statements)
    }

    fn serialize_expr(&self, expr: &Expr) -> String {
        let loc = expr.get_loc();
        match &expr.data {
            ExprData::Binary { lhs, op, rhs } => self.generate_ast_serialization(
                &op.to_string(),
                loc,
                vec![self.serialize_expr(lhs), self.serialize_expr(rhs)],
            ),
            ExprData::Unary { op, expr } => self.generate_ast_serialization(
                &op.to_string(),
                loc,
                vec![self.serialize_expr(expr)],
            ),
            ExprData::Grouping { expr } => {
                self.generate_ast_serialization("group", loc, vec![self.serialize_expr(expr)])
            }
            ExprData::Variable { name } => {
                self.generate_ast_serialization("var", loc, vec![name.clone()])
            }
            ExprData::Assign { name, expr } => self.generate_ast_serialization(
                "assign",
                loc,
                vec![name.clone(), self.serialize_expr(expr)],
            ),
            ExprData::Literal {
                literal: Literal::Boolean(value),
            } => self.generate_ast_serialization_for_literal("bool", loc, Some(value.to_string())),
            ExprData::Literal {
                literal: Literal::Nil,
            } => self.generate_ast_serialization_for_literal("nil", loc, None),
            ExprData::Literal {
                literal: Literal::String(value),
            } => self.generate_ast_serialization_for_literal("string", loc, Some(value.clone())),
            ExprData::Literal {
                literal: Literal::Number(value),
            } => {
                self.generate_ast_serialization_for_literal("number", loc, Some(value.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{
        tests::{
            new_boolean_literal, new_if_stmt, new_literal_expr, new_number_literal, new_print_stmt,
        },
        Literal, Stmt,
    };

    use super::{AstSerializerOptions, AstTopologicalSerializer};

    use crate::ast::tests::{new_expr, new_var_decl_stmt};

    fn test_serialize(stmt: Stmt, expected: String) {
        let ser = AstTopologicalSerializer::new(AstSerializerOptions {
            include_location: false,
        });
        let output = ser.serialize(&stmt);
        assert_eq!(output, expected);
    }

    fn expect_builder(tag: &str, sub_builder: Vec<String>) -> String {
        let mut content = format!("({}\n", tag);
        for sub in sub_builder {
            content += &format!(
                "{}{}\n",
                AstTopologicalSerializer::get_indent_as_string(
                    AstTopologicalSerializer::INDENT_NEXT_LEVEL
                ),
                sub
            )
        }
        content += ")\n";
        content
    }

    fn expect_builder_literal(tag: &str, value: String) -> String {
        format!("({} {})", tag, value)
    }

    fn expect_builder_nil() -> String {
        format!("(nil)")
    }

    fn expect_scalar(value: &str) -> String {
        value.to_string()
    }

    #[test]
    fn test_serialize_var_declaration() {
        test_serialize(
            new_var_decl_stmt("id", new_number_literal(0)),
            expect_builder(
                "var-decl",
                vec![
                    expect_scalar("id"),
                    expect_builder_literal("number", 0.to_string()),
                ],
            ),
        );
    }

    fn test_serialize_if_statement_without_else() {
        test_serialize(
            new_if_else_stmt(
                new_boolean_literal(true),
                new_print_stmt(new_number_literal(0)),
                None,
            ),
            expect_builder(
                "if",
                vec![expect_builder(
                    "print",
                    vec![expect_builder_literal("number", 0.to_string())],
                )],
            ),
        )
    }
}
