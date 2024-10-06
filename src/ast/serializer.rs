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

/// Serializes an AST to a string.
///
/// The `include_location` parameter controls if location information (line, column) gets printed
/// for nodes of the AST.
pub fn serialize(stmt: &Stmt, include_location: bool) -> String {
    let ser = AstTopologicalSerializer::new(AstSerializerOptions { include_location }, 0);
    ser.serialize(stmt)
}

#[derive(Clone)]
struct AstSerializerOptions {
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
    indent: i64,
}

impl AstTopologicalSerializer {
    const INDENT_NEXT_LEVEL: i64 = 2;

    fn new(options: AstSerializerOptions, indent: i64) -> Self {
        Self { options, indent }
    }

    fn new_inner_serializer(&self) -> Self {
        Self {
            options: self.options.clone(),
            indent: Self::INDENT_NEXT_LEVEL,
        }
    }

    /// Serializes the AST into a string.
    fn serialize(&self, stmt: &Stmt) -> String {
        self.serialize_stmt(stmt).join("\n")
    }

    /// Indents the input by the configured `indent` value.
    fn get_indented(&self, input: String) -> String {
        let indent = std::iter::repeat(" ")
            .take(self.indent as usize)
            .collect::<String>();
        format!("{}{}", indent, input)
    }

    /// Constructs the location-related string dependeing on the `include_location` option.
    fn get_loc_string(&self, loc: &LocationSpan) -> String {
        let mut loc_string = String::new();
        if self.options.include_location {
            loc_string = format!(" [{}]", loc.to_string());
        }
        loc_string
    }

    /// Generates a generic output for an AST node based on the (nested) statements.
    ///
    /// This function generates this output (the location information `[loc]` is optionally):
    ///
    /// ```text
    ///   (<tag> [loc]
    ///     (sub-statement-1
    ///
    ///     )
    ///     ...
    ///     (sub-statement-N
    ///
    ///     )
    ///   )
    /// ```
    fn generate_ast_serialization(
        &self,
        tag: &str,
        loc: &LocationSpan,
        sub_statements: Vec<Vec<String>>,
    ) -> Vec<String> {
        let mut content = vec![self.get_indented(format!("({}{}", tag, self.get_loc_string(loc)))];
        for sub_stmt in sub_statements {
            for line in sub_stmt {
                content.push(self.get_indented(format!("{}", line)));
            }
        }
        content.push(self.get_indented(format!(")")));
        content
    }

    /// Generates a generic output for an AST literal.
    ///
    /// This function generates the output (the location information `loc` is optionally):
    ///
    /// ```text
    ///   (<tag>) [loc]
    /// ```
    ///
    /// in case the value is not provided, or:
    ///
    /// ```text
    ///   (<tag> <value>) [loc]
    /// ```
    ///
    /// otherwise.
    fn generate_ast_serialization_for_literal(
        &self,
        tag: &str,
        loc: &LocationSpan,
        value: Option<String>,
    ) -> Vec<String> {
        let mut value_string = String::new();
        if let Some(value) = value {
            value_string = format!(" {}", value);
        }
        vec![self.get_indented(format!(
            "({}{}){}",
            tag,
            value_string,
            self.get_loc_string(loc)
        ))]
    }

    fn serialize_stmt(&self, stmt: &Stmt) -> Vec<String> {
        // statements and expressions on the inner level get serialized by this instance
        let sub_ser = self.new_inner_serializer();
        let loc = stmt.get_loc();
        match &stmt.data {
            StmtData::Block { statements } => self.serialize_stmt_list("block", loc, statements),
            StmtData::Expr { expr } => {
                self.generate_ast_serialization("expr", loc, vec![sub_ser.serialize_expr(expr)])
            }
            StmtData::If {
                condition,
                if_statement,
                else_statement,
            } => {
                let mut sub_entries = vec![
                    sub_ser.serialize_expr(condition),
                    sub_ser.serialize_stmt(if_statement),
                ];
                if let Some(else_statement) = else_statement {
                    sub_entries.push(sub_ser.serialize_stmt(else_statement));
                }
                self.generate_ast_serialization("if", loc, sub_entries)
            }
            StmtData::List { statements } => self.serialize_stmt_list("list", loc, statements),
            StmtData::Print { expr } => {
                self.generate_ast_serialization("print", loc, vec![sub_ser.serialize_expr(expr)])
            }
            StmtData::VarDecl {
                identifier,
                init_expr,
            } => self.generate_ast_serialization(
                "var-decl",
                loc,
                vec![
                    sub_ser.serialize_name(identifier),
                    sub_ser.serialize_expr(init_expr),
                ],
            ),
            StmtData::While { condition, body } => self.generate_ast_serialization(
                "while",
                loc,
                vec![
                    sub_ser.serialize_expr(condition),
                    sub_ser.serialize_stmt(body),
                ],
            ),
        }
    }

    fn serialize_stmt_list(
        &self,
        tag: &str,
        loc: &LocationSpan,
        statements: &Vec<Stmt>,
    ) -> Vec<String> {
        let sub_ser = self.new_inner_serializer();
        let serialized_statements: Vec<Vec<String>> = statements
            .iter()
            .map(|stmt| sub_ser.serialize_stmt(stmt))
            .collect();
        self.generate_ast_serialization(tag, loc, serialized_statements)
    }

    fn serialize_expr(&self, expr: &Expr) -> Vec<String> {
        // statements and expressions on the inner level get serialized by this instance
        let sub_ser = self.new_inner_serializer();
        let loc = expr.get_loc();
        match &expr.data {
            ExprData::Binary { lhs, op, rhs } => self.generate_ast_serialization(
                &op.to_string(),
                loc,
                vec![sub_ser.serialize_expr(lhs), sub_ser.serialize_expr(rhs)],
            ),
            ExprData::Unary { op, expr } => self.generate_ast_serialization(
                &op.to_string(),
                loc,
                vec![sub_ser.serialize_expr(expr)],
            ),
            ExprData::Grouping { expr } => {
                self.generate_ast_serialization("group", loc, vec![sub_ser.serialize_expr(expr)])
            }
            ExprData::Variable { name } => {
                self.generate_ast_serialization_for_literal("var", loc, Some(name.clone()))
            }
            ExprData::Assign { name, expr } => self.generate_ast_serialization(
                "assign",
                loc,
                vec![sub_ser.serialize_name(name), sub_ser.serialize_expr(expr)],
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

    fn serialize_name(&self, name: &str) -> Vec<String> {
        vec![self.get_indented(String::from(name))]
    }
}

#[cfg(test)]
pub mod tests {
    use crate::ast::{
        tests::{
            new_assign_expr, new_binary_expr, new_block_stmt, new_boolean_literal_expr,
            new_expr_stmt, new_grouping_expr, new_if_else_stmt, new_if_stmt, new_literal_expr,
            new_number_literal_expr, new_print_stmt, new_string_literal_expr, new_unary_expr,
            new_var_decl_stmt, new_variable_expr, new_while_stmt,
        },
        BinaryOperator, Expr, Literal, Stmt, UnaryOperator,
    };
    use strum::IntoEnumIterator;

    use super::{serialize, AstSerializerOptions, AstTopologicalSerializer};

    /// Serializes an expression to a list of strings with each entry representing one line of the
    /// serialized output.
    ///
    /// Note: this function is used for testing purpose only to show the diff between two serializations
    /// in a more developer-friendly ways.
    pub fn serialize_expr(expr: &Expr) -> Vec<String> {
        let ser = AstTopologicalSerializer::new(
            AstSerializerOptions {
                include_location: false,
            },
            0,
        );
        ser.serialize_expr(expr)
    }

    /// Serializes a statement to a list of strings with each entry representing one line of the
    /// serialized output.
    ///
    /// Note: this function is used for testing purpose only to show the diff between two serializations
    /// in a more developer-friendly ways.
    pub fn serialize_stmt(stmt: &Stmt) -> Vec<String> {
        let ser = AstTopologicalSerializer::new(
            AstSerializerOptions {
                include_location: false,
            },
            0,
        );
        ser.serialize_stmt(stmt)
    }

    fn test_serialize(stmt: Stmt, expected: String) {
        let output = serialize(&stmt, false);
        assert_eq!(output, expected);
    }

    fn dedent(input: &str) -> String {
        // skip the empty lines at the beginning
        let dedented_input = textwrap::dedent(input);
        let lines: Vec<String> = dedented_input
            .split('\n')
            .skip_while(|line| line.len() == 0)
            .map(|line| String::from(line))
            .collect();
        dbg!(&lines);
        lines.join("\n").trim_end().into()
    }

    #[test]
    fn test_stmt_var_declaration() {
        test_serialize(
            new_var_decl_stmt("id", new_number_literal_expr(0)),
            dedent(
                r#"
                (var-decl
                  id
                  (number 0)
                )
            "#,
            ),
        );
    }

    #[test]
    fn test_stmt_if() {
        test_serialize(
            new_if_stmt(
                new_boolean_literal_expr(true),
                new_print_stmt(new_number_literal_expr(0)),
            ),
            dedent(
                r#"
                (if
                  (bool true)
                  (print
                    (number 0)
                  )
                )
                "#,
            ),
        );
    }

    #[test]
    fn test_stmt_if_else() {
        test_serialize(
            new_if_else_stmt(
                new_boolean_literal_expr(false),
                new_expr_stmt(new_assign_expr("id", new_number_literal_expr(0))),
                new_print_stmt(new_string_literal_expr("value")),
            ),
            dedent(
                r#"
                (if
                  (bool false)
                  (expr
                    (assign
                      id
                      (number 0)
                    )
                  )
                  (print
                    (string value)
                  )
                )
                "#,
            ),
        );
    }

    #[test]
    fn test_stmt_while() {
        test_serialize(
            new_while_stmt(
                new_binary_expr(
                    BinaryOperator::LessThan,
                    new_binary_expr(
                        BinaryOperator::Add,
                        new_variable_expr("id"),
                        new_number_literal_expr(1),
                    ),
                    new_number_literal_expr(10),
                ),
                new_block_stmt(vec![new_expr_stmt(new_assign_expr(
                    "id",
                    new_binary_expr(
                        BinaryOperator::Multiply,
                        new_variable_expr("id"),
                        new_number_literal_expr(2),
                    ),
                ))]),
            ),
            dedent(
                r#"
                (while
                  (<
                    (+
                      (var id)
                      (number 1)
                    )
                    (number 10)
                  )
                  (block
                    (expr
                      (assign
                        id
                        (*
                          (var id)
                          (number 2)
                        )
                      )
                    )
                  )
                )
                "#,
            ),
        );
    }

    #[test]
    fn test_expr_binary() {
        for op in BinaryOperator::iter() {
            let ast = new_expr_stmt(new_binary_expr(
                op,
                new_number_literal_expr(0),
                new_number_literal_expr(1),
            ));
            let expected_output = dedent(&format!(
                r#"
                (expr
                  ({}
                    (number 0)
                    (number 1)
                  )
                )
                "#,
                op.to_string()
            ));
            test_serialize(ast, expected_output);
        }
    }

    #[test]
    fn test_expr_unary() {
        for op in UnaryOperator::iter() {
            test_serialize(
                new_expr_stmt(new_unary_expr(op, new_number_literal_expr(0))),
                dedent(&format!(
                    r#"
                    (expr
                      ({}
                        (number 0)
                      )
                    )
                    "#,
                    op.to_string()
                )),
            );
        }
    }

    #[test]
    fn test_expr_grouping() {
        test_serialize(
            new_expr_stmt(new_grouping_expr(new_binary_expr(
                BinaryOperator::Add,
                new_number_literal_expr(0),
                new_number_literal_expr(1),
            ))),
            dedent(
                r#"
                (expr
                  (group
                    (+
                      (number 0)
                      (number 1)
                    )
                  )
                )
                "#,
            ),
        );
    }

    #[test]
    fn test_with_locations() {
        let ast = new_block_stmt(vec![
            new_var_decl_stmt("id", new_literal_expr(Literal::Nil)),
            new_expr_stmt(new_assign_expr(
                "id",
                new_binary_expr(
                    BinaryOperator::Divide,
                    new_number_literal_expr(10),
                    new_number_literal_expr(5),
                ),
            )),
        ]);
        let expected_output = dedent(
            r#"
            (block [1:1-1:1]
              (var-decl [1:1-1:1]
                id
                (nil) [1:1-1:1]
              )
              (expr [1:1-1:1]
                (assign [1:1-1:1]
                  id
                  (/ [1:1-1:1]
                    (number 10) [1:1-1:1]
                    (number 5) [1:1-1:1]
                  )
                )
              )
            )
            "#,
        );
        assert_eq!(serialize(&ast, true), expected_output);
    }
}
