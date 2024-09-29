// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

//! A library for scanning, parsing and evaluating the [`Lox`] programming language.
//!
//! [`Lox`]: https://craftinginterpreters.com/the-lox-language.html

use anyhow::{Context, Result};
use ast::eval::eval_expr;
use ast::printer::print_ast;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{fs, io};

mod ast;
mod diagnostics;
mod parser;
mod scanner;

/// Executes a Lox file.
///
/// The parameter `show_ast` indicates if the abstract syntax tree (AST) shall be shown.
///
/// The returned error is of type `anyhow::Error` and contains the details about the location
/// in the code as well as the root cause for the error.
pub fn execute(file_path: &Path, show_ast: bool) -> Result<()> {
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file {}", file_path.display()))?;

    let tokens = scanner::tokenize(&content, file_path.to_path_buf())?;
    let ast = parser::parse(tokens, file_path.to_path_buf())
        .with_context(|| "Failed to parse the input file")?;
    let expr_value = eval_expr(&ast, file_path.to_path_buf())?;

    if show_ast {
        let ast_as_string = print_ast(&ast);
        println!("AST:\n{}", ast_as_string);
    }

    println!("Expr: {}", expr_value.to_string());
    Ok(())
}

/// Starts a Lox REPL shell to interactively run statements and expressions.
///
/// Currently, only expressions are supported by the implementation.
pub fn repl() -> Result<()> {
    loop {
        print!("lox> ");
        io::stdout().flush()?;

        let mut content = String::new();
        io::stdin().read_line(&mut content)?;
        if is_exit_command(&content) {
            break;
        }

        if let Err(err) = repl_impl(content) {
            println!("Error: {}", err)
        }
    }
    Ok(())
}

fn repl_impl(content: String) -> Result<()> {
    let file_path = PathBuf::from("repl");
    let tokens = scanner::tokenize(&content, file_path.to_path_buf())?;
    let ast_expr = parser::parse(tokens, file_path.to_path_buf())?;
    let expr_value = eval_expr(&ast_expr, file_path.to_path_buf())?;
    println!("Result: {}", expr_value.to_string());
    Ok(())
}

fn is_exit_command(content: &str) -> bool {
    content.starts_with("exit")
        && content[4..]
            .chars()
            .into_iter()
            .all(|ch| ch.is_whitespace())
}
