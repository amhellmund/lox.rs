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

pub fn execute(file_path: &Path, show_ast: bool) -> Result<()> {
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file {}", file_path.display()))?;

    let tokens = scanner::tokenize(&content, file_path.to_path_buf())?;
    let ast = parser::parse(&tokens, file_path.to_path_buf())
        .with_context(|| "Failed to parse the input file")?;
    let expr_value = eval_expr(&ast, file_path.to_path_buf())?;

    if show_ast {
        let ast_as_string = print_ast(&ast);
        println!("AST:\n{}", ast_as_string);
    }

    println!("Expr: {}", expr_value.to_string());
    Ok(())
}

pub fn repl() -> Result<()> {
    loop {
        print!("lox> ");
        io::stdout().flush()?;

        let mut content = String::new();
        io::stdin().read_line(&mut content)?;
        if is_exit_command(&content) {
            break;
        }

        let file_path = PathBuf::from("repl");
        let tokens = scanner::tokenize(&content, file_path.to_path_buf())?;
        let ast = parser::parse(&tokens, file_path.to_path_buf())
            .with_context(|| "Failed to parse the input file")?;
        let expr_value = eval_expr(&ast, file_path.to_path_buf())?;

        println!("Result: {}", expr_value.to_string());
    }
    Ok(())
}

fn is_exit_command(content: &str) -> bool {
    content.starts_with("exit")
        && content[4..]
            .chars()
            .into_iter()
            .all(|ch| ch.is_whitespace())
}
