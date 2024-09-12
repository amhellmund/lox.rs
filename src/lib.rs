use anyhow::{Context, Result};
use ast::printer::print_ast;
use std::fs;
use std::path::Path;

mod ast;
mod diagnostics;
mod parser;
mod scanner;

pub fn execute(file_path: &Path) -> Result<()> {
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file {}", file_path.display()))?;

    let tokens = scanner::tokenize(&content, file_path.to_path_buf())?;
    let ast = parser::parse(&tokens, file_path.to_path_buf())
        .with_context(|| "Failed to parse the input file")?;
    let ast_as_string = print_ast(&ast);
    println!("AST:\n{}", ast_as_string);
    Ok(())
}
