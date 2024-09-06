use std::path::Path;
use std::fs;
use anyhow::{Context, Result};

mod ast;
mod diagnostics;
mod scanner

pub fn execute(file_path: &Path) -> Result<()> {
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file {}", file_path.display()))?;

    let tokens = scanner::tokenize(&content, file_path.to_path_buf())?;
    dbg!(tokens);
    Ok(())
}

