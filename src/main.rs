use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
struct Args {
    file_path: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(file_path) = args.file_path {
        println!("File argument is {}", file_path.display());
        lox::execute(&file_path, false)
    } else {
        lox::repl()
    }
}
