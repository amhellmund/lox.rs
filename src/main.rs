use std::path::PathBuf;

use clap::Parser;
use anyhow::Result;

#[derive(Parser)]
struct Args {
    file_path: PathBuf,
}

fn main () -> Result<()>  {
    let args = Args::parse();
    println!("File argument is {}", args.file_path.display());
    lox::execute(&args.file_path)?;
    Ok(())
}