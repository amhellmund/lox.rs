// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
struct Args {
    file_path: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    print_ast: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(file_path) = args.file_path {
        println!("File argument is {}", file_path.display());
        lox::execute(&file_path, args.print_ast)
    } else {
        lox::repl()
    }
}
