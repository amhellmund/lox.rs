// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    PrintAst {
        #[arg(required = true)]
        file_path: PathBuf,
        #[arg(long, default_value_t = false)]
        show_location: bool,
    },
    Repl,
    Run {
        #[arg(required = true)]
        file_path: PathBuf,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Commands::PrintAst {
            file_path,
            show_location,
        } => lox::print_ast(&file_path, show_location),
        Commands::Repl => lox::repl(),
        Commands::Run { file_path } => lox::execute(&file_path, &mut std::io::stdout()),
    }
}
