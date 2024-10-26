// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

//! Utility classes and types for using functions
//!
//! Functions within Lox come in two flavors:
//!   o Regular Lox functions represented by a `Stmt`
//!   o Foreign/native functions (FFI)

use crate::ast::Stmt;
use anyhow::Result;

use super::{environment::ExecutionEnvironment, ExprValue, Interpreter};

use std::{io::Write, time::SystemTime};

/// Callable interface for all function types in Lox.
pub trait Callable {
    fn get_name(&self) -> String;
    fn get_function_arity(&self) -> i64;
    fn call<'a, W: Write>(
        &self,
        arguments: Vec<ExprValue>,
        interpreter: &mut Interpreter<'a, W>,
    ) -> Result<ExprValue>;
}

/// Foreign functions represent low-level functions provided by the
/// interepreter. Examples for foreign functions are time-related functions.
///
/// The design for the foreign functions interface is chosen to allow the
/// `ExprValue` to be clonable and comparable (by ==). Different approaches,
/// e.g. using function pointers, would have worked as well.
#[derive(PartialEq, Debug, Clone)]
pub enum NativeFunction {
    /// Provides the elapsed time in s since some arbitary point in time.
    Clock,
    /// Reads from stdin
    ReadStdIn,
}

fn ffi_clock(_: Vec<ExprValue>) -> Result<ExprValue> {
    let elasped_seconds = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    return Ok(ExprValue::Number(elasped_seconds));
}

fn ffi_read_std_in(_: Vec<ExprValue>) -> Result<ExprValue> {
    let mut buffer = String::new();
    std::io::stdin().read_line(&mut buffer)?;
    Ok(ExprValue::String(String::from(buffer.trim_end())))
}

impl NativeFunction {
    fn get_execution_info(&self) -> (i64, String, fn(Vec<ExprValue>) -> Result<ExprValue>) {
        match self {
            NativeFunction::Clock => (0, String::from("clock"), ffi_clock),
            NativeFunction::ReadStdIn => (0, String::from("read_stdin"), ffi_read_std_in),
        }
    }
}

impl Callable for NativeFunction {
    fn get_name(&self) -> String {
        let (_, name, _) = self.get_execution_info();
        return name;
    }

    fn get_function_arity(&self) -> i64 {
        let (arity, _, _) = self.get_execution_info();
        arity
    }

    fn call<'a, W: Write>(
        &self,
        arguments: Vec<ExprValue>,
        _: &mut Interpreter<'a, W>,
    ) -> Result<ExprValue> {
        assert_eq!(
            arguments.len(),
            self.get_function_arity() as usize,
            "Argument count mismatch"
        );
        let (_, _, callback) = self.get_execution_info();
        callback(arguments)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Function {
    name: String,
    parameters: Vec<String>,
    block: Vec<Stmt>,
    closure_environment: ExecutionEnvironment,
}

impl Function {
    pub fn new(
        name: &String,
        parameters: &Vec<String>,
        block: &Vec<Stmt>,
        closure_environment: &ExecutionEnvironment,
    ) -> Self {
        Function {
            name: name.clone(),
            parameters: parameters.clone(),
            block: block.clone(),
            closure_environment: closure_environment.clone(),
        }
    }
}

impl Callable for Function {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn get_function_arity(&self) -> i64 {
        self.parameters.len() as i64
    }

    fn call<'a, W: Write>(
        &self,
        arguments: Vec<ExprValue>,
        interpreter: &mut Interpreter<'a, W>,
    ) -> Result<ExprValue> {
        assert_eq!(
            arguments.len(),
            self.parameters.len(),
            "Argument count mismatch"
        );
        let mut new_environment = self.closure_environment.clone_with_keeping_scopes();
        new_environment.create_lexical_scope();
        for i in 0..arguments.len() {
            new_environment.define_variable(&self.parameters[i], arguments[i].clone());
        }
        let mut new_interpreter = interpreter.with_new_env(new_environment);
        new_interpreter.interpret_stmts(&self.block)?;
        // the new environment gets dropped and therefore also the innermost scope
        Ok(ExprValue::Nil)
    }
}
