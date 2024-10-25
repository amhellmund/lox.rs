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
pub trait Callable<'a, W: Write> {
    fn get_function_arity(&self) -> i64;
    fn call(
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
}

fn ffi_clock(_: Vec<ExprValue>) -> Result<ExprValue> {
    let elasped_seconds = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    return Ok(ExprValue::Number(elasped_seconds));
}

impl NativeFunction {
    pub fn get_name(&self) -> String {
        match self {
            NativeFunction::Clock => String::from("clock"),
        }
    }

    fn get_execution_info(&self) -> (i64, fn(Vec<ExprValue>) -> Result<ExprValue>) {
        match self {
            NativeFunction::Clock => (0, ffi_clock),
        }
    }
}

impl<'a, W: Write> Callable<'a, W> for NativeFunction {
    fn get_function_arity(&self) -> i64 {
        let (arity, _) = self.get_execution_info();
        arity
    }

    fn call(&self, arguments: Vec<ExprValue>, _: &mut Interpreter<'a, W>) -> Result<ExprValue> {
        let (_, callback) = self.get_execution_info();
        callback(arguments)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Function {
    name: String,
    arity: i64,
    block: Stmt,
    environment: ExecutionEnvironment,
}

impl Function {
    pub fn get_name(&self) -> String {
        self.name.clone()
    }
}

impl<'a, W: Write> Callable<'a, W> for Function {
    fn get_function_arity(&self) -> i64 {
        self.arity
    }

    fn call(&self, _: Vec<ExprValue>, interpreter: &mut Interpreter<'a, W>) -> Result<ExprValue> {
        let mut new_interpreter = interpreter.with_new_env(self.environment.clone());
        new_interpreter.interpret(&self.block);
        Ok(ExprValue::Nil)
    }
}
