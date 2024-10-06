// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

//! Environment of interpreter.
//!
//! The environment module encapsulates the management of:
//!
//!   o Variables
//!   o Scopes

use std::collections::HashMap;

use super::ExprValue;

/// Scope.
///
/// Scopes are defined hierarchically to model the language semantics that
///
///   o Variables defined in an inner scope are only 'seen' within this scope.
///   o Variables redefind in an inner scope shadow variables defined in an outer scope.
///   o Variables defined in an outer scope are still visible in an inner scope.
struct Scope {
    variables: HashMap<String, ExprValue>,
}

impl Scope {
    fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    pub fn define_variable(&mut self, name: &str, value: ExprValue) {
        self.variables.insert(name.into(), value);
    }

    pub fn get_variable(&self, name: &str) -> Option<ExprValue> {
        self.variables.get(name).cloned()
    }

    pub fn has_variable(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }
}

/// Execution Environment.
///
/// The execution environment manages the state for the interpreter. In specific, it
///
///   o handles the entrance and exit of scopes.
///   o defines variables in the current inner-most scope.
///   o gets variables' values from one of the defined scopes.
pub struct ExecutionEnvironment {
    scopes: Vec<Scope>,
}

impl ExecutionEnvironment {
    pub fn new() -> Self {
        ExecutionEnvironment {
            scopes: vec![Scope::new()],
        }
    }

    /// Creates a new inner-most scope.
    ///
    /// Creating a new scope always succeeds (unless OS limits are hit).
    pub fn create_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    /// Drops the inner-most scope.
    ///
    /// Deleting an inner-most scope is not possible if only one scope
    /// (the global scope) is available.
    ///
    /// Post-Condition: at least one scope exists.
    pub fn drop_innermost_scope(&mut self) -> bool {
        if self.scopes.len() > 1 {
            self.scopes.pop();
            assert!(self.scopes.len() >= 1);
            true
        } else {
            false
        }
    }

    /// Defines a new variable within the inner-most scope.
    pub fn define_variable(&mut self, name: &str, value: ExprValue) {
        let scope = self.scopes.last_mut().unwrap();
        scope.define_variable(name.into(), value);
    }

    /// Assigns a value to an existing variable.
    ///
    /// Returns true if the value could get assigned, or false if the variable does
    /// not exist in any scope.
    pub fn assign_variable(&mut self, name: &str, value: ExprValue) -> bool {
        for i in (0..self.scopes.len()).rev() {
            if self.scopes[i].has_variable(name) {
                self.scopes[i].define_variable(name, value);
                return true;
            }
        }
        false
    }

    /// Returns the variable value from any scope along the lexical scope chain.
    ///
    /// Returns None if the variable cannot be found in any scope.
    pub fn get_variable(&self, name: &str) -> Option<ExprValue> {
        for i in (0..self.scopes.len()).rev() {
            let scope_variable = self.scopes[i].get_variable(name);
            if scope_variable.is_some() {
                return scope_variable;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::ExecutionEnvironment;
    use crate::ast::interpreter::ExprValue;

    #[test]
    fn test_add_read_variable() {
        let mut env = ExecutionEnvironment::new();
        env.define_variable("name", ExprValue::Number(1.0));
        assert!(matches!(
            env.get_variable("name"),
            Some(ExprValue::Number(1.0))
        ));
    }

    #[test]
    fn test_read_nonexistent_variable() {
        let env = ExecutionEnvironment::new();
        assert!(matches!(env.get_variable("name"), None));
    }

    #[test]
    fn test_assign_variable() {
        let mut env = ExecutionEnvironment::new();
        env.define_variable("name", ExprValue::Number(1.0));
        let _1 = env.create_scope();
        let _2 = env.create_scope();
        env.assign_variable("name", ExprValue::Number(2.0));
        env.drop_innermost_scope();
        env.drop_innermost_scope();

        assert_eq!(env.get_variable("name").unwrap(), ExprValue::Number(2.0));
    }

    #[test]
    fn test_define_variable_in_new_scope_drop_scope() {
        let mut env = ExecutionEnvironment::new();
        env.define_variable("name", ExprValue::Number(1.0));

        {
            env.create_scope();
            env.define_variable("name", ExprValue::String("string".into()));
            let inner_value = env.get_variable("name").unwrap();
            assert_eq!(inner_value, ExprValue::String(String::from("string")));
            env.drop_innermost_scope();
        }

        let outer_value = env.get_variable("name").unwrap();
        assert_eq!(outer_value, ExprValue::Number(1.0));
    }
}
