// Copyright (c) 2024 Andi Hellmund. All rights reserved.
//
// This work is licensed under the terms of the BSD-3-Clause license.
// For a copy, see <https://opensource.org/license/bsd-3-clause>.

//! Environment of interpreter.
//!
//! The environment module encapsulates the management of:
//!
//!   o Variables

use std::collections::HashMap;

use super::ExprValue;

pub struct ExecutionEnvironment {
    variables: HashMap<String, ExprValue>,
}

impl ExecutionEnvironment {
    pub fn new() -> Self {
        ExecutionEnvironment {
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

#[cfg(test)]
mod tests {
    use super::ExecutionEnvironment;
    use crate::ast::eval::ExprValue;

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
    fn test_has_variable() {
        let mut env = ExecutionEnvironment::new();
        assert_eq!(env.has_variable("name"), false);
        env.define_variable("name", ExprValue::Nil);
        assert_eq!(env.has_variable("name"), true);
    }
}
