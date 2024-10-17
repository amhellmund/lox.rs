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

use std::{cell::RefCell, collections::HashMap, rc::Rc};

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
    parent: Option<McScopeRef>,
}

type McScopeRef = Rc<RefCell<Scope>>;

impl Scope {
    fn new(parent: Option<McScopeRef>) -> Self {
        Self {
            variables: HashMap::new(),
            parent,
        }
    }

    pub fn as_rc_ref_cell(self) -> McScopeRef {
        Rc::new(RefCell::new(self))
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
    global_scope: McScopeRef,
    current_scope: McScopeRef,
}

impl ExecutionEnvironment {
    pub fn new() -> Self {
        let global_scope = Scope::new(None).as_rc_ref_cell();
        ExecutionEnvironment {
            global_scope: Rc::clone(&global_scope),
            current_scope: Rc::clone(&global_scope),
        }
    }

    fn new_scope_with_parent(&self) -> McScopeRef {
        Scope::new(Some(Rc::clone(&self.current_scope))).as_rc_ref_cell()
    }

    /// Clones the current environment while keeping the references active.
    ///
    /// This function is used for functions that share parent scopes.
    /// It basically branches off from the current chain.
    pub fn clone_with_keeping_scopes(&self) -> Self {
        ExecutionEnvironment {
            global_scope: Rc::clone(&self.global_scope),
            current_scope: Rc::clone(&self.current_scope),
        }
    }

    /// Creates a new inner-most scope.
    ///
    /// Creating a new scope always succeeds (unless OS limits are hit).
    pub fn create_lexical_scope(&mut self) {
        self.current_scope = self.new_scope_with_parent();
    }

    /// Drops the inner-most scope.
    ///
    /// In case the last inner scope is reached, the `current_scope` becomes the `global_scope` again.
    pub fn drop_innermost_lexical_scope(&mut self) {
        let parent_scope = match self.current_scope.borrow().parent.as_ref() {
            Some(parent) => Rc::clone(parent),
            None => Rc::clone(&self.global_scope),
        };
        self.current_scope = parent_scope;
    }

    /// Defines a new variable within the inner-most scope.
    pub fn define_variable(&mut self, name: &str, value: ExprValue) {
        self.current_scope
            .borrow_mut()
            .define_variable(name.into(), value);
    }

    /// Assigns a value to an existing variable.
    ///
    /// Returns true if the value could get assigned, or false if the variable does
    /// not exist in any scope.
    pub fn assign_variable(&mut self, name: &str, value: ExprValue) -> bool {
        let mut cur_scope = Rc::clone(&self.current_scope);
        loop {
            if cur_scope.borrow().has_variable(name) {
                cur_scope.borrow_mut().define_variable(name, value);
                return true;
            } else {
                let parent_scope = match cur_scope.borrow().parent.as_ref() {
                    Some(parent) => Rc::clone(parent),
                    None => return false,
                };
                cur_scope = Rc::clone(&parent_scope);
            }
        }
    }

    /// Returns the variable value from any scope along the lexical scope chain.
    ///
    /// Returns None if the variable cannot be found in any scope.
    pub fn get_variable(&self, name: &str) -> Option<ExprValue> {
        let mut cur_scope = Rc::clone(&self.current_scope);
        loop {
            let var = cur_scope.borrow().get_variable(name);
            if var.is_some() {
                return var;
            } else {
                let parent_scope = match cur_scope.borrow().parent.as_ref() {
                    Some(parent) => Rc::clone(parent),
                    None => return None,
                };
                cur_scope = Rc::clone(&parent_scope);
            }
        }
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
        let _1 = env.create_lexical_scope();
        let _2 = env.create_lexical_scope();
        env.assign_variable("name", ExprValue::Number(2.0));
        env.drop_innermost_lexical_scope();
        env.drop_innermost_lexical_scope();

        assert_eq!(env.get_variable("name").unwrap(), ExprValue::Number(2.0));
    }

    #[test]
    fn test_define_variable_in_new_scope_drop_scope() {
        let mut env = ExecutionEnvironment::new();
        env.define_variable("name", ExprValue::Number(1.0));

        {
            env.create_lexical_scope();
            env.define_variable("name", ExprValue::String("string".into()));
            let inner_value = env.get_variable("name").unwrap();
            assert_eq!(inner_value, ExprValue::String(String::from("string")));
            env.drop_innermost_lexical_scope();
        }

        let outer_value = env.get_variable("name").unwrap();
        assert_eq!(outer_value, ExprValue::Number(1.0));
    }

    #[test]
    fn test_clone_with_keeping_scopes() {
        let mut env_first = ExecutionEnvironment::new();
        env_first.define_variable("name", ExprValue::Number(1.0));

        env_first.create_lexical_scope();
        env_first.define_variable("name1", ExprValue::String("string".into()));

        let mut env_second = env_first.clone_with_keeping_scopes();
        env_second.create_lexical_scope();
        env_second.define_variable("name1", ExprValue::Boolean(true));

        env_second.assign_variable("name", ExprValue::Nil);

        assert_eq!(
            env_first.get_variable("name1"),
            Some(ExprValue::String("string".into()))
        );

        assert_eq!(
            env_second.get_variable("name1"),
            Some(ExprValue::Boolean(true))
        );

        assert_eq!(env_first.get_variable("name"), Some(ExprValue::Nil));
        assert_eq!(env_second.get_variable("name"), Some(ExprValue::Nil));
    }
}
