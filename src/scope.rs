use std::collections::HashMap;

use crate::syntax::ast::Identifier;

pub struct NoSuchVariable {}

/// Keeps track of the current variable scope, mapping identifiers to `T` in the scope.
pub struct Scope<T> {
    /// A mapping of variables in all scopes by their identifiers to their types. The last element
    /// in the vector is the innermost scope.
    variables: Vec<HashMap<Identifier, T>>,
}

impl<T> Scope<T> {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
        }
    }

    pub fn lookup_variable(&self, identifier: &Identifier) -> Option<&T> {
        self.variables
            .iter()
            .rev()
            .filter_map(|level| level.get(identifier))
            .next()
    }

    pub fn create_variable(&mut self, identifier: Identifier, value: T) {
        self.variables
            .iter_mut()
            .next_back()
            .expect("there is always at least one scope")
            .insert(identifier, value);
    }

    pub fn assign_variable(
        &mut self,
        identifier: Identifier,
        value: T,
    ) -> Result<(), NoSuchVariable> {
        *self
            .variables
            .iter_mut()
            .rev()
            .filter_map(|level| level.get_mut(&identifier))
            .next()
            .ok_or(NoSuchVariable {})? = value;

        Ok(())
    }

    pub fn new_scope(&mut self) {
        self.variables.push(HashMap::new());
    }

    pub fn remove_scope(&mut self) {
        assert!(!self.variables.is_empty());

        self.variables.pop();
    }
}
