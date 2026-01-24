use std::{collections::HashMap, str::FromStr};

use crate::syntax::ast::{
    Ast, BinaryOperator, Expression, Identifier, Module, Node, NodeId, Primary, UnaryOperator,
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum TypErrorKind {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Typ {
    Int,
    Bool,
    Unit,
    // TODO: make clone great again
    Function { params: Vec<Typ>, ret: Box<Typ> },
}

#[derive(Debug, Clone)]
pub struct TypMap {
    pub typs: HashMap<NodeId, Typ>,
}

impl FromStr for Typ {
    // TODO: add an error type
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Int" => Ok(Typ::Int),
            "Bool" => Ok(Typ::Bool),
            _ => Err(()),
        }
    }
}

struct Environment {
    /// A mapping of variables in all scopes by their identifiers to their types. The last element
    /// in the vector is the innermost scope.
    variables: Vec<HashMap<Identifier, Typ>>,
    typmap: TypMap,
}

impl Environment {
    fn new() -> Self {
        Self {
            variables: vec![HashMap::from([(
                Identifier(String::from("print_int")),
                Typ::Int,
            )])],
            typmap: TypMap {
                typs: HashMap::new(),
            },
        }
    }

    fn lookup_variable(&self, identifier: &Identifier) -> Option<&Typ> {
        self.variables
            .iter()
            .rev()
            .filter_map(|level| level.get(identifier))
            .next()
    }

    fn set_variable_typ(&mut self, identifier: Identifier, typ: Typ) {
        self.variables
            .iter_mut()
            .next_back()
            .expect("there is always at least one scope")
            .insert(identifier, typ);
    }

    fn with_new_scope<T>(&mut self, f: impl FnOnce(&mut Environment) -> T) -> T {
        self.new_scope();
        let ret = f(self);
        self.remove_scope();

        ret
    }

    fn new_scope(&mut self) {
        self.variables.push(HashMap::new());
    }

    fn remove_scope(&mut self) {
        assert!(!self.variables.is_empty());

        self.variables.pop();
    }
}

fn typecheck(ast: &Ast) -> Result<TypMap, TypErrorKind> {
    let mut env = Environment::new();
    typecheck_module(&mut env, &ast.root)?;

    for id in (0..ast.nodes).map(NodeId) {
        assert!(env.typmap.typs.contains_key(&id));
    }

    Ok(env.typmap)
}

fn typecheck_module(env: &mut Environment, module: &Module) -> Result<Typ, TypErrorKind> {
    typecheck_expression(env, &module.body)
}

fn typecheck_expression(
    env: &mut Environment,
    expression: &Node<Expression>,
) -> Result<Typ, TypErrorKind> {
    let typ = match &expression.item {
        Expression::Binary(binary_expression) => match binary_expression.operator {
            BinaryOperator::Or
            | BinaryOperator::And
            | BinaryOperator::EqualEqual
            | BinaryOperator::NotEqual
            | BinaryOperator::LessThan
            | BinaryOperator::LessEqual
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterEqual => {
                assert_eq!(
                    typecheck_expression(env, &binary_expression.lhs)?,
                    typecheck_expression(env, &binary_expression.rhs)?
                );

                Typ::Unit
            }
            BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo => {
                assert_eq!(
                    typecheck_expression(env, &binary_expression.lhs)?,
                    typecheck_expression(env, &binary_expression.rhs)?
                );

                Typ::Int
            }
            BinaryOperator::Equals => {
                let tgt_typ = typecheck_expression(env, &binary_expression.lhs)?;
                assert_eq!(tgt_typ, typecheck_expression(env, &binary_expression.rhs)?);

                tgt_typ
            }
        },
        Expression::Unary(unary_expression) => match unary_expression.operator {
            UnaryOperator::Not => {
                assert_eq!(
                    typecheck_expression(env, &unary_expression.operand)?,
                    Typ::Bool
                );

                Typ::Bool
            }
            UnaryOperator::Negate => {
                assert_eq!(
                    typecheck_expression(env, &unary_expression.operand)?,
                    Typ::Int
                );

                Typ::Int
            }
        },
        Expression::Primary(primary) => match primary {
            Primary::Bool(_) => Typ::Bool,
            Primary::Integer(_) => Typ::Int,
            // FIXME: unwrap
            Primary::Identifier(identifier) => env.lookup_variable(identifier).unwrap().clone(),
        },
        Expression::If(if_expression) => {
            assert_eq!(
                typecheck_expression(env, &if_expression.condition)?,
                Typ::Bool
            );

            match &if_expression.els {
                Some(els_expression) => {
                    let then_typ = typecheck_expression(env, &if_expression.then)?;
                    let els_typ = typecheck_expression(env, els_expression)?;
                    assert_eq!(then_typ, els_typ);

                    then_typ
                }
                None => {
                    typecheck_expression(env, &if_expression.then)?;

                    Typ::Unit
                }
            }
        }
        Expression::While(while_expression) => {
            assert_eq!(
                typecheck_expression(env, &while_expression.condition)?,
                Typ::Bool
            );

            typecheck_expression(env, &while_expression.body)?
        }
        Expression::Call(call_expression) => {
            let f_typ = typecheck_expression(env, &call_expression.function)?;

            match f_typ {
                Typ::Function { params, ret } => {
                    assert_eq!(params.len(), call_expression.args.len());

                    for (expected, expr) in params.iter().zip(&call_expression.args) {
                        assert_eq!(*expected, typecheck_expression(env, expr)?);
                    }

                    *ret
                }
                _ => todo!(),
            }
        }
        Expression::Block(block_expression) => env.with_new_scope(|env| {
            for expr in &block_expression.expressions {
                typecheck_expression(env, expr)?;
            }

            match &block_expression.result_expression {
                Some(expr) => typecheck_expression(env, expr),
                None => Ok(Typ::Unit),
            }
        })?,
        Expression::Var(var_expression) => {
            let typ = typecheck_expression(env, &var_expression.value)?;

            if let Some(expected_typ) = var_expression.typ.as_ref() {
                let expected_typ = Typ::from_str(&expected_typ.0).unwrap();
                assert_eq!(typ, expected_typ);
            }

            env.set_variable_typ(var_expression.name.clone(), typ);

            Typ::Unit
        }
    };

    assert!(!env.typmap.typs.contains_key(&expression.id));
    env.typmap.typs.insert(expression.id, typ.clone());

    Ok(typ)
}

#[cfg(test)]
mod tests {
    use crate::syntax::parse;

    use super::*;

    fn str_to_typemap(code: &str) -> Result<TypMap, TypErrorKind> {
        let ast = parse(code).0.unwrap();
        typecheck(&ast)
    }

    fn module_to_typ(code: &str) -> Result<Typ, TypErrorKind> {
        let ast = parse(code).0.unwrap();
        Ok(typecheck(&ast)?
            .typs
            .get(&ast.root.body.id)
            .unwrap()
            .clone())
    }

    #[allow(non_snake_case)]
    fn N(id: u32) -> NodeId {
        NodeId(id)
    }

    #[test]
    fn test_simple() {
        assert_eq!(
            str_to_typemap("").unwrap().typs,
            HashMap::from([(N(0), Typ::Unit)])
        );

        assert_eq!(module_to_typ("1+1").unwrap(), Typ::Int);
        assert_eq!(module_to_typ("not not true").unwrap(), Typ::Bool);
        assert_eq!(module_to_typ("if true then 1").unwrap(), Typ::Unit);
        assert_eq!(module_to_typ("if true then 1 else 2").unwrap(), Typ::Int);
    }
}
