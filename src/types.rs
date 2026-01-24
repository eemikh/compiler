use std::{collections::HashMap, str::FromStr};

use crate::syntax::ast::{BinaryOperator, Expression, Identifier, Primary, UnaryOperator};

#[derive(Debug, Clone, PartialEq, Eq)]
enum TypErrorKind {}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Typ {
    Int,
    Bool,
    Unit,
    // TODO: make clone great again
    Function { params: Vec<Typ>, ret: Box<Typ> },
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
}

impl Environment {
    fn new() -> Self {
        Self {
            variables: vec![HashMap::from([(
                Identifier(String::from("print_int")),
                Typ::Int,
            )])],
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

fn typecheck_expression(
    env: &mut Environment,
    expression: &Expression,
) -> Result<Typ, TypErrorKind> {
    match expression {
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
                    typecheck_expression(env, &binary_expression.lhs.item)?,
                    typecheck_expression(env, &binary_expression.rhs.item)?
                );

                Ok(Typ::Unit)
            }
            BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo => {
                assert_eq!(
                    typecheck_expression(env, &binary_expression.lhs.item)?,
                    typecheck_expression(env, &binary_expression.rhs.item)?
                );

                Ok(Typ::Int)
            }
            BinaryOperator::Equals => {
                let tgt_typ = typecheck_expression(env, &binary_expression.lhs.item)?;
                assert_eq!(
                    tgt_typ,
                    typecheck_expression(env, &binary_expression.rhs.item)?
                );

                Ok(tgt_typ)
            }
        },
        Expression::Unary(unary_expression) => match unary_expression.operator {
            UnaryOperator::Not => {
                assert_eq!(
                    typecheck_expression(env, &unary_expression.operand.item)?,
                    Typ::Bool
                );

                Ok(Typ::Bool)
            }
            UnaryOperator::Negate => {
                assert_eq!(
                    typecheck_expression(env, &unary_expression.operand.item)?,
                    Typ::Int
                );

                Ok(Typ::Int)
            }
        },
        Expression::Primary(primary) => match primary {
            Primary::Bool(_) => Ok(Typ::Bool),
            Primary::Integer(_) => Ok(Typ::Int),
            // FIXME: unwrap
            Primary::Identifier(identifier) => Ok(env.lookup_variable(identifier).unwrap().clone()),
        },
        Expression::If(if_expression) => {
            assert_eq!(
                typecheck_expression(env, &if_expression.condition.item)?,
                Typ::Bool
            );

            match &if_expression.els {
                Some(els_expression) => {
                    let then_typ = typecheck_expression(env, &if_expression.then.item)?;
                    let els_typ = typecheck_expression(env, &els_expression.item)?;
                    assert_eq!(then_typ, els_typ);

                    Ok(then_typ)
                }
                None => {
                    typecheck_expression(env, &if_expression.then.item)?;

                    Ok(Typ::Unit)
                }
            }
        }
        Expression::While(while_expression) => {
            assert_eq!(
                typecheck_expression(env, &while_expression.condition.item)?,
                Typ::Bool
            );

            typecheck_expression(env, &while_expression.body.item)
        }
        Expression::Call(call_expression) => {
            let f_typ = typecheck_expression(env, &call_expression.function.item)?;

            match f_typ {
                Typ::Function { params, ret } => {
                    assert_eq!(params.len(), call_expression.args.len());

                    for (expected, expr) in params.iter().zip(&call_expression.args) {
                        assert_eq!(*expected, typecheck_expression(env, &expr.item)?);
                    }

                    Ok(*ret)
                }
                _ => todo!(),
            }
        }
        Expression::Block(block_expression) => env.with_new_scope(|env| {
            for expr in &block_expression.expressions {
                typecheck_expression(env, &expr.item)?;
            }

            match &block_expression.result_expression {
                Some(expr) => typecheck_expression(env, &expr.item),
                None => Ok(Typ::Unit),
            }
        }),
        Expression::Var(var_expression) => {
            let typ = typecheck_expression(env, &var_expression.value.item)?;

            if let Some(expected_typ) = var_expression.typ.as_ref() {
                let expected_typ = Typ::from_str(&expected_typ.0).unwrap();
                assert_eq!(typ, expected_typ);
            }

            env.set_variable_typ(var_expression.name.clone(), typ);

            Ok(Typ::Unit)
        }
    }
}
