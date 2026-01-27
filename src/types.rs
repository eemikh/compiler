use std::{collections::HashMap, str::FromStr};

use crate::syntax::ast::{
    Ast, BinaryOperator, Expression, Identifier, Module, Node, NodeId, Primary, UnaryOperator,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypErrorKind {
    BinaryMismatch {
        op: BinaryOperator,
        lhs: Typ,
        rhs: Typ,
    },
    UnaryMismatch {
        op: UnaryOperator,
        operand: Typ,
    },
    ExpectedTyp {
        expected: Typ,
        got: Typ,
    },
    IfMismatch {
        then: Typ,
        els: Typ,
    },
    InvalidParameterCount {
        expected: usize,
        got: usize,
    },
    ExpectedFunction {
        got: Typ,
    },
    InvalidTyp,
    InvalidVar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypError {
    pub node: NodeId,
    pub kind: TypErrorKind,
}

/// This type conveys that *a* type error was observed and as such, no type information is available
/// for any dependent expressions or the module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnyTypError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Typ {
    Int,
    Bool,
    Unit,
    // TODO: make clone great again
    Function { params: Vec<Typ>, ret: Box<Typ> },
}

/// Represents the typ of a variable during type checking. If the typ of a variable could not be
/// determined, the typ in the typ map is set to unknown to distinguish between nonexistent
/// variables and variables with typ erors.
#[derive(Debug, Clone, PartialEq, Eq)]
enum VarTyp {
    Unknown,
    Known(Typ),
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
    variables: Vec<HashMap<Identifier, VarTyp>>,
    typmap: TypMap,
    errors: Vec<TypError>,
}

impl Environment {
    fn new() -> Self {
        Self {
            variables: vec![HashMap::from([(
                Identifier(String::from("print_int")),
                VarTyp::Known(Typ::Function {
                    params: vec![Typ::Int],
                    ret: Box::new(Typ::Unit),
                }),
            )])],
            typmap: TypMap {
                typs: HashMap::new(),
            },
            errors: Vec::new(),
        }
    }

    fn lookup_variable(&self, identifier: &Identifier) -> Option<&VarTyp> {
        self.variables
            .iter()
            .rev()
            .filter_map(|level| level.get(identifier))
            .next()
    }

    fn set_variable_typ(&mut self, identifier: Identifier, typ: VarTyp) {
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

    fn error(&mut self, error: TypError) {
        self.errors.push(error);
    }
}

pub fn typecheck(ast: &Ast) -> Result<TypMap, Vec<TypError>> {
    let mut env = Environment::new();
    let _ = typecheck_module(&mut env, &ast.root);

    if env.errors.is_empty() {
        Ok(env.typmap)
    } else {
        Err(env.errors)
    }
}

fn typecheck_module(env: &mut Environment, module: &Module) -> Result<Typ, AnyTypError> {
    typecheck_expression(env, &module.body)
}

fn typecheck_expression(
    env: &mut Environment,
    expression: &Node<Expression>,
) -> Result<Typ, AnyTypError> {
    let typ = match &expression.item {
        Expression::Binary(binary_expression) => {
            let lhs = typecheck_expression(env, &binary_expression.lhs);
            let rhs = typecheck_expression(env, &binary_expression.rhs);

            // TODO: check that the binary can accept the type

            if let (Ok(lhs), Ok(rhs)) = (&lhs, &rhs)
                && lhs != rhs
            {
                env.error(TypError {
                    node: expression.id,
                    kind: TypErrorKind::BinaryMismatch {
                        op: binary_expression.operator,
                        // TODO: get rid of these clones (or make the clones very cheap)
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                    },
                });
            }

            match binary_expression.operator {
                BinaryOperator::Or
                | BinaryOperator::And
                | BinaryOperator::EqualEqual
                | BinaryOperator::NotEqual
                | BinaryOperator::LessThan
                | BinaryOperator::LessEqual
                | BinaryOperator::GreaterThan
                | BinaryOperator::GreaterEqual => Typ::Bool,
                BinaryOperator::Add
                | BinaryOperator::Subtract
                | BinaryOperator::Multiply
                | BinaryOperator::Divide
                | BinaryOperator::Modulo => Typ::Int,
                BinaryOperator::Equals => {
                    if let (Ok(lhs), Ok(rhs)) = (lhs, rhs)
                        && lhs == rhs
                    {
                        lhs
                    } else {
                        return Err(AnyTypError {});
                    }
                }
            }
        }
        Expression::Unary(unary_expression) => match unary_expression.operator {
            UnaryOperator::Not => {
                if let Ok(operand) = typecheck_expression(env, &unary_expression.operand)
                    && operand != Typ::Bool
                {
                    env.error(TypError {
                        node: expression.id,
                        kind: TypErrorKind::UnaryMismatch {
                            op: unary_expression.operator,
                            operand,
                        },
                    });
                }

                Typ::Bool
            }
            UnaryOperator::Negate => {
                if let Ok(operand) = typecheck_expression(env, &unary_expression.operand)
                    && operand != Typ::Int
                {
                    env.error(TypError {
                        node: expression.id,
                        kind: TypErrorKind::UnaryMismatch {
                            op: unary_expression.operator,
                            operand,
                        },
                    });
                }

                Typ::Int
            }
        },
        Expression::Primary(primary) => match primary {
            Primary::Bool(_) => Typ::Bool,
            Primary::Integer(_) => Typ::Int,
            Primary::Identifier(identifier) => match env.lookup_variable(identifier) {
                Some(typ) => match typ {
                    VarTyp::Unknown => return Err(AnyTypError {}),
                    VarTyp::Known(typ) => typ.clone(),
                },
                None => {
                    env.error(TypError {
                        node: expression.id,
                        kind: TypErrorKind::InvalidVar,
                    });

                    return Err(AnyTypError {});
                }
            },
        },
        Expression::If(if_expression) => {
            if let Ok(operand) = typecheck_expression(env, &if_expression.condition)
                && operand != Typ::Bool
            {
                env.error(TypError {
                    node: if_expression.condition.id,
                    kind: TypErrorKind::ExpectedTyp {
                        expected: Typ::Bool,
                        got: operand,
                    },
                });
            }

            match &if_expression.els {
                Some(els_expression) => {
                    let then_typ = typecheck_expression(env, &if_expression.then);
                    let els_typ = typecheck_expression(env, els_expression);

                    if let (Ok(then_typ), Ok(els_typ)) = (then_typ, els_typ) {
                        if then_typ == els_typ {
                            then_typ
                        } else {
                            env.error(TypError {
                                node: expression.id,
                                kind: TypErrorKind::IfMismatch {
                                    then: then_typ,
                                    els: els_typ,
                                },
                            });

                            return Err(AnyTypError {});
                        }
                    } else {
                        return Err(AnyTypError {});
                    }
                }
                None => {
                    // make sure the types are recursively checked but no need to do anything in
                    // case of error
                    let _ = typecheck_expression(env, &if_expression.then);

                    Typ::Unit
                }
            }
        }
        Expression::While(while_expression) => {
            if let Ok(operand) = typecheck_expression(env, &while_expression.condition)
                && operand != Typ::Bool
            {
                env.error(TypError {
                    node: while_expression.condition.id,
                    kind: TypErrorKind::ExpectedTyp {
                        expected: Typ::Bool,
                        got: operand,
                    },
                });
            }

            // make sure the types are recursively checked but no need to do anything in case of
            // error
            let _ = typecheck_expression(env, &while_expression.body);

            Typ::Unit
        }
        Expression::Call(call_expression) => {
            let f_typ = typecheck_expression(env, &call_expression.function)?;

            match f_typ {
                Typ::Function { params, ret } => {
                    if params.len() != call_expression.args.len() {
                        env.error(TypError {
                            node: expression.id,
                            kind: TypErrorKind::InvalidParameterCount {
                                expected: params.len(),
                                got: call_expression.args.len(),
                            },
                        });
                    }

                    // TODO: recursively check all parameters in all cases
                    for (expected, expr) in params.iter().zip(&call_expression.args) {
                        let got_typ = typecheck_expression(env, expr);

                        if let Ok(got_typ) = got_typ
                            && *expected != got_typ
                        {
                            env.error(TypError {
                                node: expr.id,
                                kind: TypErrorKind::ExpectedTyp {
                                    expected: expected.clone(),
                                    got: got_typ,
                                },
                            });
                        }
                    }

                    *ret
                }
                _ => {
                    env.error(TypError {
                        node: call_expression.function.id,
                        kind: TypErrorKind::ExpectedFunction { got: f_typ },
                    });

                    return Err(AnyTypError {});
                }
            }
        }
        Expression::Block(block_expression) => env.with_new_scope(|env| {
            for expr in &block_expression.expressions {
                let _ = typecheck_expression(env, expr);
            }

            match &block_expression.result_expression {
                Some(expr) => typecheck_expression(env, expr),
                None => Ok(Typ::Unit),
            }
        })?,
        Expression::Var(var_expression) => {
            let typ = typecheck_expression(env, &var_expression.value);

            if let Ok(typ) = typ {
                if let Some(expected_typ_node) = var_expression.typ.as_ref() {
                    let expected_typ = Typ::from_str(&expected_typ_node.item.0);

                    if let Ok(expected_typ) = expected_typ {
                        if typ != expected_typ {
                            env.error(TypError {
                                node: var_expression.value.id,
                                kind: TypErrorKind::ExpectedTyp {
                                    expected: expected_typ.clone(),
                                    got: typ,
                                },
                            });
                        }

                        env.set_variable_typ(
                            var_expression.name.clone(),
                            VarTyp::Known(expected_typ),
                        );
                    } else {
                        env.error(TypError {
                            node: expected_typ_node.id,
                            kind: TypErrorKind::InvalidTyp,
                        });

                        env.set_variable_typ(var_expression.name.clone(), VarTyp::Unknown);
                    }
                } else {
                    env.set_variable_typ(var_expression.name.clone(), VarTyp::Known(typ));
                }
            } else {
                env.set_variable_typ(var_expression.name.clone(), VarTyp::Unknown);
            }

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

    fn str_to_typemap(code: &str) -> Result<TypMap, Vec<TypError>> {
        let ast = parse(code).0.unwrap();
        typecheck(&ast)
    }

    fn module_to_typ(code: &str) -> Result<Typ, Vec<TypError>> {
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

    #[test]
    fn test_errors() {
        assert_eq!(
            module_to_typ("var a: Int = print_int(1)").unwrap_err()[0].kind,
            TypErrorKind::ExpectedTyp {
                expected: Typ::Int,
                got: Typ::Unit
            }
        );

        assert_eq!(
            module_to_typ("var a = 1; var b = true; a or b").unwrap_err()[0].kind,
            TypErrorKind::BinaryMismatch {
                op: BinaryOperator::Or,
                lhs: Typ::Int,
                rhs: Typ::Bool
            }
        );

        assert_eq!(
            module_to_typ("1 + true").unwrap_err()[0].kind,
            TypErrorKind::BinaryMismatch {
                op: BinaryOperator::Add,
                lhs: Typ::Int,
                rhs: Typ::Bool
            }
        );

        assert_eq!(
            module_to_typ("print_int()").unwrap_err()[0].kind,
            TypErrorKind::InvalidParameterCount {
                expected: 1,
                got: 0
            }
        );

        assert_eq!(
            module_to_typ("print_int(true)").unwrap_err()[0].kind,
            TypErrorKind::ExpectedTyp {
                expected: Typ::Int,
                got: Typ::Bool
            }
        );
    }
}
