use crate::{
    ir::{
        self, BoolOperation, Function, FunctionBuilder, Instruction, IntOperation, ModuleBuilder,
        Variable,
    },
    scope::Scope,
    syntax::ast::{
        Ast, BinaryExpression, BinaryOperator, BlockExpression, CallExpression, Expression,
        Identifier, IfExpression, Node, NodeId, Primary, UnaryExpression, UnaryOperator,
        VarExpression, WhileExpression,
    },
    types::{Typ, TypMap},
};

struct FunctionCodegen<'a> {
    builder: FunctionBuilder,
    typmap: &'a TypMap,
    variables: u32,
    scope: Scope<Variable>,
}

impl FunctionCodegen<'_> {
    fn gen_expression(&mut self, expr: &Node<Expression>) -> Option<Variable> {
        match &expr.item {
            Expression::Binary(binary_expression) => {
                Some(self.gen_binary_expression(binary_expression))
            }
            Expression::Unary(unary_expression) => self.gen_unary(unary_expression),
            Expression::Primary(primary) => self.gen_primary(primary),
            Expression::If(if_expression) => self.gen_if_expression(expr.id, if_expression),
            Expression::While(while_expression) => self.gen_while_expression(while_expression),
            Expression::Call(call_expression) => self.gen_call_expression(call_expression),
            Expression::Block(block_expression) => self.gen_block(block_expression),
            Expression::Var(var_expression) => self.gen_var_expression(var_expression),
        }
    }

    fn gen_while_expression(&mut self, expression: &WhileExpression) -> Option<Variable> {
        let start = self.builder.label();
        let body = self.builder.label();
        let end = self.builder.label();

        self.builder.emit_label(start);

        let cond = self
            .gen_expression(&expression.condition)
            .expect("type checked");

        self.builder.emit_instruction(Instruction::CondJump {
            cond_var: cond,
            then: body,
            els: end,
        });

        self.builder.emit_label(body);

        self.gen_expression(&expression.body);

        self.builder.emit_instruction(Instruction::Jump(start));

        self.builder.emit_label(end);

        None
    }

    fn gen_if_expression(&mut self, id: NodeId, expression: &IfExpression) -> Option<Variable> {
        let cond = self
            .gen_expression(&expression.condition)
            .expect("type checked");
        let typ = self
            .typmap
            .typs
            .get(&id)
            .expect("all expressions are in the typ map");
        let then = self.builder.label();
        let els = self.builder.label();
        let done = self.builder.label();

        let res = match typ {
            Typ::Unit => None,
            _ => Some(self.variable()),
        };

        self.builder.emit_instruction(Instruction::CondJump {
            cond_var: cond,
            then,
            els,
        });

        self.builder.emit_label(then);

        let then_res = self.gen_expression(&expression.then);

        if let Some(res) = res {
            let then_res = then_res.expect("type checked");
            self.builder.emit_instruction(Instruction::Copy {
                from: then_res,
                to: res,
            });
        }

        self.builder.emit_instruction(Instruction::Jump(done));

        self.builder.emit_label(els);

        if let Some(els_expression) = &expression.els {
            let els_res = self.gen_expression(els_expression);

            if let Some(res) = res {
                let els_res = els_res.expect("type checked");
                self.builder.emit_instruction(Instruction::Copy {
                    from: els_res,
                    to: res,
                });
            }
        }

        self.builder.emit_label(done);

        res
    }

    fn gen_call_expression(&mut self, expression: &CallExpression) -> Option<Variable> {
        let params = expression
            .args
            .iter()
            .map(|expr| self.gen_expression(expr).expect("type checked"))
            .collect();

        self.builder.emit_instruction(Instruction::Call {
            function: todo!(),
            parameters: params,
            return_value: todo!(),
        });

        todo!()
    }

    fn gen_var_expression(&mut self, expression: &VarExpression) -> Option<Variable> {
        let val = self
            .gen_expression(&expression.value)
            .expect("type checked");
        self.scope.create_variable(expression.name.clone(), val);

        None
    }

    fn gen_unary(&mut self, unary: &UnaryExpression) -> Option<Variable> {
        let target = self.variable();
        let temp = self.variable();
        let operand = self.gen_expression(&unary.operand).expect("type checked");

        match unary.operator {
            UnaryOperator::Not => {
                self.builder.emit_instruction(Instruction::LoadBool {
                    target: temp,
                    value: true,
                });
                self.builder.emit_instruction(Instruction::BoolOp {
                    operation: BoolOperation::Xor,
                    target,
                    lhs: temp,
                    rhs: operand,
                });
            }
            UnaryOperator::Negate => {
                self.builder.emit_instruction(Instruction::LoadInt {
                    target: temp,
                    value: 0,
                });
                self.builder.emit_instruction(Instruction::IntOp {
                    operation: IntOperation::Subtract,
                    target,
                    lhs: temp,
                    rhs: operand,
                });
            }
        }

        Some(target)
    }

    fn gen_block(&mut self, block: &BlockExpression) -> Option<Variable> {
        self.scope.new_scope();

        for expr in &block.expressions {
            self.gen_expression(expr);
        }

        let res = match &block.result_expression {
            Some(res_expr) => self.gen_expression(res_expr),
            None => None,
        };

        self.scope.remove_scope();

        res
    }

    fn gen_primary(&mut self, primary: &Primary) -> Option<Variable> {
        match primary {
            Primary::Bool(value) => {
                let target = self.variable();

                self.builder.emit_instruction(Instruction::LoadBool {
                    target,
                    value: *value,
                });

                Some(target)
            }
            Primary::Integer(value) => {
                let target = self.variable();

                self.builder.emit_instruction(Instruction::LoadInt {
                    target,
                    value: *value,
                });

                Some(target)
            }
            Primary::Identifier(identifier) => Some(
                *self
                    .scope
                    .lookup_variable(identifier)
                    .expect("type checked"),
            ),
        }
    }

    fn gen_binary_expression(&mut self, expression: &BinaryExpression) -> Variable {
        match self.typmap.typs[&expression.lhs.id] {
            Typ::Int => self.gen_int_binary_expression(expression),
            Typ::Bool => self.gen_bool_binary_expression(expression),
            _ => unreachable!("type checker won't allow"),
        }
    }

    fn gen_int_binary_expression(&mut self, expression: &BinaryExpression) -> Variable {
        let op = match expression.operator {
            BinaryOperator::Add => IntOperation::Add,
            BinaryOperator::Subtract => IntOperation::Subtract,
            BinaryOperator::Multiply => IntOperation::Multiply,
            BinaryOperator::Divide => IntOperation::Divide,
            BinaryOperator::Modulo => IntOperation::Modulo,
            BinaryOperator::EqualEqual => IntOperation::Equal,
            BinaryOperator::NotEqual => IntOperation::NotEqual,
            BinaryOperator::LessThan => IntOperation::LessThan,
            BinaryOperator::LessEqual => IntOperation::LessEqual,
            BinaryOperator::GreaterThan => IntOperation::GreaterThan,
            BinaryOperator::GreaterEqual => IntOperation::GreaterEqual,
            BinaryOperator::Equals => match &expression.lhs.item {
                Expression::Primary(Primary::Identifier(identifier)) => {
                    return self.gen_assignment(identifier, &expression.rhs);
                }
                _ => unreachable!("type checked"),
            },
            _ => unreachable!("type checker won't allow"),
        };

        let lhs = self.gen_expression(&expression.lhs).expect("type checked");
        let rhs = self.gen_expression(&expression.rhs).expect("type checked");
        let tgt = self.variable();

        self.builder.emit_instruction(Instruction::IntOp {
            operation: op,
            target: tgt,
            lhs,
            rhs,
        });

        tgt
    }

    fn gen_bool_binary_expression(&mut self, expression: &BinaryExpression) -> Variable {
        let op = match expression.operator {
            BinaryOperator::Or => BoolOperation::Or,
            BinaryOperator::And => BoolOperation::And,
            BinaryOperator::EqualEqual => BoolOperation::Equal,
            BinaryOperator::NotEqual => BoolOperation::NotEqual,
            BinaryOperator::Equals => match &expression.lhs.item {
                Expression::Primary(Primary::Identifier(identifier)) => {
                    return self.gen_assignment(identifier, &expression.rhs);
                }
                _ => unreachable!("type checked"),
            },
            _ => unreachable!("type checker won't allow"),
        };

        let lhs = self.gen_expression(&expression.lhs).expect("type checked");
        let rhs = self.gen_expression(&expression.rhs).expect("type checked");
        let tgt = self.variable();

        self.builder.emit_instruction(Instruction::BoolOp {
            operation: op,
            target: tgt,
            lhs,
            rhs,
        });

        tgt
    }

    fn gen_assignment(&mut self, tgt: &Identifier, rhs: &Node<Expression>) -> Variable {
        let from = self.gen_expression(rhs).expect("type checked");
        let tgt = self.scope.lookup_variable(tgt).expect("type checked");

        self.builder
            .emit_instruction(Instruction::Copy { from, to: *tgt });

        *tgt
    }

    fn gen_body(mut self, body: &Node<Expression>) -> Function {
        self.scope.new_scope();
        let var = self.gen_expression(body);

        self.builder.emit_instruction(Instruction::Return(var));
        self.builder.build()
    }

    fn variable(&mut self) -> Variable {
        let var = Variable(self.variables);
        self.variables += 1;

        var
    }
}

pub fn gen_ir(ast: &Ast, typmap: &TypMap) -> ir::Module {
    let codegen = FunctionCodegen {
        builder: FunctionBuilder::new(),
        typmap,
        variables: 0,
        scope: Scope::new(),
    };

    let function = codegen.gen_body(&ast.root.body);
    let mut module = ModuleBuilder::new();
    let fid = module.function();

    module.add_function(fid, function);
    module.set_entry(fid);

    module.build()
}

#[cfg(test)]
mod tests {
    use crate::{
        ir::interpreter::{Value, interpret},
        syntax::parse,
        types::typecheck,
    };

    use super::*;

    fn test(code: &str) -> Option<Value> {
        let ast = parse(code).0.unwrap();
        let typmap = typecheck(&ast).unwrap();
        let ir = gen_ir(&ast, &typmap);
        interpret(&ir)
    }

    #[test]
    fn test_binary() {
        assert_eq!(test("1 + 1"), Some(Value::Int(2)));
        assert_eq!(test("2 * 3"), Some(Value::Int(6)));
        assert_eq!(test("48 / 6"), Some(Value::Int(8)));
        assert_eq!(test("48 - 6"), Some(Value::Int(42)));
        assert_eq!(test("true or false"), Some(Value::Bool(true)));
        assert_eq!(test("false or false"), Some(Value::Bool(false)));
        assert_eq!(test("true and true and false"), Some(Value::Bool(false)));
        assert_eq!(test("true and true and true"), Some(Value::Bool(true)));
        assert_eq!(test("false and true and true"), Some(Value::Bool(false)));
    }

    #[test]
    fn test_unary() {
        assert_eq!(test("-1"), Some(Value::Int(-1)));
        assert_eq!(test("not true"), Some(Value::Bool(false)));
        assert_eq!(test("not not true"), Some(Value::Bool(true)));
    }

    #[test]
    fn test_block() {
        assert_eq!(test("2 + {1 + 1};"), None);
        assert_eq!(test("{1 + 1}"), Some(Value::Int(2)));
        assert_eq!(test("1 + 1; 2 + 2; 0"), Some(Value::Int(0)));
        assert_eq!(test("{1 + 1} {2 + 2} 0"), Some(Value::Int(0)));
        assert_eq!(test("{1 + 1}; {2 + 2}; 0"), Some(Value::Int(0)));
    }

    #[test]
    fn test_var() {
        assert_eq!(test("var a = 1; a"), Some(Value::Int(1)));
        assert_eq!(test("var a = 1; a = 2; a"), Some(Value::Int(2)));
        assert_eq!(test("var a = 1; var a = 2; a"), Some(Value::Int(2)));
        assert_eq!(test("var a = 1; {var a = 2} a"), Some(Value::Int(1)));
    }

    #[test]
    fn test_if() {
        assert_eq!(
            test("var a = 1; if true then {a = 2}; a"),
            Some(Value::Int(2))
        );
        assert_eq!(
            test("var a = 1; if false then {a = 2}; a"),
            Some(Value::Int(1))
        );
        assert_eq!(
            test("var a = 1; if false then {a = 2} else {a = 3}; a"),
            Some(Value::Int(3))
        );
        assert_eq!(
            test("var a = 1; if true then {a = 2} else {a = 3}; a"),
            Some(Value::Int(2))
        );
        assert_eq!(test("if true then {1} else 2"), Some(Value::Int(1)));
        assert_eq!(test("if false then {1} else 2"), Some(Value::Int(2)));
        assert_eq!(test("if false then {1}"), None);
    }

    #[test]
    fn test_while() {
        assert_eq!(
            test("var a = 1; while a < 10 do {a = a + 1}; a"),
            Some(Value::Int(10))
        );
        assert_eq!(
            test("var a = 1; while a > 10 do {a = a + 1}; a"),
            Some(Value::Int(1))
        );
    }

    #[test]
    fn collatz() {
        assert_eq!(
            test(
                "
                    var i = 27;
                    var c = 0;
                    while i != 1 do {
                        c = c + 1;
                        if i % 2 == 0 then {
                            i = i / 2;
                        } else {
                            i = 3 * i + 1;
                        }
                    }
                    c
                "
            ),
            Some(Value::Int(111))
        );
    }

    #[test]
    fn fibonacci() {
        assert_eq!(
            test(
                "
                    var sum = 0;
                    var last1 = 1;
                    var last2 = 0;
                    var c = 0;
                    while c < 9 do {
                        sum = sum + last1;
                        c = c + 1;
                        var tmp = last1 + last2;
                        last2 = last1;
                        last1 = tmp;
                    }
                    sum
                "
            ),
            Some(Value::Int(88))
        );
    }
}
