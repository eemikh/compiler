use crate::{
    ir::{
        self, BoolOperation, Function, FunctionBuilder, Instruction, IntOperation, ModuleBuilder,
        Variable,
    },
    scope::Scope,
    syntax::ast::{Ast, BinaryExpression, BinaryOperator, Expression, Identifier, Node, Primary},
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
            Expression::Binary(binary_expression) => self.gen_binary_expression(binary_expression),
            Expression::Unary(unary_expression) => todo!(),
            Expression::Primary(primary) => todo!(),
            Expression::If(if_expression) => todo!(),
            Expression::While(while_expression) => todo!(),
            Expression::Call(call_expression) => todo!(),
            Expression::Block(block_expression) => todo!(),
            Expression::Var(var_expression) => todo!(),
        }
    }

    fn gen_binary_expression(&mut self, expression: &BinaryExpression) -> Option<Variable> {
        match self.typmap.typs[&expression.lhs.id] {
            Typ::Int => todo!(),
            Typ::Bool => self.gen_bool_binary_expression(expression),
            _ => unreachable!("type checker won't allow"),
        }
    }

    fn gen_int_binary_expression(&mut self, expression: &BinaryExpression) -> Option<Variable> {
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
                    return Some(self.gen_assignment(identifier, &expression.rhs));
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

        Some(tgt)
    }

    fn gen_bool_binary_expression(&mut self, expression: &BinaryExpression) -> Option<Variable> {
        let op = match expression.operator {
            BinaryOperator::Or => BoolOperation::Or,
            BinaryOperator::And => BoolOperation::And,
            BinaryOperator::EqualEqual => BoolOperation::Equal,
            BinaryOperator::NotEqual => BoolOperation::NotEqual,
            BinaryOperator::Equals => match &expression.lhs.item {
                Expression::Primary(Primary::Identifier(identifier)) => {
                    return Some(self.gen_assignment(identifier, &expression.rhs));
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

        Some(tgt)
    }

    fn gen_assignment(&mut self, tgt: &Identifier, rhs: &Node<Expression>) -> Variable {
        let from = self.gen_expression(rhs).expect("type checked");
        let tmp_variable = self.variable();
        let tgt = self.scope.lookup_variable(tgt).expect("type checked");

        self.builder
            .emit_instruction(Instruction::Copy { from, to: *tgt });
        // it is assumed that the variable returned may be modified. because of that, we need to
        // return a copy, not a reference to the variable (in code) we modified
        self.builder.emit_instruction(Instruction::Copy {
            from,
            to: tmp_variable,
        });

        tmp_variable
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
