use std::collections::HashMap;

use crate::ir::{Function, FunctionId, Instruction, InternalFunction, LabelId, Module};

pub struct FunctionBuilder {
    function: InternalFunction,
    labels: u32,
}

impl FunctionBuilder {
    pub fn new() -> Self {
        Self {
            function: InternalFunction {
                instructions: Vec::new(),
                labels: HashMap::new(),
            },
            labels: 0,
        }
    }

    pub fn build(self) -> InternalFunction {
        self.function
    }

    pub fn label(&mut self) -> LabelId {
        let label_id = LabelId(self.labels);
        self.labels += 1;

        label_id
    }

    pub fn emit_label(&mut self, label: LabelId) {
        let had_label = self
            .function
            .labels
            .insert(label, self.function.instructions.len())
            .is_some();

        assert!(!had_label);
    }

    pub fn emit_instruction(&mut self, instruction: Instruction) {
        self.function.instructions.push(instruction);
    }
}

pub struct ModuleBuilder {
    functions: HashMap<FunctionId, Function>,
    num_functions: u32,
    entry: Option<FunctionId>,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            num_functions: 0,
            entry: None,
        }
    }

    pub fn build(mut self) -> Module {
        let mut module = Module {
            functions: Vec::new(),
            entry: self.entry.unwrap(),
        };

        for function_id in (0..self.num_functions).map(FunctionId) {
            module
                .functions
                .push(self.functions.remove(&function_id).unwrap());
        }

        module
    }

    pub fn set_entry(&mut self, entry: FunctionId) {
        self.entry.replace(entry);
    }

    pub fn function(&mut self) -> FunctionId {
        let function = FunctionId(self.num_functions);
        self.num_functions += 1;

        function
    }

    pub fn add_function(&mut self, id: FunctionId, function: Function) {
        let had_function = self.functions.insert(id, function).is_some();

        assert!(!had_function);
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{FunctionKind, Variable};

    use super::*;

    #[test]
    fn test_function() {
        let mut builder = FunctionBuilder::new();

        let l1 = builder.label();
        let l2 = builder.label();
        let i1 = Instruction::Copy {
            from: Variable(0),
            to: Variable(1),
        };
        let i2 = Instruction::Copy {
            from: Variable(1),
            to: Variable(0),
        };
        builder.emit_instruction(i1.clone());
        builder.emit_label(l1);
        builder.emit_instruction(i2.clone());
        builder.emit_label(l2);

        assert_eq!(
            builder.build(),
            InternalFunction {
                instructions: vec![i1, i2],
                labels: HashMap::from([(LabelId(0), 1), (LabelId(1), 2)])
            }
        );
    }

    #[test]
    #[should_panic]
    fn test_function_dup_label() {
        let mut builder = FunctionBuilder::new();
        let l1 = builder.label();
        builder.emit_label(l1);
        builder.emit_label(l1);
    }

    #[test]
    fn test_module() {
        let mut builder = ModuleBuilder::new();
        let f1id = builder.function();
        let f2id = builder.function();
        let f1 = Function {
            name: String::from("f1"),
            kind: FunctionKind::Internal(InternalFunction {
                instructions: vec![],
                labels: HashMap::new(),
            }),
        };
        let f2 = Function {
            name: String::from("f2"),
            kind: FunctionKind::Internal(InternalFunction {
                instructions: vec![Instruction::Copy {
                    from: Variable(0),
                    to: Variable(1),
                }],
                labels: HashMap::new(),
            }),
        };

        builder.add_function(f2id, f2.clone());
        builder.add_function(f1id, f1.clone());

        builder.set_entry(f2id);

        assert_eq!(
            builder.build(),
            Module {
                functions: vec![f1, f2],
                entry: f2id,
            }
        )
    }

    #[test]
    #[should_panic]
    fn test_module_missing_function() {
        let mut builder = ModuleBuilder::new();
        let f = builder.function();
        builder.set_entry(f);
        builder.build();
    }

    #[test]
    #[should_panic]
    fn test_module_missing_entry() {
        let builder = ModuleBuilder::new();
        builder.build();
    }
}
