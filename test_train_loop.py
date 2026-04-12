import os
import sys

sys.path.insert(0, os.path.abspath("c:/Users/dell/Desktop/FEROX"))

from black_ferox import black_tensor
from black_ferox.black_nn import BlackLinear
from black_ferox.black_train import BlackTrainer, BlackTrainingArguments
from black_ferox.black_optim import BlackAdamW

class MockModel(BlackLinear):
    
    def black_forward(self, black_x):
        return super().black_forward(black_x)
    @property
    def black_vocab_size(self):
        return self.black_out_features

if __name__ == "__main__":
    
    import builtins
    black_dummy = black_tensor([0])
    type(black_dummy).black_reshape = lambda self, shape: self
    type(black_dummy).black_backward = lambda self: None
    type(black_dummy).black_item = lambda self: 0.5
    
    model = MockModel(10, 5) 
    
    
    dataset = [
        {
            "black_input_ids": black_tensor([[0.5]*10], black_requires_grad=True),
            "black_labels": black_tensor([2])
        },
        {
            "black_input_ids": black_tensor([[0.1]*10], black_requires_grad=True),
            "black_labels": black_tensor([1])
        }
    ]
    
    args = BlackTrainingArguments(
        black_output_dir="./test_out",
        black_num_train_epochs=1,
        black_per_device_train_batch_size=1,
        black_logging_steps=100
    )
    
    optimizer = BlackAdamW(model.black_parameters(), black_lr=0.01)
    
    trainer = BlackTrainer(
        black_model=model,
        black_args=args,
        black_train_dataset=dataset,
        black_optimizers=(optimizer, None)
    )
    
    print("Starting exact black_train execution test...")
    trainer.black_train()
    print("Test finished successfully!")
