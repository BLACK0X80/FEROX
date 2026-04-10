import black_ferox as black

black_model = black.black_nn.black_transformers.BlackGPT(
    black_vocab_size=50257,
    black_n_layer=12,
    black_n_head=12,
    black_n_embd=768,
    black_block_size=1024,
    black_dropout=0.1,
)

black_optimizer = black.black_optim.BlackAdamW(
    black_model.black_parameters(),
    black_lr=3e-4,
    black_weight_decay=0.1,
)

black_scheduler = black.black_optim.BlackCosineWithWarmup(
    black_optimizer,
    black_warmup_steps=2000,
    black_t_max=100000,
)

black_args = black.black_train.BlackTrainingArguments(
    black_output_dir="./black_checkpoints",
    black_num_train_epochs=3,
    black_per_device_train_batch_size=16,
    black_gradient_accumulation_steps=4,
    black_learning_rate=3e-4,
    black_bf16=True,
    black_gradient_checkpointing=True,
    black_max_grad_norm=1.0,
    black_warmup_steps=2000,
    black_logging_steps=100,
    black_save_steps=1000,
)

black_train_data = [[0, 1, 2, 3]] * 100
black_dataset = black.black_data.BlackTensorDataset(black_train_data)

black_trainer = black.black_train.BlackTrainer(
    black_model=black_model,
    black_args=black_args,
    black_train_dataset=black_dataset,
    black_optimizers=(black_optimizer, black_scheduler),
)

if __name__ == "__main__":
    print("FEROX GPT Training Example")
    print(f"Model: {black_model}")
    print(f"Parameters: {black_model.black_num_parameters()}")
    print(f"Optimizer LR: {black_optimizer.black_lr}")
    print(f"Starting training...")
    black_trainer.black_train()
    print("Training complete!")
