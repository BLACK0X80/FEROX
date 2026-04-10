import black_ferox as black

black_model = black.black_nn.black_transformers.BlackVisionTransformer(
    black_image_size=224,
    black_patch_size=16,
    black_num_classes=1000,
    black_dim=768,
    black_depth=12,
    black_heads=12,
    black_mlp_dim=3072,
    black_dropout=0.1,
)

black_optimizer = black.black_optim.BlackAdamW(
    black_model.black_parameters(),
    black_lr=1e-3,
    black_weight_decay=0.05,
)

black_train_data = [[0.0] * (3 * 224 * 224)] * 50
black_train_labels = [0] * 50
black_train_dataset = black.black_data.BlackTensorDataset(black_train_data, black_train_labels)

black_val_data = [[0.0] * (3 * 224 * 224)] * 10
black_val_labels = [0] * 10
black_val_dataset = black.black_data.BlackTensorDataset(black_val_data, black_val_labels)

black_train_loader = black.black_data.BlackDataLoader(
    black_train_dataset,
    black_batch_size=256,
    black_shuffle=True,
    black_num_workers=0,
    black_pin_memory=True,
)

black_args = black.black_train.BlackTrainingArguments(
    black_output_dir="./black_vit_checkpoints",
    black_num_train_epochs=100,
    black_per_device_train_batch_size=256,
    black_learning_rate=1e-3,
    black_bf16=True,
    black_logging_steps=50,
    black_save_steps=5000,
)

black_trainer = black.black_train.BlackTrainer(
    black_model=black_model,
    black_args=black_args,
    black_train_dataset=black_train_dataset,
    black_eval_dataset=black_val_dataset,
    black_optimizers=(black_optimizer, None),
)

if __name__ == "__main__":
    print("FEROX Vision Transformer Training Example")
    print(f"Model: {black_model}")
    print(f"Parameters: {black_model.black_num_parameters()}")
    print(f"Training data: {len(black_train_dataset)} samples")
    print("Starting training...")
    black_trainer.black_train()
    print("Training complete!")
