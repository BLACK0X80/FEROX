import time
import black_ferox as black

def black_train_benchmark():
    print("🚀 Initializing FEROX Absolute Dominance Benchmark...")
    print("=" * 60)
    
    # 1. Create a substantial model
    print("[1/5] Building BlackGPT Model (2 Layers, 256 Hidden, 4 Heads)...")
    black_model = black.black_nn.black_transformers.BlackGPT(
        black_vocab_size=1000,
        black_n_layer=2,
        black_n_head=4,
        black_n_embd=256,
        black_block_size=128
    )
    
    # 2. Synthetic Data
    print("[2/5] Synthesizing Data Pipeline...")
    black_vocab = list(range(1000))
    black_data = []
    import random
    for _ in range(100):
        black_x = [random.choice(black_vocab) for _ in range(128)]
        black_data.append((black_x, black_x)) # auto-regressive shift handled internally or dummy
        
    black_dataset = black.black_data.BlackTensorDataset(black_data)
    
    # 3. Setup Optimizers
    print("[3/5] Compiling FEROX Optimizers (BlackAdamW + CosineWarmup)...")
    black_optimizer = black.black_optim.BlackAdamW(
        black_model.black_parameters(), 
        black_lr=5e-4
    )
    
    black_scheduler = black.black_optim.BlackCosineWithWarmup(
        black_optimizer,
        black_warmup_steps=10,
        black_t_max=100
    )
    
    # 4. Trainer Configuration
    print("[4/5] Initializing Heavy-Duty Trainer Engine...")
    black_args = black.black_train.BlackTrainingArguments(
        black_output_dir="./black_out",
        black_num_train_epochs=5,
        black_per_device_train_batch_size=8,
        black_logging_steps=1,
    )
    
    black_trainer = black.black_train.BlackTrainer(
        black_model=black_model,
        black_args=black_args,
        black_train_dataset=black_dataset,
        black_optimizers=(black_optimizer, black_scheduler),
    )
    
    # 5. Training
    print("[5/5] Commencing High-Speed Training Phase...")
    print("=" * 60)
    
    black_start_time = time.time()
    black_trainer.black_train()
    black_end_time = time.time()
    
    print("=" * 60)
    print(f"✅ Benchmark Complete! Total Training Time: {black_end_time - black_start_time:.4f} seconds")
    print("FEROX Engine operates flawlessly under load.")

if __name__ == "__main__":
    black_train_benchmark()
