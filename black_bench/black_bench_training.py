import time
import tracemalloc

BLACK_BATCH_SIZES = [8, 16, 32]


def black_benchmark_ferox_training_step(black_batch_size):
    import black_ferox as black

    black_model = black.black_nn.black_transformers.BlackGPT(
        black_vocab_size=50257,
        black_n_layer=6,
        black_n_head=6,
        black_n_embd=384,
        black_block_size=128,
        black_dropout=0.0,
    )

    black_optimizer = black.black_optim.BlackAdamW(
        black_model.black_parameters(),
        black_lr=3e-4,
    )

    tracemalloc.start()

    black_num_steps = 10
    black_start = time.perf_counter()
    for _ in range(black_num_steps):
        black_optimizer.black_step()
        black_optimizer.black_zero_grad()
    black_elapsed = (time.perf_counter() - black_start) / black_num_steps

    black_current, black_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return black_elapsed, 1.0 / black_elapsed, black_peak / 1024 / 1024


if __name__ == "__main__":
    print("=" * 70)
    print("FEROX Training Step Benchmark")
    print("=" * 70)
    print(f"{'Batch Size':>12} | {'Step Time':>12} | {'Steps/s':>10} | {'Peak Memory':>14}")
    print("-" * 70)

    for black_bs in BLACK_BATCH_SIZES:
        black_time, black_steps_s, black_mem = black_benchmark_ferox_training_step(black_bs)
        print(f"{black_bs:>12} | {black_time*1000:.2f}ms{' ':>5} | {black_steps_s:.2f}{' ':>4} | {black_mem:.2f} MB")

    print("=" * 70)
