import tracemalloc
import time

BLACK_NUM_STEPS = 100


def black_benchmark_memory_default():
    import black_ferox as black

    tracemalloc.start()

    black_model = black.black_nn.black_transformers.BlackGPT(
        black_vocab_size=50257,
        black_n_layer=4,
        black_n_head=4,
        black_n_embd=256,
        black_block_size=64,
    )

    black_optimizer = black.black_optim.BlackAdamW(
        black_model.black_parameters(),
        black_lr=3e-4,
    )

    black_peak_samples = []
    for black_step in range(BLACK_NUM_STEPS):
        black_optimizer.black_step()
        black_optimizer.black_zero_grad()
        _, black_peak = tracemalloc.get_traced_memory()
        black_peak_samples.append(black_peak)

    black_current, black_final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return black_peak_samples, black_final_peak


if __name__ == "__main__":
    print("=" * 70)
    print("FEROX Memory Benchmark")
    print(f"Running {BLACK_NUM_STEPS} training steps...")
    print("=" * 70)

    black_samples, black_peak = black_benchmark_memory_default()

    print(f"Peak memory: {black_peak / 1024 / 1024:.2f} MB")
    print(f"Step 1 peak: {black_samples[0] / 1024 / 1024:.2f} MB")
    print(f"Step 50 peak: {black_samples[49] / 1024 / 1024:.2f} MB")
    print(f"Step 100 peak: {black_samples[-1] / 1024 / 1024:.2f} MB")
    print(f"Memory growth (step 1 -> 100): {(black_samples[-1] - black_samples[0]) / 1024:.2f} KB")
    print("=" * 70)
