import time
import sys

BLACK_SIZES = [256, 512, 1024, 2048, 4096]


def black_benchmark_ferox_matmul(black_size):
    try:
        from black_ferox._black_ferox_core import BlackTensor
        black_a = BlackTensor.black_rand([black_size, black_size])
        black_b = BlackTensor.black_rand([black_size, black_size])

        black_warmup_iters = 3
        for _ in range(black_warmup_iters):
            _ = black_a @ black_b

        black_num_iters = 10
        black_start = time.perf_counter()
        for _ in range(black_num_iters):
            _ = black_a @ black_b
        black_elapsed = (time.perf_counter() - black_start) / black_num_iters

        black_flops = 2.0 * black_size ** 3
        black_gflops = black_flops / black_elapsed / 1e9
        return black_elapsed, black_gflops
    except ImportError:
        return None, None


def black_benchmark_numpy_matmul(black_size):
    import numpy as black_np
    black_a = black_np.random.randn(black_size, black_size).astype(black_np.float32)
    black_b = black_np.random.randn(black_size, black_size).astype(black_np.float32)

    black_warmup_iters = 3
    for _ in range(black_warmup_iters):
        _ = black_a @ black_b

    black_num_iters = 10
    black_start = time.perf_counter()
    for _ in range(black_num_iters):
        _ = black_a @ black_b
    black_elapsed = (time.perf_counter() - black_start) / black_num_iters

    black_flops = 2.0 * black_size ** 3
    black_gflops = black_flops / black_elapsed / 1e9
    return black_elapsed, black_gflops


if __name__ == "__main__":
    print("=" * 70)
    print("FEROX Matmul Benchmark")
    print("=" * 70)
    print(f"{'Size':>6} | {'NumPy Time':>12} | {'NumPy GFLOPS':>14} | {'FEROX Time':>12} | {'FEROX GFLOPS':>14}")
    print("-" * 70)

    for black_size in BLACK_SIZES:
        black_np_time, black_np_gflops = black_benchmark_numpy_matmul(black_size)
        black_fx_time, black_fx_gflops = black_benchmark_ferox_matmul(black_size)

        black_np_str = f"{black_np_time*1000:.2f}ms" if black_np_time else "N/A"
        black_np_g = f"{black_np_gflops:.2f}" if black_np_gflops else "N/A"
        black_fx_str = f"{black_fx_time*1000:.2f}ms" if black_fx_time else "N/A"
        black_fx_g = f"{black_fx_gflops:.2f}" if black_fx_gflops else "N/A"

        print(f"{black_size:>6} | {black_np_str:>12} | {black_np_g:>14} | {black_fx_str:>12} | {black_fx_g:>14}")

    print("=" * 70)
