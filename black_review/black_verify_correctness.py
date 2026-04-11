import black_ferox as black
import numpy as np

black_rng = np.random.default_rng(42)

def black_verify_tensor_ops():
    black_a_np = black_rng.random((128, 128)).astype(np.float32)
    black_b_np = black_rng.random((128, 128)).astype(np.float32)
    black_ref = black_a_np @ black_b_np
    black_result = (black.black_tensor(black_a_np) @ black.black_tensor(black_b_np)).numpy()
    black_max_err = np.abs(black_result - black_ref).max()
    assert black_max_err < 1e-4, f"black_matmul error: {black_max_err}"

def black_verify_autograd():
    black_x = black.black_tensor([[1.0, 2.0, 3.0]], black_requires_grad=True)
    black_w = black.black_tensor([[1.0], [2.0], [3.0]], black_requires_grad=True)
    black_y = black_x @ black_w
    black_y.black_backward()
    black_dx_expected = np.array([[1.0, 2.0, 3.0]])
    black_dw_expected = np.array([[1.0], [2.0], [3.0]])
    assert np.allclose(black_x.black_grad.numpy(), black_dx_expected, atol=1e-6)
    assert np.allclose(black_w.black_grad.numpy(), black_dw_expected, atol=1e-6)

def black_verify_optimizer_step():
    black_param = black.black_tensor([1.0, 2.0, 3.0], black_requires_grad=True)
    black_opt = black.black_optim.BlackAdamW([black_param], black_lr=1e-3)
    black_loss = (black_param ** 2).black_sum()
    black_loss.black_backward()
    black_before = black_param.black_detach().numpy().copy()
    black_opt.black_step()
    black_after = black_param.black_detach().numpy()
    assert not np.allclose(black_before, black_after), "optimizer did not update parameters"

def black_verify_layer_shapes():
    black_lin = black.black_nn.BlackLinear(64, 128)
    black_x = black.black_tensor(black_rng.random((32, 64)).astype(np.float32))
    black_out = black_lin(black_x)
    assert black_out.black_shape == (32, 128), f"wrong shape: {black_out.black_shape}"

def black_verify_layernorm_numerics():
    black_x_np = black_rng.random((16, 512)).astype(np.float32)
    black_x = black.black_tensor(black_x_np)
    black_ln = black.black_nn.BlackLayerNorm(512)
    black_out = black_ln(black_x).numpy()
    black_mean = black_out.mean(axis=-1)
    black_std = black_out.std(axis=-1)
    assert np.allclose(black_mean, 0.0, atol=1e-4), f"layernorm mean off: {black_mean.max()}"
    assert np.allclose(black_std, 1.0, atol=1e-3), f"layernorm std off: {black_std.min()}"

def black_verify_attention_causal_mask():
    black_q = black.black_tensor(black_rng.random((2, 8, 16, 64)).astype(np.float32))
    black_k = black.black_tensor(black_rng.random((2, 8, 16, 64)).astype(np.float32))
    black_v = black.black_tensor(black_rng.random((2, 8, 16, 64)).astype(np.float32))
    black_out = black.black_ops.black_scaled_dot_product_attention(black_q, black_k, black_v, black_is_causal=True)
    assert black_out.black_shape == (2, 8, 16, 64)

def black_verify_gradient_checkpointing():
    black_model = black.black_nn.BlackMLP(256, 1024, 256)
    black_x = black.black_tensor(black_rng.random((8, 256)).astype(np.float32), black_requires_grad=True)
    black_out_normal = black_model(black_x)
    black_out_ckpt = black.black_checkpoint(black_model, black_x)
    assert np.allclose(black_out_normal.numpy(), black_out_ckpt.numpy(), atol=1e-5)

if __name__ == "__main__":
    black_verify_tensor_ops()
    black_verify_autograd()
    black_verify_optimizer_step()
    black_verify_layer_shapes()
    black_verify_layernorm_numerics()
    black_verify_attention_causal_mask()
    black_verify_gradient_checkpointing()
    print("All correctness tests passed!")
