use crate::black_buffer::BlackBuffer;
use crate::black_device::BlackDevice;
use crate::black_dtype::BlackDType;
use crate::black_error::{BlackError, BlackResult};
use crate::black_tensor::BlackTensor;
use crate::black_ops::black_matmul::black_matmul;

pub fn black_scaled_dot_product_attention(
    black_query: &BlackTensor,
    black_key: &BlackTensor,
    black_value: &BlackTensor,
    black_mask: Option<&BlackTensor>,
    _black_dropout_p: f32,
    black_is_causal: bool,
) -> BlackResult<BlackTensor> {
    let black_q_dims = black_query.black_shape.black_dims();
    if black_q_dims.len() < 3 {
        return Err(BlackError::BlackShapeError {
            black_msg: "attention requires at least 3D tensors".into(),
        });
    }

    let black_d_k = black_q_dims[black_q_dims.len() - 1] as f32;
    let black_scale = 1.0 / black_d_k.sqrt();

    let black_key_t = black_key.black_t()?;
    let mut black_scores = black_matmul(black_query, &black_key_t)?;

    let black_scores_buf = std::sync::Arc::make_mut(&mut black_scores.black_buffer);
    let black_scores_data = black_scores_buf.black_as_f32_mut_slice();
    for black_v in black_scores_data.iter_mut() {
        *black_v *= black_scale;
    }

    if black_is_causal {
        let black_s_dims = black_scores.black_shape.black_dims();
        let black_seq_len = black_s_dims[black_s_dims.len() - 1];
        let black_seq_q = black_s_dims[black_s_dims.len() - 2];
        let black_batch_size: usize = black_s_dims[..black_s_dims.len() - 2].iter().product();
        let black_scores_data = std::sync::Arc::make_mut(&mut black_scores.black_buffer)
            .black_as_f32_mut_slice();

        for black_b in 0..black_batch_size {
            for black_i in 0..black_seq_q {
                for black_j in (black_i + 1)..black_seq_len {
                    let black_idx = black_b * black_seq_q * black_seq_len
                        + black_i * black_seq_len
                        + black_j;
                    if black_idx < black_scores_data.len() {
                        black_scores_data[black_idx] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    if let Some(black_m) = black_mask {
        let black_m_data = black_m.black_contiguous()?.black_buffer.black_as_f32_slice().to_vec();
        let black_scores_data = std::sync::Arc::make_mut(&mut black_scores.black_buffer)
            .black_as_f32_mut_slice();
        let black_m_len = black_m_data.len();
        for black_i in 0..black_scores_data.len() {
            black_scores_data[black_i] += black_m_data[black_i % black_m_len];
        }
    }

    let black_attn = black_softmax_last_dim(&black_scores)?;
    black_matmul(&black_attn, black_value)
}

fn black_softmax_last_dim(black_input: &BlackTensor) -> BlackResult<BlackTensor> {
    let black_contig = black_input.black_contiguous()?;
    let black_data = black_contig.black_buffer.black_as_f32_slice();
    let black_dims = black_contig.black_shape.black_dims();
    let black_last_dim = black_dims[black_dims.len() - 1];
    let black_outer: usize = black_dims[..black_dims.len() - 1].iter().product();
    let black_numel = black_contig.black_numel();

    let mut black_out_data = vec![0.0f32; black_numel];

    for black_o in 0..black_outer {
        let black_start = black_o * black_last_dim;
        let black_end = black_start + black_last_dim;
        let black_slice = &black_data[black_start..black_end];

        let mut black_max_val = f32::NEG_INFINITY;
        for black_v in black_slice {
            if *black_v > black_max_val {
                black_max_val = *black_v;
            }
        }

        let mut black_sum_exp = 0.0f32;
        for black_i in 0..black_last_dim {
            let black_e = (black_slice[black_i] - black_max_val).exp();
            black_out_data[black_start + black_i] = black_e;
            black_sum_exp += black_e;
        }

        for black_i in 0..black_last_dim {
            black_out_data[black_start + black_i] /= black_sum_exp;
        }
    }

    let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
    Ok(BlackTensor::black_new(
        black_buf,
        black_contig.black_shape.clone(),
        BlackDType::BlackF32,
        BlackDevice::BlackCpu,
    ))
}

pub fn black_rope_embedding(
    black_x: &BlackTensor,
    black_cos: &BlackTensor,
    black_sin: &BlackTensor,
    _black_position_ids: &BlackTensor,
) -> BlackResult<BlackTensor> {
    let black_x_contig = black_x.black_contiguous()?;
    let black_x_data = black_x_contig.black_buffer.black_as_f32_slice();
    let black_x_dims = black_x_contig.black_shape.black_dims();

    let black_cos_data = black_cos.black_contiguous()?.black_buffer.black_as_f32_slice().to_vec();
    let black_sin_data = black_sin.black_contiguous()?.black_buffer.black_as_f32_slice().to_vec();

    if black_x_dims.len() < 2 {
        return Err(BlackError::BlackShapeError {
            black_msg: "RoPE requires at least 2D input".into(),
        });
    }

    let black_head_dim = black_x_dims[black_x_dims.len() - 1];
    let black_half_dim = black_head_dim / 2;
    let black_numel = black_x_contig.black_numel();
    let mut black_out_data = vec![0.0f32; black_numel];

    let black_outer = black_numel / black_head_dim;

    for black_o in 0..black_outer {
        let black_base = black_o * black_head_dim;
        for black_i in 0..black_half_dim {
            let black_x0 = black_x_data[black_base + black_i];
            let black_x1 = black_x_data[black_base + black_i + black_half_dim];
            let black_c = black_cos_data[black_i % black_cos_data.len()];
            let black_s = black_sin_data[black_i % black_sin_data.len()];
            black_out_data[black_base + black_i] = black_x0 * black_c - black_x1 * black_s;
            black_out_data[black_base + black_i + black_half_dim] =
                black_x1 * black_c + black_x0 * black_s;
        }
    }

    let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
    Ok(BlackTensor::black_new(
        black_buf,
        black_x_contig.black_shape.clone(),
        BlackDType::BlackF32,
        BlackDevice::BlackCpu,
    ))
}
