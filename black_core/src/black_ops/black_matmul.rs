use crate::black_buffer::BlackBuffer;
use crate::black_device::BlackDevice;
use crate::black_dtype::BlackDType;
use crate::black_error::{BlackError, BlackResult};
use crate::black_ops::black_elementwise::black_add;
use crate::black_shape::BlackShape;
use crate::black_tensor::BlackTensor;

const BLACK_TILE_SIZE: usize = 64;

fn black_matmul_tiled_inner_loop(
    black_a_ptr: *const f32,
    black_b_ptr: *const f32,
    black_c_ptr: *mut f32,
    black_m: usize,
    black_n: usize,
    black_k: usize,
    black_lda: usize,
    black_ldb: usize,
    black_ldc: usize,
) {
    for black_ii in (0..black_m).step_by(BLACK_TILE_SIZE) {
        for black_jj in (0..black_n).step_by(BLACK_TILE_SIZE) {
            for black_kk in (0..black_k).step_by(BLACK_TILE_SIZE) {
                let black_i_end = (black_ii + BLACK_TILE_SIZE).min(black_m);
                let black_j_end = (black_jj + BLACK_TILE_SIZE).min(black_n);
                let black_k_end = (black_kk + BLACK_TILE_SIZE).min(black_k);

                for black_i in black_ii..black_i_end {
                    for black_kk2 in black_kk..black_k_end {
                        unsafe {
                            let black_a_val = *black_a_ptr.add(black_i * black_lda + black_kk2);
                            for black_j in black_jj..black_j_end {
                                let black_c_off = black_i * black_ldc + black_j;
                                let black_b_val =
                                    *black_b_ptr.add(black_kk2 * black_ldb + black_j);
                                *black_c_ptr.add(black_c_off) +=
                                    black_a_val * black_b_val;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn black_matmul(black_a: &BlackTensor, black_b: &BlackTensor) -> BlackResult<BlackTensor> {
    let black_a_contig = black_a.black_contiguous()?;
    let black_b_contig = black_b.black_contiguous()?;
    let black_a_dims = black_a_contig.black_shape.black_dims();
    let black_b_dims = black_b_contig.black_shape.black_dims();

    if black_a_dims.len() < 2 || black_b_dims.len() < 2 {
        return Err(BlackError::BlackShapeError {
            black_msg: "matmul requires at least 2D tensors".into(),
        });
    }

    let black_m = black_a_dims[black_a_dims.len() - 2];
    let black_k = black_a_dims[black_a_dims.len() - 1];
    let black_k2 = black_b_dims[black_b_dims.len() - 2];
    let black_n = black_b_dims[black_b_dims.len() - 1];

    if black_k != black_k2 {
        return Err(BlackError::BlackShapeError {
            black_msg: format!("matmul inner dimensions mismatch: {} vs {}", black_k, black_k2),
        });
    }

    let black_batch_a: usize = black_a_dims[..black_a_dims.len() - 2].iter().product();
    let black_batch_b: usize = black_b_dims[..black_b_dims.len() - 2].iter().product();
    let black_batch = black_batch_a.max(black_batch_b);

    if black_batch_a != 1 && black_batch_b != 1 && black_batch_a != black_batch_b {
        return Err(BlackError::BlackShapeError {
            black_msg: "batch dimensions not broadcastable".into(),
        });
    }

    let black_a_data = black_a_contig.black_buffer.black_as_f32_slice();
    let black_b_data = black_b_contig.black_buffer.black_as_f32_slice();
    let black_a_stride = black_m * black_k;
    let black_b_stride = black_k * black_n;
    let black_c_stride = black_m * black_n;

    let mut black_out_data = vec![0.0f32; black_batch * black_c_stride];

    for black_bat in 0..black_batch {
        let black_a_off = if black_batch_a == 1 {
            0
        } else {
            black_bat * black_a_stride
        };
        let black_b_off = if black_batch_b == 1 {
            0
        } else {
            black_bat * black_b_stride
        };
        let black_c_off = black_bat * black_c_stride;

        let black_a_ptr = black_a_data[black_a_off..].as_ptr();
        let black_b_ptr = black_b_data[black_b_off..].as_ptr();
        let black_c_ptr = black_out_data[black_c_off..].as_mut_ptr();

        black_matmul_tiled_inner_loop(
            black_a_ptr,
            black_b_ptr,
            black_c_ptr,
            black_m,
            black_n,
            black_k,
            black_k,
            black_n,
            black_n,
        );
    }

    let mut black_out_shape_dims = Vec::new();
    let black_max_batch_dims = if black_a_dims.len() > black_b_dims.len() {
        &black_a_dims[..black_a_dims.len() - 2]
    } else {
        &black_b_dims[..black_b_dims.len() - 2]
    };
    black_out_shape_dims.extend_from_slice(black_max_batch_dims);
    black_out_shape_dims.push(black_m);
    black_out_shape_dims.push(black_n);

    let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
    Ok(BlackTensor::black_new(
        black_buf,
        BlackShape::black_new(&black_out_shape_dims),
        BlackDType::BlackF32,
        BlackDevice::BlackCpu,
    ))
}

pub fn black_bmm(black_a: &BlackTensor, black_b: &BlackTensor) -> BlackResult<BlackTensor> {
    black_matmul(black_a, black_b)
}

pub fn black_linear(
    black_input: &BlackTensor,
    black_weight: &BlackTensor,
    black_bias: Option<&BlackTensor>,
) -> BlackResult<BlackTensor> {
    let black_weight_t = black_weight.black_t()?;
    let mut black_out = black_matmul(black_input, &black_weight_t)?;

    if let Some(black_b) = black_bias {
        black_out = black_add(&black_out, black_b)?;
    }

    Ok(black_out)
}
