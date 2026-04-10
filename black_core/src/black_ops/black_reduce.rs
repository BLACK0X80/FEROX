use crate::black_buffer::BlackBuffer;
use crate::black_device::BlackDevice;
use crate::black_dtype::BlackDType;
use crate::black_error::{BlackError, BlackResult};
use crate::black_shape::BlackShape;
use crate::black_tensor::BlackTensor;

pub fn black_sum(
    black_input: &BlackTensor,
    black_dim: Option<usize>,
    black_keepdim: bool,
) -> BlackResult<BlackTensor> {
    let black_contig = black_input.black_contiguous()?;
    let black_data = black_contig.black_buffer.black_as_f32_slice();
    let black_dims = black_contig.black_shape.black_dims();

    match black_dim {
        None => {
            let black_total: f32 = black_data[..black_contig.black_numel()].iter().sum();
            BlackTensor::black_scalar_f32(black_total)
        }
        Some(black_d) => {
            if black_d >= black_dims.len() {
                return Err(BlackError::BlackIndexError {
                    black_msg: format!("dim {} out of range for {}D tensor", black_d, black_dims.len()),
                });
            }
            let black_dim_size = black_dims[black_d];
            let mut black_out_dims: Vec<usize> = black_dims.to_vec();
            if black_keepdim {
                black_out_dims[black_d] = 1;
            } else {
                black_out_dims.remove(black_d);
            }
            if black_out_dims.is_empty() {
                black_out_dims.push(1);
            }

            let black_out_shape = BlackShape::black_new(&black_out_dims);
            let black_out_numel = black_out_shape.black_numel();
            let mut black_out_data = vec![0.0f32; black_out_numel];

            let black_outer: usize = black_dims[..black_d].iter().product();
            let black_inner: usize = black_dims[black_d + 1..].iter().product();

            for black_o in 0..black_outer {
                for black_i in 0..black_inner {
                    let mut black_acc = 0.0f32;
                    for black_k in 0..black_dim_size {
                        let black_idx =
                            black_o * black_dim_size * black_inner + black_k * black_inner + black_i;
                        black_acc += black_data[black_idx];
                    }
                    let black_out_idx = black_o * black_inner + black_i;
                    black_out_data[black_out_idx] = black_acc;
                }
            }

            let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
            Ok(BlackTensor::black_new(
                black_buf,
                black_out_shape,
                BlackDType::BlackF32,
                BlackDevice::BlackCpu,
            ))
        }
    }
}

pub fn black_mean(
    black_input: &BlackTensor,
    black_dim: Option<usize>,
    black_keepdim: bool,
) -> BlackResult<BlackTensor> {
    let black_contig = black_input.black_contiguous()?;
    let black_dims = black_contig.black_shape.black_dims();

    match black_dim {
        None => {
            let black_data = black_contig.black_buffer.black_as_f32_slice();
            let black_n = black_contig.black_numel();
            let black_total: f32 = black_data[..black_n].iter().sum();
            BlackTensor::black_scalar_f32(black_total / black_n as f32)
        }
        Some(black_d) => {
            let black_dim_size = black_dims[black_d];
            let mut black_sum_tensor = black_sum(black_input, Some(black_d), black_keepdim)?;
            let black_buf = std::sync::Arc::make_mut(&mut black_sum_tensor.black_buffer);
            let black_slice = black_buf.black_as_f32_mut_slice();
            for black_v in black_slice.iter_mut() {
                *black_v /= black_dim_size as f32;
            }
            Ok(black_sum_tensor)
        }
    }
}

pub fn black_max(
    black_input: &BlackTensor,
    black_dim: Option<usize>,
    black_keepdim: bool,
) -> BlackResult<BlackTensor> {
    let black_contig = black_input.black_contiguous()?;
    let black_data = black_contig.black_buffer.black_as_f32_slice();
    let black_dims = black_contig.black_shape.black_dims();

    match black_dim {
        None => {
            let black_n = black_contig.black_numel();
            let mut black_max_val = f32::NEG_INFINITY;
            for black_i in 0..black_n {
                if black_data[black_i] > black_max_val {
                    black_max_val = black_data[black_i];
                }
            }
            BlackTensor::black_scalar_f32(black_max_val)
        }
        Some(black_d) => {
            if black_d >= black_dims.len() {
                return Err(BlackError::BlackIndexError {
                    black_msg: format!("dim {} out of range", black_d),
                });
            }
            let black_dim_size = black_dims[black_d];
            let mut black_out_dims: Vec<usize> = black_dims.to_vec();
            if black_keepdim {
                black_out_dims[black_d] = 1;
            } else {
                black_out_dims.remove(black_d);
            }
            if black_out_dims.is_empty() {
                black_out_dims.push(1);
            }

            let black_out_shape = BlackShape::black_new(&black_out_dims);
            let black_out_numel = black_out_shape.black_numel();
            let mut black_out_data = vec![f32::NEG_INFINITY; black_out_numel];

            let black_outer: usize = black_dims[..black_d].iter().product();
            let black_inner: usize = black_dims[black_d + 1..].iter().product();

            for black_o in 0..black_outer {
                for black_i in 0..black_inner {
                    for black_k in 0..black_dim_size {
                        let black_idx =
                            black_o * black_dim_size * black_inner + black_k * black_inner + black_i;
                        let black_out_idx = black_o * black_inner + black_i;
                        if black_data[black_idx] > black_out_data[black_out_idx] {
                            black_out_data[black_out_idx] = black_data[black_idx];
                        }
                    }
                }
            }

            let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
            Ok(BlackTensor::black_new(
                black_buf,
                black_out_shape,
                BlackDType::BlackF32,
                BlackDevice::BlackCpu,
            ))
        }
    }
}

pub fn black_min(
    black_input: &BlackTensor,
    black_dim: Option<usize>,
    black_keepdim: bool,
) -> BlackResult<BlackTensor> {
    let black_contig = black_input.black_contiguous()?;
    let black_data = black_contig.black_buffer.black_as_f32_slice();
    let black_dims = black_contig.black_shape.black_dims();

    match black_dim {
        None => {
            let black_n = black_contig.black_numel();
            let mut black_min_val = f32::INFINITY;
            for black_i in 0..black_n {
                if black_data[black_i] < black_min_val {
                    black_min_val = black_data[black_i];
                }
            }
            BlackTensor::black_scalar_f32(black_min_val)
        }
        Some(black_d) => {
            if black_d >= black_dims.len() {
                return Err(BlackError::BlackIndexError {
                    black_msg: format!("dim {} out of range", black_d),
                });
            }
            let black_dim_size = black_dims[black_d];
            let mut black_out_dims: Vec<usize> = black_dims.to_vec();
            if black_keepdim {
                black_out_dims[black_d] = 1;
            } else {
                black_out_dims.remove(black_d);
            }
            if black_out_dims.is_empty() {
                black_out_dims.push(1);
            }

            let black_out_shape = BlackShape::black_new(&black_out_dims);
            let black_out_numel = black_out_shape.black_numel();
            let mut black_out_data = vec![f32::INFINITY; black_out_numel];

            let black_outer: usize = black_dims[..black_d].iter().product();
            let black_inner: usize = black_dims[black_d + 1..].iter().product();

            for black_o in 0..black_outer {
                for black_i in 0..black_inner {
                    for black_k in 0..black_dim_size {
                        let black_idx =
                            black_o * black_dim_size * black_inner + black_k * black_inner + black_i;
                        let black_out_idx = black_o * black_inner + black_i;
                        if black_data[black_idx] < black_out_data[black_out_idx] {
                            black_out_data[black_out_idx] = black_data[black_idx];
                        }
                    }
                }
            }

            let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
            Ok(BlackTensor::black_new(
                black_buf,
                black_out_shape,
                BlackDType::BlackF32,
                BlackDevice::BlackCpu,
            ))
        }
    }
}

pub fn black_argmax(black_input: &BlackTensor, black_dim: usize) -> BlackResult<BlackTensor> {
    let black_contig = black_input.black_contiguous()?;
    let black_data = black_contig.black_buffer.black_as_f32_slice();
    let black_dims = black_contig.black_shape.black_dims();

    if black_dim >= black_dims.len() {
        return Err(BlackError::BlackIndexError {
            black_msg: format!("dim {} out of range", black_dim),
        });
    }

    let black_dim_size = black_dims[black_dim];
    let mut black_out_dims: Vec<usize> = black_dims.to_vec();
    black_out_dims.remove(black_dim);
    if black_out_dims.is_empty() {
        black_out_dims.push(1);
    }

    let black_outer: usize = black_dims[..black_dim].iter().product();
    let black_inner: usize = black_dims[black_dim + 1..].iter().product();
    let black_out_numel = black_outer * black_inner;

    let mut black_out_data = vec![0.0f32; black_out_numel];

    for black_o in 0..black_outer {
        for black_i in 0..black_inner {
            let mut black_max_val = f32::NEG_INFINITY;
            let mut black_max_idx = 0usize;
            for black_k in 0..black_dim_size {
                let black_idx =
                    black_o * black_dim_size * black_inner + black_k * black_inner + black_i;
                if black_data[black_idx] > black_max_val {
                    black_max_val = black_data[black_idx];
                    black_max_idx = black_k;
                }
            }
            black_out_data[black_o * black_inner + black_i] = black_max_idx as f32;
        }
    }

    let black_out_shape = BlackShape::black_new(&black_out_dims);
    let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
    Ok(BlackTensor::black_new(
        black_buf,
        black_out_shape,
        BlackDType::BlackF32,
        BlackDevice::BlackCpu,
    ))
}

pub fn black_argmin(black_input: &BlackTensor, black_dim: usize) -> BlackResult<BlackTensor> {
    let black_contig = black_input.black_contiguous()?;
    let black_data = black_contig.black_buffer.black_as_f32_slice();
    let black_dims = black_contig.black_shape.black_dims();

    if black_dim >= black_dims.len() {
        return Err(BlackError::BlackIndexError {
            black_msg: format!("dim {} out of range", black_dim),
        });
    }

    let black_dim_size = black_dims[black_dim];
    let mut black_out_dims: Vec<usize> = black_dims.to_vec();
    black_out_dims.remove(black_dim);
    if black_out_dims.is_empty() {
        black_out_dims.push(1);
    }

    let black_outer: usize = black_dims[..black_dim].iter().product();
    let black_inner: usize = black_dims[black_dim + 1..].iter().product();
    let black_out_numel = black_outer * black_inner;

    let mut black_out_data = vec![0.0f32; black_out_numel];

    for black_o in 0..black_outer {
        for black_i in 0..black_inner {
            let mut black_min_val = f32::INFINITY;
            let mut black_min_idx = 0usize;
            for black_k in 0..black_dim_size {
                let black_idx =
                    black_o * black_dim_size * black_inner + black_k * black_inner + black_i;
                if black_data[black_idx] < black_min_val {
                    black_min_val = black_data[black_idx];
                    black_min_idx = black_k;
                }
            }
            black_out_data[black_o * black_inner + black_i] = black_min_idx as f32;
        }
    }

    let black_out_shape = BlackShape::black_new(&black_out_dims);
    let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
    Ok(BlackTensor::black_new(
        black_buf,
        black_out_shape,
        BlackDType::BlackF32,
        BlackDevice::BlackCpu,
    ))
}

pub fn black_var(
    black_input: &BlackTensor,
    black_dim: Option<usize>,
    black_correction: usize,
) -> BlackResult<BlackTensor> {
    let black_mean_tensor = black_mean(black_input, black_dim, true)?;
    let black_contig = black_input.black_contiguous()?;
    let black_data = black_contig.black_buffer.black_as_f32_slice();
    let black_dims = black_contig.black_shape.black_dims();

    match black_dim {
        None => {
            let black_n = black_contig.black_numel();
            let black_m = black_mean_tensor.black_item_f32()?;
            let mut black_acc = 0.0f32;
            for black_i in 0..black_n {
                let black_diff = black_data[black_i] - black_m;
                black_acc += black_diff * black_diff;
            }
            let black_denom = (black_n - black_correction) as f32;
            BlackTensor::black_scalar_f32(black_acc / black_denom)
        }
        Some(black_d) => {
            let black_dim_size = black_dims[black_d];
            let black_mean_data = black_mean_tensor.black_buffer.black_as_f32_slice();
            let mut black_out_dims: Vec<usize> = black_dims.to_vec();
            black_out_dims.remove(black_d);
            if black_out_dims.is_empty() {
                black_out_dims.push(1);
            }

            let black_outer: usize = black_dims[..black_d].iter().product();
            let black_inner: usize = black_dims[black_d + 1..].iter().product();
            let black_out_numel = black_outer * black_inner;
            let mut black_out_data = vec![0.0f32; black_out_numel];

            for black_o in 0..black_outer {
                for black_i in 0..black_inner {
                    let black_m_idx = black_o * black_inner + black_i;
                    let black_m = black_mean_data[black_m_idx];
                    let mut black_acc = 0.0f32;
                    for black_k in 0..black_dim_size {
                        let black_idx =
                            black_o * black_dim_size * black_inner + black_k * black_inner + black_i;
                        let black_diff = black_data[black_idx] - black_m;
                        black_acc += black_diff * black_diff;
                    }
                    black_out_data[black_m_idx] =
                        black_acc / (black_dim_size - black_correction) as f32;
                }
            }

            let black_out_shape = BlackShape::black_new(&black_out_dims);
            let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
            Ok(BlackTensor::black_new(
                black_buf,
                black_out_shape,
                BlackDType::BlackF32,
                BlackDevice::BlackCpu,
            ))
        }
    }
}

pub fn black_std(
    black_input: &BlackTensor,
    black_dim: Option<usize>,
    black_correction: usize,
) -> BlackResult<BlackTensor> {
    let mut black_var_tensor = black_var(black_input, black_dim, black_correction)?;
    let black_buf = std::sync::Arc::make_mut(&mut black_var_tensor.black_buffer);
    let black_slice = black_buf.black_as_f32_mut_slice();
    for black_v in black_slice.iter_mut() {
        *black_v = black_v.sqrt();
    }
    Ok(black_var_tensor)
}
