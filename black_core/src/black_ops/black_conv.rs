use crate::black_buffer::BlackBuffer;
use crate::black_device::BlackDevice;
use crate::black_dtype::BlackDType;
use crate::black_error::{BlackError, BlackResult};

use crate::black_shape::BlackShape;
use crate::black_tensor::BlackTensor;

#[allow(clippy::too_many_arguments)]
fn black_im2col(
    black_input: &[f32],
    black_batch: usize,
    black_channels: usize,
    black_h: usize,
    black_w: usize,
    black_kh: usize,
    black_kw: usize,
    black_stride_h: usize,
    black_stride_w: usize,
    black_pad_h: usize,
    black_pad_w: usize,
    black_dilation_h: usize,
    black_dilation_w: usize,
) -> (Vec<f32>, usize, usize) {
    let black_oh = (black_h + 2 * black_pad_h - black_dilation_h * (black_kh - 1) - 1) / black_stride_h + 1;
    let black_ow = (black_w + 2 * black_pad_w - black_dilation_w * (black_kw - 1) - 1) / black_stride_w + 1;
    let black_col_h = black_channels * black_kh * black_kw;
    let black_col_w = black_oh * black_ow;

    let mut black_col = vec![0.0f32; black_batch * black_col_h * black_col_w];

    for black_b in 0..black_batch {
        for black_c in 0..black_channels {
            for black_kh_i in 0..black_kh {
                for black_kw_i in 0..black_kw {
                    let black_col_row =
                        black_c * black_kh * black_kw + black_kh_i * black_kw + black_kw_i;
                    for black_oh_i in 0..black_oh {
                        for black_ow_i in 0..black_ow {
                            let black_ih =
                                black_oh_i * black_stride_h + black_kh_i * black_dilation_h;
                            let black_iw =
                                black_ow_i * black_stride_w + black_kw_i * black_dilation_w;

                            let black_ih_actual = black_ih as isize - black_pad_h as isize;
                            let black_iw_actual = black_iw as isize - black_pad_w as isize;

                            let black_val = if black_ih_actual >= 0
                                && black_ih_actual < black_h as isize
                                && black_iw_actual >= 0
                                && black_iw_actual < black_w as isize
                            {
                                let black_idx = black_b * black_channels * black_h * black_w
                                    + black_c * black_h * black_w
                                    + black_ih_actual as usize * black_w
                                    + black_iw_actual as usize;
                                black_input[black_idx]
                            } else {
                                0.0
                            };

                            let black_col_idx = black_b * black_col_h * black_col_w
                                + black_col_row * black_col_w
                                + black_oh_i * black_ow + black_ow_i;
                            black_col[black_col_idx] = black_val;
                        }
                    }
                }
            }
        }
    }

    (black_col, black_oh, black_ow)
}

pub fn black_conv2d(
    black_input: &BlackTensor,
    black_weight: &BlackTensor,
    black_bias: Option<&BlackTensor>,
    black_stride: (usize, usize),
    black_padding: (usize, usize),
    black_dilation: (usize, usize),
    black_groups: usize,
) -> BlackResult<BlackTensor> {
    let black_input_contig = black_input.black_contiguous()?;
    let black_weight_contig = black_weight.black_contiguous()?;
    let black_in_dims = black_input_contig.black_shape.black_dims();
    let black_w_dims = black_weight_contig.black_shape.black_dims();

    if black_in_dims.len() != 4 || black_w_dims.len() != 4 {
        return Err(BlackError::BlackShapeError {
            black_msg: "conv2d requires 4D input and weight".into(),
        });
    }

    let black_batch = black_in_dims[0];
    let black_in_channels = black_in_dims[1];
    let black_h = black_in_dims[2];
    let black_w = black_in_dims[3];
    let black_out_channels = black_w_dims[0];
    let black_kh = black_w_dims[2];
    let black_kw = black_w_dims[3];

    let black_in_data = black_input_contig.black_buffer.black_as_f32_slice();

    if black_groups == 1 {
        let (black_col, black_oh, black_ow) = black_im2col(
            black_in_data,
            black_batch,
            black_in_channels,
            black_h,
            black_w,
            black_kh,
            black_kw,
            black_stride.0,
            black_stride.1,
            black_padding.0,
            black_padding.1,
            black_dilation.0,
            black_dilation.1,
        );

        let black_col_h = black_in_channels * black_kh * black_kw;
        let black_col_w = black_oh * black_ow;

        let black_weight_data = black_weight_contig.black_buffer.black_as_f32_slice();
        let mut black_out_data = vec![0.0f32; black_batch * black_out_channels * black_oh * black_ow];

        for black_b in 0..black_batch {
            for black_oc in 0..black_out_channels {
                for black_col_j in 0..black_col_w {
                    let mut black_acc = 0.0f32;
                    for black_col_i in 0..black_col_h {
                        let black_w_idx = black_oc * black_col_h + black_col_i;
                        let black_c_idx =
                            black_b * black_col_h * black_col_w + black_col_i * black_col_w + black_col_j;
                        black_acc += black_weight_data[black_w_idx] * black_col[black_c_idx];
                    }
                    let black_out_idx =
                        black_b * black_out_channels * black_oh * black_ow + black_oc * black_oh * black_ow + black_col_j;
                    black_out_data[black_out_idx] = black_acc;
                }
            }
        }

        if let Some(black_bias_tensor) = black_bias {
            let black_bias_data = black_bias_tensor.black_buffer.black_as_f32_slice();
            for black_b in 0..black_batch {
                for (black_oc, &black_bias_val) in black_bias_data.iter().enumerate().take(black_out_channels) {
                    for black_j in 0..black_oh * black_ow {
                        let black_idx = black_b * black_out_channels * black_oh * black_ow
                            + black_oc * black_oh * black_ow
                            + black_j;
                        black_out_data[black_idx] += black_bias_val;
                    }
                }
            }
        }

        let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
        Ok(BlackTensor::black_new(
            black_buf,
            BlackShape::black_new(&[black_batch, black_out_channels, black_oh, black_ow]),
            BlackDType::BlackF32,
            BlackDevice::BlackCpu,
        ))
    } else {
        let black_oh = (black_h + 2 * black_padding.0 - black_dilation.0 * (black_kh - 1) - 1) / black_stride.0 + 1;
        let black_ow = (black_w + 2 * black_padding.1 - black_dilation.1 * (black_kw - 1) - 1) / black_stride.1 + 1;
        let mut black_out_data = vec![0.0f32; black_batch * black_out_channels * black_oh * black_ow];

        let black_in_per_group = black_in_channels / black_groups;
        let black_out_per_group = black_out_channels / black_groups;
        let black_weight_data = black_weight_contig.black_buffer.black_as_f32_slice();

        for black_b in 0..black_batch {
            for black_g in 0..black_groups {
                for black_oc in 0..black_out_per_group {
                    let black_oc_abs = black_g * black_out_per_group + black_oc;
                    for black_oh_i in 0..black_oh {
                        for black_ow_i in 0..black_ow {
                            let mut black_acc = 0.0f32;
                            for black_ic in 0..black_in_per_group {
                                let black_ic_abs = black_g * black_in_per_group + black_ic;
                                for black_kh_i in 0..black_kh {
                                    for black_kw_i in 0..black_kw {
                                        let black_ih = black_oh_i * black_stride.0 + black_kh_i * black_dilation.0;
                                        let black_iw = black_ow_i * black_stride.1 + black_kw_i * black_dilation.1;
                                        let black_ih_a = black_ih as isize - black_padding.0 as isize;
                                        let black_iw_a = black_iw as isize - black_padding.1 as isize;

                                        if black_ih_a >= 0
                                            && black_ih_a < black_h as isize
                                            && black_iw_a >= 0
                                            && black_iw_a < black_w as isize
                                        {
                                            let black_in_idx = black_b * black_in_channels * black_h * black_w
                                                + black_ic_abs * black_h * black_w
                                                + black_ih_a as usize * black_w
                                                + black_iw_a as usize;
                                            let black_w_idx = black_oc_abs * black_in_per_group * black_kh * black_kw
                                                + black_ic * black_kh * black_kw
                                                + black_kh_i * black_kw
                                                + black_kw_i;
                                            black_acc += black_in_data[black_in_idx] * black_weight_data[black_w_idx];
                                        }
                                    }
                                }
                            }
                            let black_out_idx = black_b * black_out_channels * black_oh * black_ow
                                + black_oc_abs * black_oh * black_ow
                                + black_oh_i * black_ow
                                + black_ow_i;
                            black_out_data[black_out_idx] = black_acc;
                        }
                    }
                }
            }
        }

        if let Some(black_bias_tensor) = black_bias {
            let black_bias_data = black_bias_tensor.black_buffer.black_as_f32_slice();
            for black_b in 0..black_batch {
                for (black_oc, &black_bias_val) in black_bias_data.iter().enumerate().take(black_out_channels) {
                    for black_j in 0..black_oh * black_ow {
                        let black_idx = black_b * black_out_channels * black_oh * black_ow
                            + black_oc * black_oh * black_ow + black_j;
                        black_out_data[black_idx] += black_bias_val;
                    }
                }
            }
        }

        let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
        Ok(BlackTensor::black_new(
            black_buf,
            BlackShape::black_new(&[black_batch, black_out_channels, black_oh, black_ow]),
            BlackDType::BlackF32,
            BlackDevice::BlackCpu,
        ))
    }
}

pub fn black_conv1d(
    black_input: &BlackTensor,
    black_weight: &BlackTensor,
    black_bias: Option<&BlackTensor>,
    black_stride: usize,
    black_padding: usize,
    black_dilation: usize,
    _black_groups: usize,
) -> BlackResult<BlackTensor> {
    let black_in_dims = black_input.black_shape.black_dims();
    if black_in_dims.len() != 3 {
        return Err(BlackError::BlackShapeError {
            black_msg: "conv1d requires 3D input [N, C, L]".into(),
        });
    }
    let black_n = black_in_dims[0];
    let black_c = black_in_dims[1];
    let black_l = black_in_dims[2];

    let black_in_4d = black_input.black_reshape(&[black_n, black_c, 1, black_l])?;
    let black_w_dims = black_weight.black_shape.black_dims();
    let black_w_4d = black_weight.black_reshape(&[black_w_dims[0], black_w_dims[1], 1, black_w_dims[2]])?;

    let black_out_4d = black_conv2d(
        &black_in_4d,
        &black_w_4d,
        black_bias,
        (1, black_stride),
        (0, black_padding),
        (1, black_dilation),
        _black_groups,
    )?;

    let black_out_dims = black_out_4d.black_shape.black_dims();
    black_out_4d.black_reshape(&[black_out_dims[0], black_out_dims[1], black_out_dims[3]])
}

pub fn black_conv3d(
    black_input: &BlackTensor,
    black_weight: &BlackTensor,
    black_bias: Option<&BlackTensor>,
    black_stride: (usize, usize, usize),
    black_padding: (usize, usize, usize),
    black_dilation: (usize, usize, usize),
    _black_groups: usize,
) -> BlackResult<BlackTensor> {
    let black_in_dims = black_input.black_shape.black_dims();
    if black_in_dims.len() != 5 {
        return Err(BlackError::BlackShapeError {
            black_msg: "conv3d requires 5D input [N, C, D, H, W]".into(),
        });
    }

    let black_contig = black_input.black_contiguous()?;
    let black_w_contig = black_weight.black_contiguous()?;
    let black_in_data = black_contig.black_buffer.black_as_f32_slice();
    let black_w_data = black_w_contig.black_buffer.black_as_f32_slice();
    let black_w_shape = black_w_contig.black_shape.black_dims();

    let black_batch = black_in_dims[0];
    let black_in_channels = black_in_dims[1];
    let black_d = black_in_dims[2];
    let black_h = black_in_dims[3];
    let black_w = black_in_dims[4];
    let black_out_channels = black_w_shape[0];
    let black_kd = black_w_shape[2];
    let black_kh = black_w_shape[3];
    let black_kw = black_w_shape[4];

    let black_od = (black_d + 2 * black_padding.0 - black_dilation.0 * (black_kd - 1) - 1) / black_stride.0 + 1;
    let black_oh = (black_h + 2 * black_padding.1 - black_dilation.1 * (black_kh - 1) - 1) / black_stride.1 + 1;
    let black_ow = (black_w + 2 * black_padding.2 - black_dilation.2 * (black_kw - 1) - 1) / black_stride.2 + 1;

    let mut black_out_data = vec![0.0f32; black_batch * black_out_channels * black_od * black_oh * black_ow];

    for black_b in 0..black_batch {
        for black_oc in 0..black_out_channels {
            for black_od_i in 0..black_od {
                for black_oh_i in 0..black_oh {
                    for black_ow_i in 0..black_ow {
                        let mut black_acc = 0.0f32;
                        for black_ic in 0..black_in_channels {
                            for black_kd_i in 0..black_kd {
                                for black_kh_i in 0..black_kh {
                                    for black_kw_i in 0..black_kw {
                                        let black_id = (black_od_i * black_stride.0 + black_kd_i * black_dilation.0) as isize - black_padding.0 as isize;
                                        let black_ih = (black_oh_i * black_stride.1 + black_kh_i * black_dilation.1) as isize - black_padding.1 as isize;
                                        let black_iw = (black_ow_i * black_stride.2 + black_kw_i * black_dilation.2) as isize - black_padding.2 as isize;

                                        if black_id >= 0 && black_id < black_d as isize
                                            && black_ih >= 0 && black_ih < black_h as isize
                                            && black_iw >= 0 && black_iw < black_w as isize
                                        {
                                            let black_in_idx = black_b * black_in_channels * black_d * black_h * black_w
                                                + black_ic * black_d * black_h * black_w
                                                + black_id as usize * black_h * black_w
                                                + black_ih as usize * black_w
                                                + black_iw as usize;
                                            let black_w_idx = black_oc * black_in_channels * black_kd * black_kh * black_kw
                                                + black_ic * black_kd * black_kh * black_kw
                                                + black_kd_i * black_kh * black_kw
                                                + black_kh_i * black_kw
                                                + black_kw_i;
                                            black_acc += black_in_data[black_in_idx] * black_w_data[black_w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        let black_out_idx = black_b * black_out_channels * black_od * black_oh * black_ow
                            + black_oc * black_od * black_oh * black_ow
                            + black_od_i * black_oh * black_ow
                            + black_oh_i * black_ow
                            + black_ow_i;
                        black_out_data[black_out_idx] = black_acc;
                    }
                }
            }
        }
    }

    if let Some(black_bias_tensor) = black_bias {
        let black_bias_data = black_bias_tensor.black_buffer.black_as_f32_slice();
        for black_b in 0..black_batch {
            for (black_oc, &black_bias_val) in black_bias_data.iter().enumerate().take(black_out_channels) {
                for black_j in 0..black_od * black_oh * black_ow {
                    let black_idx = black_b * black_out_channels * black_od * black_oh * black_ow
                        + black_oc * black_od * black_oh * black_ow + black_j;
                    black_out_data[black_idx] += black_bias_val;
                }
            }
        }
    }

    let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
    Ok(BlackTensor::black_new(
        black_buf,
        BlackShape::black_new(&[black_batch, black_out_channels, black_od, black_oh, black_ow]),
        BlackDType::BlackF32,
        BlackDevice::BlackCpu,
    ))
}

pub fn black_conv_transpose2d(
    black_input: &BlackTensor,
    black_weight: &BlackTensor,
    black_bias: Option<&BlackTensor>,
    black_stride: (usize, usize),
    black_padding: (usize, usize),
    black_output_padding: (usize, usize),
) -> BlackResult<BlackTensor> {
    let black_in_dims = black_input.black_shape.black_dims();
    let black_w_dims = black_weight.black_shape.black_dims();

    if black_in_dims.len() != 4 || black_w_dims.len() != 4 {
        return Err(BlackError::BlackShapeError {
            black_msg: "conv_transpose2d requires 4D tensors".into(),
        });
    }

    let black_batch = black_in_dims[0];
    let black_in_channels = black_in_dims[1];
    let black_ih = black_in_dims[2];
    let black_iw = black_in_dims[3];
    let black_out_channels = black_w_dims[1];
    let black_kh = black_w_dims[2];
    let black_kw = black_w_dims[3];

    let black_oh = (black_ih - 1) * black_stride.0 - 2 * black_padding.0 + black_kh + black_output_padding.0;
    let black_ow = (black_iw - 1) * black_stride.1 - 2 * black_padding.1 + black_kw + black_output_padding.1;

    let black_in_data = black_input.black_contiguous()?.black_buffer.black_as_f32_slice().to_vec();
    let black_w_data = black_weight.black_contiguous()?.black_buffer.black_as_f32_slice().to_vec();

    let mut black_out_data = vec![0.0f32; black_batch * black_out_channels * black_oh * black_ow];

    for black_b in 0..black_batch {
        for black_ic in 0..black_in_channels {
            for black_ih_i in 0..black_ih {
                for black_iw_i in 0..black_iw {
                    let black_in_val = black_in_data[
                        black_b * black_in_channels * black_ih * black_iw
                        + black_ic * black_ih * black_iw
                        + black_ih_i * black_iw + black_iw_i
                    ];
                    for black_oc in 0..black_out_channels {
                        for black_kh_i in 0..black_kh {
                            for black_kw_i in 0..black_kw {
                                let black_oh_i = black_ih_i * black_stride.0 + black_kh_i;
                                let black_ow_i = black_iw_i * black_stride.1 + black_kw_i;
                                if black_oh_i >= black_padding.0
                                    && black_oh_i < black_oh + black_padding.0
                                    && black_ow_i >= black_padding.1
                                    && black_ow_i < black_ow + black_padding.1
                                {
                                    let black_w_idx = black_ic * black_out_channels * black_kh * black_kw
                                        + black_oc * black_kh * black_kw
                                        + black_kh_i * black_kw + black_kw_i;
                                    let black_out_idx = black_b * black_out_channels * black_oh * black_ow
                                        + black_oc * black_oh * black_ow
                                        + (black_oh_i - black_padding.0) * black_ow
                                        + (black_ow_i - black_padding.1);
                                    black_out_data[black_out_idx] += black_in_val * black_w_data[black_w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(black_bias_tensor) = black_bias {
        let black_bias_data = black_bias_tensor.black_contiguous()?.black_buffer.black_as_f32_slice().to_vec();
        for black_b in 0..black_batch {
            for (black_oc, &black_bias_val) in black_bias_data.iter().enumerate().take(black_out_channels) {
                for black_j in 0..black_oh * black_ow {
                    let black_idx = black_b * black_out_channels * black_oh * black_ow
                        + black_oc * black_oh * black_ow + black_j;
                    black_out_data[black_idx] += black_bias_val;
                }
            }
        }
    }

    let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
    Ok(BlackTensor::black_new(
        black_buf,
        BlackShape::black_new(&[black_batch, black_out_channels, black_oh, black_ow]),
        BlackDType::BlackF32,
        BlackDevice::BlackCpu,
    ))
}

pub fn black_depthwise_conv2d(
    black_input: &BlackTensor,
    black_weight: &BlackTensor,
    black_bias: Option<&BlackTensor>,
    black_stride: (usize, usize),
    black_padding: (usize, usize),
    black_dilation: (usize, usize),
) -> BlackResult<BlackTensor> {
    let black_in_dims = black_input.black_shape.black_dims();
    let black_channels = black_in_dims[1];
    black_conv2d(
        black_input,
        black_weight,
        black_bias,
        black_stride,
        black_padding,
        black_dilation,
        black_channels,
    )
}
