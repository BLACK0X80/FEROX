use crate::black_buffer::BlackBuffer;
use crate::black_device::BlackDevice;
use crate::black_dtype::BlackDType;
use crate::black_error::BlackResult;
use crate::black_tensor::BlackTensor;

macro_rules! black_binary_op_f32 {
    ($black_fn_name:ident, $black_op:expr) => {
        pub fn $black_fn_name(
            black_a: &BlackTensor,
            black_b: &BlackTensor,
        ) -> BlackResult<BlackTensor> {
            let black_out_shape = black_a
                .black_shape
                .black_broadcast_with(&black_b.black_shape)?;
            let black_a_contig = black_a.black_contiguous()?;
            let black_b_contig = black_b.black_contiguous()?;
            let black_a_data = black_a_contig.black_buffer.black_as_f32_slice();
            let black_b_data = black_b_contig.black_buffer.black_as_f32_slice();
            let black_out_numel = black_out_shape.black_numel();
            let black_a_numel = black_a_contig.black_numel();
            let black_b_numel = black_b_contig.black_numel();

            let mut black_out_data = vec![0.0f32; black_out_numel];

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    black_binary_op_avx2(
                        black_a_data,
                        black_b_data,
                        &mut black_out_data,
                        black_a_numel,
                        black_b_numel,
                        $black_op,
                    );
                } else {
                    black_binary_op_scalar(
                        black_a_data,
                        black_b_data,
                        &mut black_out_data,
                        black_a_numel,
                        black_b_numel,
                        $black_op,
                    );
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                black_binary_op_scalar(
                    black_a_data,
                    black_b_data,
                    &mut black_out_data,
                    black_a_numel,
                    black_b_numel,
                    $black_op,
                );
            }

            let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
            Ok(BlackTensor::black_new(
                black_buf,
                black_out_shape,
                BlackDType::BlackF32,
                BlackDevice::BlackCpu,
            ))
        }
    };
}

fn black_binary_op_scalar(
    black_a: &[f32],
    black_b: &[f32],
    black_out: &mut [f32],
    black_a_len: usize,
    black_b_len: usize,
    black_op: fn(f32, f32) -> f32,
) {
    for black_i in 0..black_out.len() {
        let black_ai = black_a[black_i % black_a_len];
        let black_bi = black_b[black_i % black_b_len];
        black_out[black_i] = black_op(black_ai, black_bi);
    }
}

#[cfg(target_arch = "x86_64")]
fn black_binary_op_avx2(
    black_a: &[f32],
    black_b: &[f32],
    black_out: &mut [f32],
    black_a_len: usize,
    black_b_len: usize,
    black_op: fn(f32, f32) -> f32,
) {
    for black_i in 0..black_out.len() {
        let black_ai = black_a[black_i % black_a_len];
        let black_bi = black_b[black_i % black_b_len];
        black_out[black_i] = black_op(black_ai, black_bi);
    }
}

black_binary_op_f32!(black_add, |black_x: f32, black_y: f32| black_x + black_y);
black_binary_op_f32!(black_sub, |black_x: f32, black_y: f32| black_x - black_y);
black_binary_op_f32!(black_mul, |black_x: f32, black_y: f32| black_x * black_y);
black_binary_op_f32!(black_div, |black_x: f32, black_y: f32| black_x / black_y);
black_binary_op_f32!(black_pow, |black_x: f32, black_y: f32| black_x.powf(black_y));

macro_rules! black_unary_op_f32 {
    ($black_fn_name:ident, $black_op:expr) => {
        pub fn $black_fn_name(black_input: &BlackTensor) -> BlackResult<BlackTensor> {
            let black_contig = black_input.black_contiguous()?;
            let black_data = black_contig.black_buffer.black_as_f32_slice();
            let black_numel = black_contig.black_numel();

            let mut black_out_data = vec![0.0f32; black_numel];

            for black_i in 0..black_numel {
                black_out_data[black_i] = $black_op(black_data[black_i]);
            }

            let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
            Ok(BlackTensor::black_new(
                black_buf,
                black_contig.black_shape.clone(),
                BlackDType::BlackF32,
                BlackDevice::BlackCpu,
            ))
        }
    };
}

black_unary_op_f32!(black_relu, |black_x: f32| if black_x > 0.0 {
    black_x
} else {
    0.0
});

black_unary_op_f32!(black_gelu, |black_x: f32| {
    let black_cdf =
        0.5 * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (black_x + 0.044715 * black_x.powi(3))).tanh());
    black_x * black_cdf
});

black_unary_op_f32!(black_silu, |black_x: f32| {
    black_x / (1.0 + (-black_x).exp())
});

black_unary_op_f32!(black_tanh, |black_x: f32| black_x.tanh());
black_unary_op_f32!(black_sigmoid, |black_x: f32| 1.0 / (1.0 + (-black_x).exp()));
black_unary_op_f32!(black_exp, |black_x: f32| black_x.exp());
black_unary_op_f32!(black_log, |black_x: f32| black_x.ln());
black_unary_op_f32!(black_sqrt, |black_x: f32| black_x.sqrt());
black_unary_op_f32!(black_abs, |black_x: f32| black_x.abs());
black_unary_op_f32!(black_neg, |black_x: f32| -black_x);

pub fn black_clamp(
    black_input: &BlackTensor,
    black_min: f32,
    black_max: f32,
) -> BlackResult<BlackTensor> {
    let black_contig = black_input.black_contiguous()?;
    let black_data = black_contig.black_buffer.black_as_f32_slice();
    let black_numel = black_contig.black_numel();

    let mut black_out_data = vec![0.0f32; black_numel];
    for black_i in 0..black_numel {
        black_out_data[black_i] = black_data[black_i].clamp(black_min, black_max);
    }

    let black_buf = BlackBuffer::black_from_vec_f32(black_out_data)?;
    Ok(BlackTensor::black_new(
        black_buf,
        black_contig.black_shape.clone(),
        BlackDType::BlackF32,
        BlackDevice::BlackCpu,
    ))
}

pub fn black_add_inplace(black_a: &mut BlackTensor, black_b: &BlackTensor) -> BlackResult<()> {
    let black_a_buf = std::sync::Arc::make_mut(&mut black_a.black_buffer);
    let black_a_data = black_a_buf.black_as_f32_mut_slice();
    let black_b_contig = black_b.black_contiguous()?;
    let black_b_data = black_b_contig.black_buffer.black_as_f32_slice();
    let black_b_numel = black_b_contig.black_numel();

    for black_i in 0..black_a_data.len() {
        black_a_data[black_i] += black_b_data[black_i % black_b_numel];
    }
    Ok(())
}
