use crate::black_grad::black_var::{BlackGradFn, BlackVar};
use crate::black_ops::black_elementwise;
use crate::black_ops::black_matmul;
use crate::black_tensor::BlackTensor;
use std::sync::Arc;
use parking_lot::RwLock;

pub struct BlackAddGradFn {
    pub black_input_a: Arc<RwLock<BlackVar>>,
    pub black_input_b: Arc<RwLock<BlackVar>>,
}

impl BlackGradFn for BlackAddGradFn {
    fn black_backward(&self, black_upstream_grad: &BlackTensor) -> Vec<Option<BlackTensor>> {
        vec![
            Some(black_upstream_grad.clone()),
            Some(black_upstream_grad.clone()),
        ]
    }

    fn black_inputs(&self) -> Vec<Arc<RwLock<BlackVar>>> {
        vec![
            Arc::clone(&self.black_input_a),
            Arc::clone(&self.black_input_b),
        ]
    }
}

pub struct BlackSubGradFn {
    pub black_input_a: Arc<RwLock<BlackVar>>,
    pub black_input_b: Arc<RwLock<BlackVar>>,
}

impl BlackGradFn for BlackSubGradFn {
    fn black_backward(&self, black_upstream_grad: &BlackTensor) -> Vec<Option<BlackTensor>> {
        let black_neg_grad = black_elementwise::black_neg(black_upstream_grad).ok();
        vec![Some(black_upstream_grad.clone()), black_neg_grad]
    }

    fn black_inputs(&self) -> Vec<Arc<RwLock<BlackVar>>> {
        vec![
            Arc::clone(&self.black_input_a),
            Arc::clone(&self.black_input_b),
        ]
    }
}

pub struct BlackMulGradFn {
    pub black_input_a: Arc<RwLock<BlackVar>>,
    pub black_input_b: Arc<RwLock<BlackVar>>,
    pub black_saved_a: BlackTensor,
    pub black_saved_b: BlackTensor,
}

impl BlackGradFn for BlackMulGradFn {
    fn black_backward(&self, black_upstream_grad: &BlackTensor) -> Vec<Option<BlackTensor>> {
        let black_grad_a = black_elementwise::black_mul(black_upstream_grad, &self.black_saved_b).ok();
        let black_grad_b = black_elementwise::black_mul(black_upstream_grad, &self.black_saved_a).ok();
        vec![black_grad_a, black_grad_b]
    }

    fn black_inputs(&self) -> Vec<Arc<RwLock<BlackVar>>> {
        vec![
            Arc::clone(&self.black_input_a),
            Arc::clone(&self.black_input_b),
        ]
    }
}

pub struct BlackDivGradFn {
    pub black_input_a: Arc<RwLock<BlackVar>>,
    pub black_input_b: Arc<RwLock<BlackVar>>,
    pub black_saved_a: BlackTensor,
    pub black_saved_b: BlackTensor,
}

impl BlackGradFn for BlackDivGradFn {
    fn black_backward(&self, black_upstream_grad: &BlackTensor) -> Vec<Option<BlackTensor>> {
        let black_grad_a = black_elementwise::black_div(black_upstream_grad, &self.black_saved_b).ok();

        let black_neg_a = black_elementwise::black_neg(&self.black_saved_a).ok();
        let black_b_sq = black_elementwise::black_mul(&self.black_saved_b, &self.black_saved_b).ok();
        let black_grad_b = if let (Some(black_na), Some(black_bsq)) = (black_neg_a, black_b_sq) {
            let black_ratio = black_elementwise::black_div(&black_na, &black_bsq).ok();
            if let Some(black_r) = black_ratio {
                black_elementwise::black_mul(black_upstream_grad, &black_r).ok()
            } else {
                None
            }
        } else {
            None
        };

        vec![black_grad_a, black_grad_b]
    }

    fn black_inputs(&self) -> Vec<Arc<RwLock<BlackVar>>> {
        vec![
            Arc::clone(&self.black_input_a),
            Arc::clone(&self.black_input_b),
        ]
    }
}

pub struct BlackMatmulGradFn {
    pub black_input_a: Arc<RwLock<BlackVar>>,
    pub black_input_b: Arc<RwLock<BlackVar>>,
    pub black_saved_a: BlackTensor,
    pub black_saved_b: BlackTensor,
}

impl BlackGradFn for BlackMatmulGradFn {
    fn black_backward(&self, black_upstream_grad: &BlackTensor) -> Vec<Option<BlackTensor>> {
        let black_b_t = self.black_saved_b.black_t().ok();
        let black_grad_a = if let Some(black_bt) = black_b_t {
            black_matmul::black_matmul(black_upstream_grad, &black_bt).ok()
        } else {
            None
        };

        let black_a_t = self.black_saved_a.black_t().ok();
        let black_grad_b = if let Some(black_at) = black_a_t {
            black_matmul::black_matmul(&black_at, black_upstream_grad).ok()
        } else {
            None
        };

        vec![black_grad_a, black_grad_b]
    }

    fn black_inputs(&self) -> Vec<Arc<RwLock<BlackVar>>> {
        vec![
            Arc::clone(&self.black_input_a),
            Arc::clone(&self.black_input_b),
        ]
    }
}

pub struct BlackReluGradFn {
    pub black_input: Arc<RwLock<BlackVar>>,
    pub black_saved_input: BlackTensor,
}

impl BlackGradFn for BlackReluGradFn {
    fn black_backward(&self, black_upstream_grad: &BlackTensor) -> Vec<Option<BlackTensor>> {
        let black_data = self.black_saved_input.black_buffer.black_as_f32_slice();
        let black_grad_data = black_upstream_grad.black_buffer.black_as_f32_slice();
        let black_numel = self.black_saved_input.black_numel();

        let mut black_out = vec![0.0f32; black_numel];
        for black_i in 0..black_numel {
            black_out[black_i] = if black_data[black_i] > 0.0 {
                black_grad_data[black_i]
            } else {
                0.0
            };
        }

        let black_buf = crate::black_buffer::BlackBuffer::black_from_vec_f32(black_out).ok();
        let black_result = black_buf.map(|black_b| {
            BlackTensor::black_new(
                black_b,
                self.black_saved_input.black_shape.clone(),
                crate::black_dtype::BlackDType::BlackF32,
                crate::black_device::BlackDevice::BlackCpu,
            )
        });

        vec![black_result]
    }

    fn black_inputs(&self) -> Vec<Arc<RwLock<BlackVar>>> {
        vec![Arc::clone(&self.black_input)]
    }
}

pub struct BlackGeluGradFn {
    pub black_input: Arc<RwLock<BlackVar>>,
    pub black_saved_input: BlackTensor,
}

impl BlackGradFn for BlackGeluGradFn {
    fn black_backward(&self, black_upstream_grad: &BlackTensor) -> Vec<Option<BlackTensor>> {
        let black_data = self.black_saved_input.black_buffer.black_as_f32_slice();
        let black_grad_data = black_upstream_grad.black_buffer.black_as_f32_slice();
        let black_numel = self.black_saved_input.black_numel();

        let mut black_out = vec![0.0f32; black_numel];
        let black_sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();

        for black_i in 0..black_numel {
            let black_x = black_data[black_i];
            let black_inner = black_sqrt_2_pi * (black_x + 0.044715 * black_x.powi(3));
            let black_tanh_val = black_inner.tanh();
            let black_sech2 = 1.0 - black_tanh_val * black_tanh_val;
            let black_inner_deriv = black_sqrt_2_pi * (1.0 + 3.0 * 0.044715 * black_x * black_x);
            let black_grad_val = 0.5 * (1.0 + black_tanh_val) + 0.5 * black_x * black_sech2 * black_inner_deriv;
            black_out[black_i] = black_grad_data[black_i] * black_grad_val;
        }

        let black_buf = crate::black_buffer::BlackBuffer::black_from_vec_f32(black_out).ok();
        let black_result = black_buf.map(|black_b| {
            BlackTensor::black_new(
                black_b,
                self.black_saved_input.black_shape.clone(),
                crate::black_dtype::BlackDType::BlackF32,
                crate::black_device::BlackDevice::BlackCpu,
            )
        });

        vec![black_result]
    }

    fn black_inputs(&self) -> Vec<Arc<RwLock<BlackVar>>> {
        vec![Arc::clone(&self.black_input)]
    }
}

pub struct BlackSigmoidGradFn {
    pub black_input: Arc<RwLock<BlackVar>>,
    pub black_saved_output: BlackTensor,
}

impl BlackGradFn for BlackSigmoidGradFn {
    fn black_backward(&self, black_upstream_grad: &BlackTensor) -> Vec<Option<BlackTensor>> {
        let black_out_data = self.black_saved_output.black_buffer.black_as_f32_slice();
        let black_grad_data = black_upstream_grad.black_buffer.black_as_f32_slice();
        let black_numel = self.black_saved_output.black_numel();

        let mut black_result = vec![0.0f32; black_numel];
        for black_i in 0..black_numel {
            let black_s = black_out_data[black_i];
            black_result[black_i] = black_grad_data[black_i] * black_s * (1.0 - black_s);
        }

        let black_buf = crate::black_buffer::BlackBuffer::black_from_vec_f32(black_result).ok();
        let black_tensor = black_buf.map(|black_b| {
            BlackTensor::black_new(
                black_b,
                self.black_saved_output.black_shape.clone(),
                crate::black_dtype::BlackDType::BlackF32,
                crate::black_device::BlackDevice::BlackCpu,
            )
        });

        vec![black_tensor]
    }

    fn black_inputs(&self) -> Vec<Arc<RwLock<BlackVar>>> {
        vec![Arc::clone(&self.black_input)]
    }
}

pub struct BlackTanhGradFn {
    pub black_input: Arc<RwLock<BlackVar>>,
    pub black_saved_output: BlackTensor,
}

impl BlackGradFn for BlackTanhGradFn {
    fn black_backward(&self, black_upstream_grad: &BlackTensor) -> Vec<Option<BlackTensor>> {
        let black_out_data = self.black_saved_output.black_buffer.black_as_f32_slice();
        let black_grad_data = black_upstream_grad.black_buffer.black_as_f32_slice();
        let black_numel = self.black_saved_output.black_numel();

        let mut black_result = vec![0.0f32; black_numel];
        for black_i in 0..black_numel {
            let black_t = black_out_data[black_i];
            black_result[black_i] = black_grad_data[black_i] * (1.0 - black_t * black_t);
        }

        let black_buf = crate::black_buffer::BlackBuffer::black_from_vec_f32(black_result).ok();
        let black_tensor = black_buf.map(|black_b| {
            BlackTensor::black_new(
                black_b,
                self.black_saved_output.black_shape.clone(),
                crate::black_dtype::BlackDType::BlackF32,
                crate::black_device::BlackDevice::BlackCpu,
            )
        });

        vec![black_tensor]
    }

    fn black_inputs(&self) -> Vec<Arc<RwLock<BlackVar>>> {
        vec![Arc::clone(&self.black_input)]
    }
}

pub struct BlackExpGradFn {
    pub black_input: Arc<RwLock<BlackVar>>,
    pub black_saved_output: BlackTensor,
}

impl BlackGradFn for BlackExpGradFn {
    fn black_backward(&self, black_upstream_grad: &BlackTensor) -> Vec<Option<BlackTensor>> {
        let black_result = black_elementwise::black_mul(black_upstream_grad, &self.black_saved_output).ok();
        vec![black_result]
    }

    fn black_inputs(&self) -> Vec<Arc<RwLock<BlackVar>>> {
        vec![Arc::clone(&self.black_input)]
    }
}
