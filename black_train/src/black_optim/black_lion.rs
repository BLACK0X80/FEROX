use super::{BlackOptimizer, BlackOptimizerState};
use black_core::black_error::BlackResult;
use black_core::black_grad::black_var::BlackVar;
use black_core::black_tensor::BlackTensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub struct BlackLion {
    pub black_lr: f64,
    pub black_beta1: f64,
    pub black_beta2: f64,
    pub black_weight_decay: f64,
    black_m: Vec<BlackTensor>,
    black_initialized: bool,
}

impl BlackLion {
    pub fn black_new(
        black_lr: f64,
        black_beta1: f64,
        black_beta2: f64,
        black_weight_decay: f64,
    ) -> Self {
        BlackLion {
            black_lr,
            black_beta1,
            black_beta2,
            black_weight_decay,
            black_m: Vec::new(),
            black_initialized: false,
        }
    }
}

impl BlackOptimizer for BlackLion {
    fn black_step(&mut self, black_params: &mut [Arc<RwLock<BlackVar>>]) -> BlackResult<()> {
        if !self.black_initialized {
            for black_p in black_params.iter() {
                let black_p_r = black_p.read();
                let black_shape = black_p_r.black_data.black_shape.black_dims();
                self.black_m.push(BlackTensor::black_zeros(
                    black_shape,
                    black_core::black_dtype::BlackDType::BlackF32,
                )?);
            }
            self.black_initialized = true;
        }

        let black_lr_f = self.black_lr as f32;
        let black_b1 = self.black_beta1 as f32;
        let black_b2 = self.black_beta2 as f32;
        let black_wd = self.black_weight_decay as f32;

        for (black_idx, black_param) in black_params.iter().enumerate() {
            let mut black_p_w = black_param.write();
            let black_grad = match &black_p_w.black_grad {
                Some(black_g) => black_g.clone(),
                None => continue,
            };

            let black_param_buf = std::sync::Arc::make_mut(&mut black_p_w.black_data.black_buffer);
            let black_param_data = black_param_buf.black_as_f32_mut_slice();
            let black_grad_data = black_grad.black_buffer.black_as_f32_slice();
            let black_m_buf = std::sync::Arc::make_mut(&mut self.black_m[black_idx].black_buffer);
            let black_m_data = black_m_buf.black_as_f32_mut_slice();

            for black_i in 0..black_param_data.len() {
                if black_i >= black_grad_data.len() {
                    break;
                }
                let black_g = black_grad_data[black_i];

                black_param_data[black_i] *= 1.0 - black_lr_f * black_wd;

                let black_interp = black_b1 * black_m_data[black_i] + (1.0 - black_b1) * black_g;
                let black_update = if black_interp > 0.0 {
                    1.0f32
                } else if black_interp < 0.0 {
                    -1.0f32
                } else {
                    0.0f32
                };

                black_param_data[black_i] -= black_lr_f * black_update;

                black_m_data[black_i] = black_b2 * black_m_data[black_i] + (1.0 - black_b2) * black_g;
            }
        }

        Ok(())
    }

    fn black_zero_grad(&mut self, black_params: &mut [Arc<RwLock<BlackVar>>]) {
        for black_p in black_params.iter() {
            black_p.write().black_zero_grad();
        }
    }

    fn black_state_dict(&self) -> BlackOptimizerState {
        BlackOptimizerState {
            black_step_count: 0,
            black_state: HashMap::new(),
        }
    }

    fn black_load_state_dict(&mut self, _black_state: BlackOptimizerState) {}
}
