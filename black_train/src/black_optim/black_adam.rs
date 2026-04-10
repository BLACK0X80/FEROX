use super::{BlackOptimizer, BlackOptimizerState};
use black_core::black_error::BlackResult;
use black_core::black_grad::black_var::BlackVar;
use black_core::black_tensor::BlackTensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub struct BlackAdam {
    pub black_lr: f64,
    pub black_beta1: f64,
    pub black_beta2: f64,
    pub black_eps: f64,
    pub black_weight_decay: f64,
    black_step_count: u64,
    black_m: Vec<BlackTensor>,
    black_v: Vec<BlackTensor>,
    black_initialized: bool,
}

impl BlackAdam {
    pub fn black_new(
        black_lr: f64,
        black_beta1: f64,
        black_beta2: f64,
        black_eps: f64,
        black_weight_decay: f64,
    ) -> Self {
        BlackAdam {
            black_lr,
            black_beta1,
            black_beta2,
            black_eps,
            black_weight_decay,
            black_step_count: 0,
            black_m: Vec::new(),
            black_v: Vec::new(),
            black_initialized: false,
        }
    }
}

impl BlackOptimizer for BlackAdam {
    fn black_step(&mut self, black_params: &mut [Arc<RwLock<BlackVar>>]) -> BlackResult<()> {
        if !self.black_initialized {
            for black_p in black_params.iter() {
                let black_p_r = black_p.read();
                let black_shape = black_p_r.black_data.black_shape.black_dims();
                self.black_m.push(BlackTensor::black_zeros(
                    black_shape,
                    black_core::black_dtype::BlackDType::BlackF32,
                )?);
                self.black_v.push(BlackTensor::black_zeros(
                    black_shape,
                    black_core::black_dtype::BlackDType::BlackF32,
                )?);
            }
            self.black_initialized = true;
        }

        self.black_step_count += 1;
        let black_t = self.black_step_count as f64;
        let black_bc1 = 1.0 - self.black_beta1.powf(black_t);
        let black_bc2 = 1.0 - self.black_beta2.powf(black_t);
        let black_lr_t = self.black_lr * black_bc2.sqrt() / black_bc1;

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
            let black_v_buf = std::sync::Arc::make_mut(&mut self.black_v[black_idx].black_buffer);
            let black_v_data = black_v_buf.black_as_f32_mut_slice();

            let black_b1 = self.black_beta1 as f32;
            let black_b2 = self.black_beta2 as f32;
            let black_e = self.black_eps as f32;
            let black_wd = self.black_weight_decay as f32;
            let black_lr_f = black_lr_t as f32;

            for black_i in 0..black_param_data.len() {
                if black_i >= black_grad_data.len() {
                    break;
                }
                let black_g = black_grad_data[black_i] + black_wd * black_param_data[black_i];
                black_m_data[black_i] = black_b1 * black_m_data[black_i] + (1.0 - black_b1) * black_g;
                black_v_data[black_i] = black_b2 * black_v_data[black_i] + (1.0 - black_b2) * black_g * black_g;
                black_param_data[black_i] -= black_lr_f * black_m_data[black_i] / (black_v_data[black_i].sqrt() + black_e);
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
            black_step_count: self.black_step_count,
            black_state: HashMap::new(),
        }
    }

    fn black_load_state_dict(&mut self, black_state: BlackOptimizerState) {
        self.black_step_count = black_state.black_step_count;
    }
}
