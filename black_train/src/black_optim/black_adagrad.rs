use super::{BlackOptimizer, BlackOptimizerState};
use black_core::black_error::BlackResult;
use black_core::black_grad::black_var::BlackVar;
use black_core::black_tensor::BlackTensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub struct BlackAdagrad {
    pub black_lr: f64,
    pub black_eps: f64,
    pub black_weight_decay: f64,
    pub black_lr_decay: f64,
    black_step_count: u64,
    black_sum_sq: Vec<BlackTensor>,
    black_initialized: bool,
}

impl BlackAdagrad {
    pub fn black_new(
        black_lr: f64,
        black_eps: f64,
        black_weight_decay: f64,
        black_lr_decay: f64,
    ) -> Self {
        BlackAdagrad {
            black_lr,
            black_eps,
            black_weight_decay,
            black_lr_decay,
            black_step_count: 0,
            black_sum_sq: Vec::new(),
            black_initialized: false,
        }
    }
}

impl BlackOptimizer for BlackAdagrad {
    fn black_step(&mut self, black_params: &mut [Arc<RwLock<BlackVar>>]) -> BlackResult<()> {
        if !self.black_initialized {
            for black_p in black_params.iter() {
                let black_p_r = black_p.read();
                let black_shape = black_p_r.black_data.black_shape.black_dims();
                self.black_sum_sq.push(BlackTensor::black_zeros(
                    black_shape,
                    black_core::black_dtype::BlackDType::BlackF32,
                )?);
            }
            self.black_initialized = true;
        }

        self.black_step_count += 1;
        let black_clr = self.black_lr / (1.0 + (self.black_step_count - 1) as f64 * self.black_lr_decay);
        let black_lr_f = black_clr as f32;
        let black_e = self.black_eps as f32;
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
            let black_sq_buf = std::sync::Arc::make_mut(&mut self.black_sum_sq[black_idx].black_buffer);
            let black_sq_data = black_sq_buf.black_as_f32_mut_slice();

            for black_i in 0..black_param_data.len() {
                if black_i >= black_grad_data.len() {
                    break;
                }
                let black_g = black_grad_data[black_i] + black_wd * black_param_data[black_i];
                black_sq_data[black_i] += black_g * black_g;
                black_param_data[black_i] -= black_lr_f * black_g / (black_sq_data[black_i].sqrt() + black_e);
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
