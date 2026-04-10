use super::{BlackOptimizer, BlackOptimizerState};
use black_core::black_error::BlackResult;
use black_core::black_grad::black_var::BlackVar;
use black_core::black_tensor::BlackTensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub struct BlackSGD {
    pub black_lr: f64,
    pub black_momentum: f64,
    pub black_dampening: f64,
    pub black_weight_decay: f64,
    pub black_nesterov: bool,
    black_buf: Vec<Option<BlackTensor>>,
    black_initialized: bool,
}

impl BlackSGD {
    pub fn black_new(
        black_lr: f64,
        black_momentum: f64,
        black_dampening: f64,
        black_weight_decay: f64,
        black_nesterov: bool,
    ) -> Self {
        BlackSGD {
            black_lr,
            black_momentum,
            black_dampening,
            black_weight_decay,
            black_nesterov,
            black_buf: Vec::new(),
            black_initialized: false,
        }
    }
}

impl BlackOptimizer for BlackSGD {
    fn black_step(&mut self, black_params: &mut [Arc<RwLock<BlackVar>>]) -> BlackResult<()> {
        if !self.black_initialized {
            self.black_buf = vec![None; black_params.len()];
            self.black_initialized = true;
        }

        let black_lr_f = self.black_lr as f32;
        let black_mom = self.black_momentum as f32;
        let black_damp = self.black_dampening as f32;
        let black_wd = self.black_weight_decay as f32;

        for (black_idx, black_param) in black_params.iter().enumerate() {
            let mut black_p_w = black_param.write();

            let black_grad = match &black_p_w.black_grad {
                Some(black_g) => black_g.clone(),
                None => continue,
            };

            let black_param_buf =
                std::sync::Arc::make_mut(&mut black_p_w.black_data.black_buffer);
            let black_param_data = black_param_buf.black_as_f32_mut_slice();
            let black_grad_data = black_grad.black_buffer.black_as_f32_slice();

            if black_mom != 0.0 {
                let black_buf_tensor = self.black_buf[black_idx].get_or_insert_with(|| {
                    black_grad.clone()
                });

                let black_buf_data = std::sync::Arc::make_mut(&mut black_buf_tensor.black_buffer)
                    .black_as_f32_mut_slice();

                for black_i in 0..black_param_data.len() {
                    if black_i >= black_grad_data.len() {
                        break;
                    }
                    let black_d = black_grad_data[black_i] + black_wd * black_param_data[black_i];
                    black_buf_data[black_i] =
                        black_mom * black_buf_data[black_i] + (1.0 - black_damp) * black_d;

                    let black_update = if self.black_nesterov {
                        black_d + black_mom * black_buf_data[black_i]
                    } else {
                        black_buf_data[black_i]
                    };

                    black_param_data[black_i] -= black_lr_f * black_update;
                }
            } else {
                for black_i in 0..black_param_data.len() {
                    if black_i >= black_grad_data.len() {
                        break;
                    }
                    let black_d = black_grad_data[black_i] + black_wd * black_param_data[black_i];
                    black_param_data[black_i] -= black_lr_f * black_d;
                }
            }
        }

        Ok(())
    }

    fn black_zero_grad(&mut self, black_params: &mut [Arc<RwLock<BlackVar>>]) {
        for black_p in black_params.iter() {
            let mut black_p_w = black_p.write();
            black_p_w.black_zero_grad();
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
