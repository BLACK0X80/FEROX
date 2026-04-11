use black_core::black_error::BlackResult;
use black_core::black_grad::black_var::{self, BlackVar};
use black_core::black_tensor::BlackTensor;
use crate::black_optim::BlackOptimizer;
use parking_lot::RwLock;
use std::sync::Arc;

pub struct BlackTrainLoop {
    pub black_steps: usize,
    pub black_grad_accum_steps: usize,
    pub black_max_grad_norm: f64,
}

impl BlackTrainLoop {
    pub fn black_new(
        black_steps: usize,
        black_grad_accum_steps: usize,
        black_max_grad_norm: f64,
    ) -> Self {
        BlackTrainLoop {
            black_steps,
            black_grad_accum_steps,
            black_max_grad_norm,
        }
    }

    pub fn black_run<F, L>(
        &self,
        black_model_fn: &F,
        black_loss_fn: &L,
        black_optimizer: &mut dyn BlackOptimizer,
        black_params: &mut [Arc<RwLock<BlackVar>>],
        black_inputs: &[BlackTensor],
        black_targets: &[BlackTensor],
    ) -> BlackResult<Vec<f32>>
    where
        F: Fn(&BlackTensor) -> BlackResult<BlackTensor>,
        L: Fn(&BlackTensor, &BlackTensor) -> BlackResult<BlackTensor>,
    {
        let mut black_losses = Vec::new();
        let black_num_samples = black_inputs.len().min(black_targets.len());

        for black_step in 0..self.black_steps {
            let black_sample_idx = black_step % black_num_samples;
            let black_input = &black_inputs[black_sample_idx];
            let black_target = &black_targets[black_sample_idx];

            let black_output = black_model_fn(black_input)?;
            let black_loss = black_loss_fn(&black_output, black_target)?;

            let black_loss_val = black_loss.black_item_f32()?;
            black_losses.push(black_loss_val);

            let black_loss_var = BlackVar::black_new(black_loss.clone(), true);
            black_var::black_backward(&black_loss_var)?;

            if (black_step + 1) % self.black_grad_accum_steps == 0 {
                black_clip_grad_norm(black_params, self.black_max_grad_norm as f32)?;
                black_optimizer.black_step(black_params)?;
                black_optimizer.black_zero_grad(black_params);
            }
        }

        Ok(black_losses)
    }
}

pub fn black_clip_grad_norm(
    black_params: &[Arc<RwLock<BlackVar>>],
    black_max_norm: f32,
) -> BlackResult<f32> {
    let mut black_total_norm_sq = 0.0f32;

    for black_p in black_params {
        let black_p_r = black_p.read();
        if let Some(ref black_grad) = black_p_r.black_grad {
            let black_norm = black_grad.black_norm()?;
            black_total_norm_sq += black_norm * black_norm;
        }
    }

    let black_total_norm = black_total_norm_sq.sqrt();

    if black_total_norm > black_max_norm {
        let black_clip_coef = black_max_norm / (black_total_norm + 1e-6);
        for black_p in black_params {
            let mut black_p_w = black_p.write();
            if let Some(ref mut black_grad) = black_p_w.black_grad {
                black_grad.black_scale_inplace(black_clip_coef)?;
            }
        }
    }

    Ok(black_total_norm)
}

pub struct BlackGradScaler {
    pub black_scale: f32,
    pub black_growth_factor: f32,
    pub black_backoff_factor: f32,
    pub black_growth_interval: u64,
    black_step_count: u64,
    black_consecutive_clean: u64,
}

impl BlackGradScaler {
    pub fn black_new() -> Self {
        BlackGradScaler {
            black_scale: 65536.0,
            black_growth_factor: 2.0,
            black_backoff_factor: 0.5,
            black_growth_interval: 2000,
            black_step_count: 0,
            black_consecutive_clean: 0,
        }
    }

    pub fn black_scale_loss(&self, black_loss: &BlackTensor) -> BlackResult<BlackTensor> {
        let mut black_scaled = black_loss.clone();
        black_scaled.black_scale_inplace(self.black_scale)?;
        Ok(black_scaled)
    }

    pub fn black_check_and_step(
        &mut self,
        black_optimizer: &mut dyn BlackOptimizer,
        black_params: &mut [Arc<RwLock<BlackVar>>],
    ) -> BlackResult<bool> {
        let black_finite = black_check_finite(black_params)?;

        if black_finite {
            let black_inv_scale = 1.0 / self.black_scale;
            for black_p in black_params.iter() {
                let mut black_p_w = black_p.write();
                if let Some(ref mut black_g) = black_p_w.black_grad {
                    black_g.black_scale_inplace(black_inv_scale)?;
                }
            }
            black_optimizer.black_step(black_params)?;
            self.black_consecutive_clean += 1;

            if self.black_consecutive_clean >= self.black_growth_interval {
                self.black_scale *= self.black_growth_factor;
                self.black_consecutive_clean = 0;
            }
        } else {
            self.black_scale *= self.black_backoff_factor;
            self.black_consecutive_clean = 0;
        }

        self.black_step_count += 1;
        Ok(black_finite)
    }
}

fn black_check_finite(black_params: &[Arc<RwLock<BlackVar>>]) -> BlackResult<bool> {
    for black_p in black_params {
        let black_p_r = black_p.read();
        if let Some(ref black_g) = black_p_r.black_grad {
            if !black_g.black_is_finite()? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
