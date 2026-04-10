pub mod black_adamw;
pub mod black_sgd;
pub mod black_adam;
pub mod black_lion;
pub mod black_adagrad;

use black_core::black_error::BlackResult;
use black_core::black_grad::black_var::BlackVar;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackOptimizerState {
    pub black_step_count: u64,
    pub black_state: HashMap<String, Vec<f32>>,
}

pub trait BlackOptimizer: Send + Sync {
    fn black_step(&mut self, black_params: &mut [Arc<RwLock<BlackVar>>]) -> BlackResult<()>;
    fn black_zero_grad(&mut self, black_params: &mut [Arc<RwLock<BlackVar>>]);
    fn black_state_dict(&self) -> BlackOptimizerState;
    fn black_load_state_dict(&mut self, black_state: BlackOptimizerState);
}
