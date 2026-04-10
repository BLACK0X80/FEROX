use crate::black_error::BlackResult;
use crate::black_grad::black_var::BlackVar;
use crate::black_tensor::BlackTensor;
use std::sync::Arc;
use parking_lot::RwLock;

pub struct BlackCheckpoint<F>
where
    F: Fn(&[BlackTensor]) -> BlackResult<BlackTensor> + Send + Sync,
{
    black_fn: F,
    black_inputs: Vec<Arc<RwLock<BlackVar>>>,
}

impl<F> BlackCheckpoint<F>
where
    F: Fn(&[BlackTensor]) -> BlackResult<BlackTensor> + Send + Sync,
{
    pub fn black_new(black_fn: F, black_inputs: Vec<Arc<RwLock<BlackVar>>>) -> Self {
        BlackCheckpoint {
            black_fn,
            black_inputs,
        }
    }

    pub fn black_apply(&self) -> BlackResult<BlackTensor> {
        let black_input_tensors: Vec<BlackTensor> = self
            .black_inputs
            .iter()
            .map(|black_v| {
                let black_r = black_v.read();
                black_r.black_data.clone()
            })
            .collect();

        (self.black_fn)(&black_input_tensors)
    }
}

pub fn black_checkpoint<F>(
    black_fn: F,
    black_inputs: Vec<Arc<RwLock<BlackVar>>>,
) -> BlackResult<BlackTensor>
where
    F: Fn(&[BlackTensor]) -> BlackResult<BlackTensor> + Send + Sync,
{
    let black_cp = BlackCheckpoint::black_new(black_fn, black_inputs);
    black_cp.black_apply()
}
