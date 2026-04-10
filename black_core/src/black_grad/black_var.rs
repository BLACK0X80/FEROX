use crate::black_error::{BlackError, BlackResult};
use crate::black_ops::black_elementwise::black_add;
use crate::black_tensor::BlackTensor;
use std::collections::HashSet;
use std::sync::Arc;
use parking_lot::RwLock;

pub trait BlackGradFn: Send + Sync {
    fn black_backward(&self, black_upstream_grad: &BlackTensor) -> Vec<Option<BlackTensor>>;
    fn black_inputs(&self) -> Vec<Arc<RwLock<BlackVar>>>;
}

pub struct BlackVar {
    pub black_data: BlackTensor,
    pub black_grad: Option<BlackTensor>,
    pub black_grad_fn: Option<Arc<dyn BlackGradFn>>,
    pub black_requires_grad: bool,
    pub black_is_leaf: bool,
    pub black_version: u64,
}

impl BlackVar {
    pub fn black_new(black_data: BlackTensor, black_requires_grad: bool) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(BlackVar {
            black_data,
            black_grad: None,
            black_grad_fn: None,
            black_requires_grad,
            black_is_leaf: true,
            black_version: 0,
        }))
    }

    pub fn black_with_grad_fn(
        black_data: BlackTensor,
        black_grad_fn: Arc<dyn BlackGradFn>,
    ) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(BlackVar {
            black_data,
            black_grad: None,
            black_grad_fn: Some(black_grad_fn),
            black_requires_grad: true,
            black_is_leaf: false,
            black_version: 0,
        }))
    }

    pub fn black_zero_grad(&mut self) {
        if let Some(ref mut black_g) = self.black_grad {
            let black_buf = std::sync::Arc::make_mut(&mut black_g.black_buffer);
            black_buf.black_zero();
        }
    }

    pub fn black_accumulate_grad(&mut self, black_incoming: &BlackTensor) -> BlackResult<()> {
        match &mut self.black_grad {
            Some(ref mut black_existing) => {
                let black_new_grad = black_add(black_existing, black_incoming)?;
                *black_existing = black_new_grad;
            }
            None => {
                self.black_grad = Some(black_incoming.clone());
            }
        }
        Ok(())
    }
}

pub fn black_backward(black_root: &Arc<RwLock<BlackVar>>) -> BlackResult<()> {
    let black_root_data_shape;
    {
        let black_root_r = black_root.read();
        black_root_data_shape = black_root_r.black_data.black_shape.clone();
    }

    let black_initial_grad =
        BlackTensor::black_ones(black_root_data_shape.black_dims(), crate::black_dtype::BlackDType::BlackF32)?;

    {
        let mut black_root_w = black_root.write();
        black_root_w.black_accumulate_grad(&black_initial_grad)?;
    }

    let mut black_topo_order: Vec<Arc<RwLock<BlackVar>>> = Vec::new();
    let mut black_visited: HashSet<usize> = HashSet::new();

    fn black_topo_sort(
        black_node: &Arc<RwLock<BlackVar>>,
        black_visited: &mut HashSet<usize>,
        black_order: &mut Vec<Arc<RwLock<BlackVar>>>,
    ) {
        let black_ptr = Arc::as_ptr(black_node) as usize;
        if black_visited.contains(&black_ptr) {
            return;
        }
        black_visited.insert(black_ptr);

        let black_grad_fn;
        {
            let black_node_r = black_node.read();
            black_grad_fn = black_node_r.black_grad_fn.clone();
        }

        if let Some(ref black_gf) = black_grad_fn {
            for black_input in black_gf.black_inputs() {
                black_topo_sort(&black_input, black_visited, black_order);
            }
        }

        black_order.push(Arc::clone(black_node));
    }

    black_topo_sort(black_root, &mut black_visited, &mut black_topo_order);
    black_topo_order.reverse();

    for black_node in &black_topo_order {
        let black_upstream;
        let black_grad_fn;
        {
            let black_node_r = black_node.read();
            black_upstream = black_node_r.black_grad.clone();
            black_grad_fn = black_node_r.black_grad_fn.clone();
        }

        if let (Some(ref black_up), Some(ref black_gf)) = (&black_upstream, &black_grad_fn) {
            let black_grads = black_gf.black_backward(black_up);
            let black_inputs = black_gf.black_inputs();

            for (black_idx, black_maybe_grad) in black_grads.into_iter().enumerate() {
                if let Some(black_g) = black_maybe_grad {
                    if black_idx < black_inputs.len() {
                        let black_input = &black_inputs[black_idx];
                        let black_req;
                        {
                            let black_inp_r = black_input.read();
                            black_req = black_inp_r.black_requires_grad;
                        }
                        if black_req {
                            let mut black_inp_w = black_input.write();
                            black_inp_w.black_accumulate_grad(&black_g)?;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
