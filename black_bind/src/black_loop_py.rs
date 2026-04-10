use pyo3::prelude::*;

#[pyclass(name = "BlackTrainLoop")]
pub struct BlackTrainLoopPy {
    black_steps: usize,
    black_grad_accum_steps: usize,
    black_max_grad_norm: f64,
}

#[pymethods]
impl BlackTrainLoopPy {
    #[new]
    #[pyo3(signature = (black_steps, black_grad_accum_steps=1, black_max_grad_norm=1.0))]
    fn black_py_new(
        black_steps: usize,
        black_grad_accum_steps: usize,
        black_max_grad_norm: f64,
    ) -> Self {
        BlackTrainLoopPy {
            black_steps,
            black_grad_accum_steps,
            black_max_grad_norm,
        }
    }

    fn black_get_steps(&self) -> usize {
        self.black_steps
    }

    fn black_get_grad_accum_steps(&self) -> usize {
        self.black_grad_accum_steps
    }

    fn black_get_max_grad_norm(&self) -> f64 {
        self.black_max_grad_norm
    }
}
