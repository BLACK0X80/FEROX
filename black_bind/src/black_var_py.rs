use black_core::black_grad::black_var::{self, BlackVar};

use crate::black_tensor_py::BlackTensorPy;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Arc;
use parking_lot::RwLock;

#[pyclass(name = "BlackVar")]
#[derive(Clone)]
pub struct BlackVarPy {
    pub black_inner: Arc<RwLock<BlackVar>>,
}

#[pymethods]
impl BlackVarPy {
    #[new]
    fn black_py_new(black_tensor: &BlackTensorPy, black_requires_grad: bool) -> PyResult<Self> {
        let black_data = black_tensor.black_inner.read().clone();
        let black_var = BlackVar::black_new(black_data, black_requires_grad);
        Ok(BlackVarPy {
            black_inner: black_var,
        })
    }

    #[getter]
    fn black_data(&self) -> PyResult<BlackTensorPy> {
        let black_r = self.black_inner.read();
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_r.black_data.clone())),
        })
    }

    #[getter]
    fn black_grad(&self) -> PyResult<Option<BlackTensorPy>> {
        let black_r = self.black_inner.read();
        match &black_r.black_grad {
            Some(black_g) => Ok(Some(BlackTensorPy {
                black_inner: Arc::new(RwLock::new(black_g.clone())),
            })),
            None => Ok(None),
        }
    }

    #[getter]
    fn black_requires_grad(&self) -> bool {
        let black_r = self.black_inner.read();
        black_r.black_requires_grad
    }

    fn black_backward(&self) -> PyResult<()> {
        black_var::black_backward(&self.black_inner)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))
    }

    fn black_zero_grad(&self) -> PyResult<()> {
        let mut black_w = self.black_inner.write();
        black_w.black_zero_grad();
        Ok(())
    }

    fn __repr__(&self) -> String {
        let black_r = self.black_inner.read();
        format!(
            "BlackVar(data={}, requires_grad={})",
            black_r.black_data, black_r.black_requires_grad
        )
    }
}
