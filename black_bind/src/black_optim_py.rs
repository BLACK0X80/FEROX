use black_train::black_optim::black_adamw::BlackAdamW;
use black_train::black_optim::black_sgd::BlackSGD;
use black_train::black_optim::black_lion::BlackLion;
use black_train::black_optim::BlackOptimizer;
use black_core::black_grad::black_var::BlackVar;
use crate::black_var_py::BlackVarPy;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Arc;
use parking_lot::RwLock;

#[pyclass(name = "BlackAdamW")]
pub struct BlackAdamWPy {
    black_inner: BlackAdamW,
    black_params: Vec<Arc<RwLock<BlackVar>>>,
}

#[pymethods]
impl BlackAdamWPy {
    #[new]
    #[pyo3(signature = (black_params, black_lr=1e-3, black_beta1=0.9, black_beta2=0.999, black_eps=1e-8, black_weight_decay=1e-2, black_amsgrad=false))]
    fn black_py_new(
        black_params: Vec<BlackVarPy>,
        black_lr: f64,
        black_beta1: f64,
        black_beta2: f64,
        black_eps: f64,
        black_weight_decay: f64,
        black_amsgrad: bool,
    ) -> Self {
        let black_p: Vec<Arc<RwLock<BlackVar>>> = black_params
            .iter()
            .map(|black_v| Arc::clone(&black_v.black_inner))
            .collect();

        BlackAdamWPy {
            black_inner: BlackAdamW::black_new(
                black_lr,
                black_beta1,
                black_beta2,
                black_eps,
                black_weight_decay,
                black_amsgrad,
            ),
            black_params: black_p,
        }
    }

    fn black_step(&mut self) -> PyResult<()> {
        self.black_inner
            .black_step(&mut self.black_params)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))
    }

    fn black_zero_grad(&mut self) {
        self.black_inner.black_zero_grad(&mut self.black_params);
    }
}

#[pyclass(name = "BlackSGD")]
pub struct BlackSGDPy {
    black_inner: BlackSGD,
    black_params: Vec<Arc<RwLock<BlackVar>>>,
}

#[pymethods]
impl BlackSGDPy {
    #[new]
    #[pyo3(signature = (black_params, black_lr=0.01, black_momentum=0.0, black_dampening=0.0, black_weight_decay=0.0, black_nesterov=false))]
    fn black_py_new(
        black_params: Vec<BlackVarPy>,
        black_lr: f64,
        black_momentum: f64,
        black_dampening: f64,
        black_weight_decay: f64,
        black_nesterov: bool,
    ) -> Self {
        let black_p: Vec<Arc<RwLock<BlackVar>>> = black_params
            .iter()
            .map(|black_v| Arc::clone(&black_v.black_inner))
            .collect();

        BlackSGDPy {
            black_inner: BlackSGD::black_new(
                black_lr,
                black_momentum,
                black_dampening,
                black_weight_decay,
                black_nesterov,
            ),
            black_params: black_p,
        }
    }

    fn black_step(&mut self) -> PyResult<()> {
        self.black_inner
            .black_step(&mut self.black_params)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))
    }

    fn black_zero_grad(&mut self) {
        self.black_inner.black_zero_grad(&mut self.black_params);
    }
}

#[pyclass(name = "BlackLion")]
pub struct BlackLionPy {
    black_inner: BlackLion,
    black_params: Vec<Arc<RwLock<BlackVar>>>,
}

#[pymethods]
impl BlackLionPy {
    #[new]
    #[pyo3(signature = (black_params, black_lr=1e-4, black_beta1=0.9, black_beta2=0.99, black_weight_decay=0.0))]
    fn black_py_new(
        black_params: Vec<BlackVarPy>,
        black_lr: f64,
        black_beta1: f64,
        black_beta2: f64,
        black_weight_decay: f64,
    ) -> Self {
        let black_p: Vec<Arc<RwLock<BlackVar>>> = black_params
            .iter()
            .map(|black_v| Arc::clone(&black_v.black_inner))
            .collect();

        BlackLionPy {
            black_inner: BlackLion::black_new(
                black_lr,
                black_beta1,
                black_beta2,
                black_weight_decay,
            ),
            black_params: black_p,
        }
    }

    fn black_step(&mut self) -> PyResult<()> {
        self.black_inner
            .black_step(&mut self.black_params)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))
    }

    fn black_zero_grad(&mut self) {
        self.black_inner.black_zero_grad(&mut self.black_params);
    }
}
