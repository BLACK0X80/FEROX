use black_core::black_grad::black_var::{self, BlackVar};
use black_core::black_grad::black_grad_fn::*;
use black_core::black_ops::{black_elementwise, black_matmul, black_reduce};

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

    fn __add__(&self, black_other: &BlackVarPy) -> PyResult<Self> {
        let black_a; let black_b;
        {
            let a_r = self.black_inner.read(); black_a = a_r.black_data.clone();
        }
        {
            let b_r = black_other.black_inner.read(); black_b = b_r.black_data.clone();
        }
        let black_res = black_elementwise::black_add(&black_a, &black_b)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let black_grad_fn = Arc::new(BlackAddGradFn {
            black_input_a: Arc::clone(&self.black_inner),
            black_input_b: Arc::clone(&black_other.black_inner),
        });
        let black_var = BlackVar::black_with_grad_fn(black_res, black_grad_fn);
        Ok(BlackVarPy { black_inner: black_var })
    }

    fn __sub__(&self, black_other: &BlackVarPy) -> PyResult<Self> {
        let black_a; let black_b;
        {
            let a_r = self.black_inner.read(); black_a = a_r.black_data.clone();
        }
        {
            let b_r = black_other.black_inner.read(); black_b = b_r.black_data.clone();
        }
        let black_res = black_elementwise::black_sub(&black_a, &black_b)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let black_grad_fn = Arc::new(BlackSubGradFn {
            black_input_a: Arc::clone(&self.black_inner),
            black_input_b: Arc::clone(&black_other.black_inner),
        });
        let black_var = BlackVar::black_with_grad_fn(black_res, black_grad_fn);
        Ok(BlackVarPy { black_inner: black_var })
    }

    fn __mul__(&self, black_other: &BlackVarPy) -> PyResult<Self> {
        let black_a; let black_b;
        {
            let a_r = self.black_inner.read(); black_a = a_r.black_data.clone();
        }
        {
            let b_r = black_other.black_inner.read(); black_b = b_r.black_data.clone();
        }
        let black_res = black_elementwise::black_mul(&black_a, &black_b)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let black_grad_fn = Arc::new(BlackMulGradFn {
            black_input_a: Arc::clone(&self.black_inner),
            black_input_b: Arc::clone(&black_other.black_inner),
            black_saved_a: black_a,
            black_saved_b: black_b,
        });
        let black_var = BlackVar::black_with_grad_fn(black_res, black_grad_fn);
        Ok(BlackVarPy { black_inner: black_var })
    }

    fn __truediv__(&self, black_other: &BlackVarPy) -> PyResult<Self> {
        let black_a; let black_b;
        {
            let a_r = self.black_inner.read(); black_a = a_r.black_data.clone();
        }
        {
            let b_r = black_other.black_inner.read(); black_b = b_r.black_data.clone();
        }
        let black_res = black_elementwise::black_div(&black_a, &black_b)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let black_grad_fn = Arc::new(BlackDivGradFn {
            black_input_a: Arc::clone(&self.black_inner),
            black_input_b: Arc::clone(&black_other.black_inner),
            black_saved_a: black_a,
            black_saved_b: black_b,
        });
        let black_var = BlackVar::black_with_grad_fn(black_res, black_grad_fn);
        Ok(BlackVarPy { black_inner: black_var })
    }

    fn __matmul__(&self, black_other: &BlackVarPy) -> PyResult<Self> {
        let black_a; let black_b;
        {
            let a_r = self.black_inner.read(); black_a = a_r.black_data.clone();
        }
        {
            let b_r = black_other.black_inner.read(); black_b = b_r.black_data.clone();
        }
        let black_res = black_matmul::black_matmul(&black_a, &black_b)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let black_grad_fn = Arc::new(BlackMatmulGradFn {
            black_input_a: Arc::clone(&self.black_inner),
            black_input_b: Arc::clone(&black_other.black_inner),
            black_saved_a: black_a,
            black_saved_b: black_b,
        });
        let black_var = BlackVar::black_with_grad_fn(black_res, black_grad_fn);
        Ok(BlackVarPy { black_inner: black_var })
    }

    fn black_gelu(&self) -> PyResult<Self> {
        let black_a;
        {
            let a_r = self.black_inner.read(); black_a = a_r.black_data.clone();
        }
        
        let black_contig = black_a.black_contiguous().map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let black_data = black_contig.black_buffer.black_as_f32_slice();
        let mut black_out = vec![0.0f32; black_contig.black_numel()];
        let black_sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        for black_i in 0..black_contig.black_numel() {
            let black_x = black_data[black_i];
            let black_inner = black_sqrt_2_pi * (black_x + 0.044715 * black_x.powi(3));
            let black_tanh_val = black_inner.tanh();
            black_out[black_i] = 0.5 * black_x * (1.0 + black_tanh_val);
        }
        let black_buf = black_core::black_buffer::BlackBuffer::black_from_vec_f32(black_out)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let black_res = black_core::black_tensor::BlackTensor::black_new(
            black_buf,
            black_contig.black_shape.clone(),
            black_core::black_dtype::BlackDType::BlackF32,
            black_core::black_device::BlackDevice::BlackCpu,
        );

        let black_grad_fn = Arc::new(BlackGeluGradFn {
            black_input: Arc::clone(&self.black_inner),
            black_saved_input: black_contig,
        });
        let black_var = BlackVar::black_with_grad_fn(black_res, black_grad_fn);
        Ok(BlackVarPy { black_inner: black_var })
    }

    fn black_sum(&self) -> PyResult<Self> {
        let black_a;
        {
            let a_r = self.black_inner.read(); black_a = a_r.black_data.clone();
        }
        let black_res = black_reduce::black_sum(&black_a, None, false)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let black_grad_fn = Arc::new(BlackSumGradFn {
            black_input: Arc::clone(&self.black_inner),
            black_saved_shape: black_a.black_shape.clone(),
        });
        let black_var = BlackVar::black_with_grad_fn(black_res, black_grad_fn);
        Ok(BlackVarPy { black_inner: black_var })
    }

    fn black_t(&self) -> PyResult<Self> {
        let black_a;
        {
            let a_r = self.black_inner.read(); black_a = a_r.black_data.clone();
        }
        let black_res = black_a.black_t().map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let dims = black_a.black_shape.black_dims();
        let black_grad_fn = Arc::new(BlackTransposeGradFn {
            black_input: Arc::clone(&self.black_inner),
            black_dim0: dims.len() - 2,
            black_dim1: dims.len() - 1,
        });
        let black_var = BlackVar::black_with_grad_fn(black_res, black_grad_fn);
        Ok(BlackVarPy { black_inner: black_var })
    }

    fn black_view(&self, black_shape: Vec<usize>) -> PyResult<Self> {
        let black_a;
        {
            let a_r = self.black_inner.read(); black_a = a_r.black_data.clone();
        }
        let black_res = black_a.black_view(&black_shape).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let black_grad_fn = Arc::new(BlackViewGradFn {
            black_input: Arc::clone(&self.black_inner),
            black_saved_shape: black_a.black_shape.clone(),
        });
        let black_var = BlackVar::black_with_grad_fn(black_res, black_grad_fn);
        Ok(BlackVarPy { black_inner: black_var })
    }

    #[pyo3(signature = (black_d0, black_d1=None, black_d2=None, black_d3=None))]
    fn black_reshape(&self, black_d0: i64, black_d1: Option<i64>, black_d2: Option<i64>, black_d3: Option<i64>) -> PyResult<Self> {
        let mut black_dims: Vec<i64> = vec![black_d0];
        if let Some(black_v) = black_d1 { black_dims.push(black_v); }
        if let Some(black_v) = black_d2 { black_dims.push(black_v); }
        if let Some(black_v) = black_d3 { black_dims.push(black_v); }
        let black_a;
        {
            let a_r = self.black_inner.read(); black_a = a_r.black_data.clone();
        }
        let black_numel = black_a.black_numel();
        let mut black_inferred_idx: Option<usize> = None;
        let mut black_known_product: usize = 1;
        for (black_i, &black_d) in black_dims.iter().enumerate() {
            if black_d == -1 {
                if black_inferred_idx.is_some() {
                    return Err(PyValueError::new_err("only one dimension can be -1"));
                }
                black_inferred_idx = Some(black_i);
            } else if black_d < 0 {
                return Err(PyValueError::new_err(format!("invalid dimension: {}", black_d)));
            } else {
                black_known_product *= black_d as usize;
            }
        }
        let black_resolved: Vec<usize> = if let Some(black_idx) = black_inferred_idx {
            if black_known_product == 0 {
                return Err(PyValueError::new_err("cannot infer dimension with zero-size dims"));
            }
            let black_inferred = black_numel / black_known_product;
            black_dims.iter().enumerate().map(|(black_i, &black_d)| {
                if black_i == black_idx { black_inferred } else { black_d as usize }
            }).collect()
        } else {
            black_dims.iter().map(|&black_d| black_d as usize).collect()
        };
        let black_res = black_a.black_reshape(&black_resolved)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let black_grad_fn = Arc::new(BlackViewGradFn {
            black_input: Arc::clone(&self.black_inner),
            black_saved_shape: black_a.black_shape.clone(),
        });
        let black_var = BlackVar::black_with_grad_fn(black_res, black_grad_fn);
        Ok(BlackVarPy { black_inner: black_var })
    }
}
