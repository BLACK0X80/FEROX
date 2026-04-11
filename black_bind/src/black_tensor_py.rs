use black_core::black_dtype::BlackDType;
use black_core::black_tensor::BlackTensor;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Arc;
use parking_lot::RwLock;

#[pyclass(name = "BlackTensor")]
#[derive(Clone)]
pub struct BlackTensorPy {
    pub black_inner: Arc<RwLock<BlackTensor>>,
}

#[pymethods]
impl BlackTensorPy {
    #[new]
    fn black_py_new(black_data: Vec<f32>, black_shape: Vec<usize>) -> PyResult<Self> {
        let black_tensor =
            BlackTensor::black_from_slice_f32(&black_data, &black_shape)
                .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_tensor)),
        })
    }

    #[staticmethod]
    fn black_zeros(black_shape: Vec<usize>) -> PyResult<Self> {
        let black_tensor = BlackTensor::black_zeros(&black_shape, BlackDType::BlackF32)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_tensor)),
        })
    }

    #[staticmethod]
    fn black_ones(black_shape: Vec<usize>) -> PyResult<Self> {
        let black_tensor = BlackTensor::black_ones(&black_shape, BlackDType::BlackF32)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_tensor)),
        })
    }

    #[staticmethod]
    fn black_rand(black_shape: Vec<usize>) -> PyResult<Self> {
        let black_tensor = BlackTensor::black_rand(&black_shape)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_tensor)),
        })
    }

    #[staticmethod]
    fn black_randn(black_shape: Vec<usize>) -> PyResult<Self> {
        let black_tensor = BlackTensor::black_randn(&black_shape)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_tensor)),
        })
    }

    fn black_shape(&self) -> Vec<usize> {
        let black_r = self.black_inner.read();
        black_r.black_shape.black_dims().to_vec()
    }

    fn black_ndim(&self) -> usize {
        let black_r = self.black_inner.read();
        black_r.black_ndim()
    }

    fn black_numel(&self) -> usize {
        let black_r = self.black_inner.read();
        black_r.black_numel()
    }

    fn black_dtype(&self) -> String {
        let black_r = self.black_inner.read();
        format!("{}", black_r.black_dtype)
    }

    fn black_device(&self) -> String {
        let black_r = self.black_inner.read();
        format!("{}", black_r.black_device)
    }

    fn black_to_list(&self) -> PyResult<Vec<f32>> {
        let black_r = self.black_inner.read();
        black_r
            .black_to_vec_f32()
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))
    }

    fn black_item(&self) -> PyResult<f32> {
        let black_r = self.black_inner.read();
        black_r
            .black_item_f32()
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))
    }

    fn black_reshape(&self, black_shape: Vec<usize>) -> PyResult<Self> {
        let black_r = self.black_inner.read();
        let black_new = black_r
            .black_reshape(&black_shape)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_new)),
        })
    }

    fn black_view(&self, black_shape: Vec<usize>) -> PyResult<Self> {
        let black_r = self.black_inner.read();
        let black_new = black_r
            .black_view(&black_shape)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_new)),
        })
    }

    fn black_transpose(&self, black_dim0: usize, black_dim1: usize) -> PyResult<Self> {
        let black_r = self.black_inner.read();
        let black_new = black_r
            .black_transpose(black_dim0, black_dim1)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_new)),
        })
    }

    fn black_t(&self) -> PyResult<Self> {
        let black_r = self.black_inner.read();
        let black_new = black_r
            .black_t()
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_new)),
        })
    }

    fn black_contiguous(&self) -> PyResult<Self> {
        let black_r = self.black_inner.read();
        let black_new = black_r
            .black_contiguous()
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_new)),
        })
    }

    fn __repr__(&self) -> String {
        let black_r = self.black_inner.read();
        format!("{}", *black_r)
    }

    fn __add__(&self, black_other: &BlackTensorPy) -> PyResult<Self> {
        let black_a = self.black_inner.read();
        let black_b = black_other.black_inner.read();
        let black_result = black_core::black_ops::black_elementwise::black_add(&black_a, &black_b)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_result)),
        })
    }

    fn __sub__(&self, black_other: &BlackTensorPy) -> PyResult<Self> {
        let black_a = self.black_inner.read();
        let black_b = black_other.black_inner.read();
        let black_result = black_core::black_ops::black_elementwise::black_sub(&black_a, &black_b)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_result)),
        })
    }

    fn __mul__(&self, black_other: &BlackTensorPy) -> PyResult<Self> {
        let black_a = self.black_inner.read();
        let black_b = black_other.black_inner.read();
        let black_result = black_core::black_ops::black_elementwise::black_mul(&black_a, &black_b)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_result)),
        })
    }

    fn __truediv__(&self, black_other: &BlackTensorPy) -> PyResult<Self> {
        let black_a = self.black_inner.read();
        let black_b = black_other.black_inner.read();
        let black_result = black_core::black_ops::black_elementwise::black_div(&black_a, &black_b)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_result)),
        })
    }

    fn __matmul__(&self, black_other: &BlackTensorPy) -> PyResult<Self> {
        let black_a = self.black_inner.read();
        let black_b = black_other.black_inner.read();
        let black_result = black_core::black_ops::black_matmul::black_matmul(&black_a, &black_b)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_result)),
        })
    }

    fn __neg__(&self) -> PyResult<Self> {
        let black_a = self.black_inner.read();
        let black_result = black_core::black_ops::black_elementwise::black_neg(&black_a)
            .map_err(|black_e| PyValueError::new_err(format!("{}", black_e)))?;
        Ok(BlackTensorPy {
            black_inner: Arc::new(RwLock::new(black_result)),
        })
    }
}
