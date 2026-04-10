mod black_tensor_py;
mod black_var_py;
mod black_optim_py;
mod black_loop_py;

use pyo3::prelude::*;

#[pymodule]
fn _black_ferox_core(_black_py: Python<'_>, black_m: &Bound<'_, PyModule>) -> PyResult<()> {
    black_m.add_class::<black_tensor_py::BlackTensorPy>()?;
    black_m.add_class::<black_var_py::BlackVarPy>()?;
    black_m.add_class::<black_optim_py::BlackAdamWPy>()?;
    black_m.add_class::<black_optim_py::BlackSGDPy>()?;
    black_m.add_class::<black_optim_py::BlackLionPy>()?;
    black_m.add_class::<black_loop_py::BlackTrainLoopPy>()?;
    Ok(())
}
