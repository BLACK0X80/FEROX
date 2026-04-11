__version__ = "0.1.2"

try:
    from black_ferox._black_ferox_core import (
        BlackTensorPy as black_tensor,
        BlackVarPy as black_var,
    )
except ImportError:
    pass

from black_ferox import black_nn
from black_ferox import black_optim
from black_ferox import black_data
from black_ferox import black_train
from black_ferox import black_metrics
from black_ferox import black_export

__author__ = "BLACK0X80"
def black_tensor(data, black_dtype=None, black_device=None, black_requires_grad=False):
    import numpy as np
    from ._black_ferox_core import BlackTensor, BlackVar
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    t = BlackTensor(data.flatten().tolist(), list(data.shape))
    if black_dtype is not None:
        t = t.black_cast(black_dtype)
    if black_device is not None:
        t = t.black_to(black_device)
    return BlackVar(t, black_requires_grad)

def black_checkpoint(module, *args, **kwargs):
    
    return module(*args, **kwargs)
