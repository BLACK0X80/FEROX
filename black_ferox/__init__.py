from black_ferox import black_nn as black_nn
from black_ferox import black_optim as black_optim
from black_ferox import black_data as black_data
from black_ferox import black_train as black_train
from black_ferox import black_metrics as black_metrics
from black_ferox import black_export as black_export

try:
    from black_ferox._black_ferox_core import (  # noqa: F401
        BlackTensor as BlackTensor,
        BlackVar as BlackVar,
        BlackAdamW as _BlackAdamWRust,
        BlackSGD as _BlackSGDRust,
        BlackLion as _BlackLionRust,
        BlackTrainLoop as _BlackTrainLoopRust,
    )
    BLACK_RUST_AVAILABLE = True
except ImportError:
    BLACK_RUST_AVAILABLE = False

__black_version__ = "0.1.0"

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
    # Dummy gradient checkpointing for tests
    return module(*args, **kwargs)
