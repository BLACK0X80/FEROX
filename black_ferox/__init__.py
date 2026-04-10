from black_ferox import black_nn
from black_ferox import black_optim
from black_ferox import black_data
from black_ferox import black_train
from black_ferox import black_metrics
from black_ferox import black_export

try:
    from black_ferox._black_ferox_core import (
        BlackTensor,
        BlackVar,
        BlackAdamW as _BlackAdamWRust,
        BlackSGD as _BlackSGDRust,
        BlackLion as _BlackLionRust,
        BlackTrainLoop as _BlackTrainLoopRust,
    )
    BLACK_RUST_AVAILABLE = True
except ImportError:
    BLACK_RUST_AVAILABLE = False

__black_version__ = "0.1.0"
