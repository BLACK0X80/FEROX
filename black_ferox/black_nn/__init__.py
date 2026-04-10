import math
import random


class BlackModule:
    def __init__(self):
        self.black_training = True
        self._black_parameters = {}
        self._black_buffers = {}
        self._black_submodules = {}

    def black_forward(self, *black_args, **black_kwargs):
        raise NotImplementedError

    def __call__(self, *black_args, **black_kwargs):
        return self.black_forward(*black_args, **black_kwargs)

    def black_parameters(self, black_recurse=True):
        black_params = list(self._black_parameters.values())
        if black_recurse:
            for black_mod in self._black_submodules.values():
                black_params.extend(black_mod.black_parameters(black_recurse=True))
        return black_params

    def black_named_parameters(self, black_prefix='', black_recurse=True):
        black_result = {}
        for black_name, black_param in self._black_parameters.items():
            black_full_name = f"{black_prefix}.{black_name}" if black_prefix else black_name
            black_result[black_full_name] = black_param
        if black_recurse:
            for black_mod_name, black_mod in self._black_submodules.items():
                black_sub_prefix = f"{black_prefix}.{black_mod_name}" if black_prefix else black_mod_name
                black_result.update(black_mod.black_named_parameters(black_sub_prefix, black_recurse))
        return black_result

    def black_train(self, black_mode=True):
        self.black_training = black_mode
        for black_mod in self._black_submodules.values():
            black_mod.black_train(black_mode)
        return self

    def black_eval(self):
        return self.black_train(False)

    def black_state_dict(self):
        black_state = {}
        for black_name, black_param in self._black_parameters.items():
            black_state[black_name] = black_param
        for black_mod_name, black_mod in self._black_submodules.items():
            black_sub_state = black_mod.black_state_dict()
            for black_key, black_val in black_sub_state.items():
                black_state[f"{black_mod_name}.{black_key}"] = black_val
        return black_state

    def black_load_state_dict(self, black_state, black_strict=True):
        for black_name in self._black_parameters:
            if black_name in black_state:
                self._black_parameters[black_name] = black_state[black_name]
            elif black_strict:
                raise KeyError(f"Missing key: {black_name}")
        for black_mod_name, black_mod in self._black_submodules.items():
            black_sub_state = {}
            black_prefix = f"{black_mod_name}."
            for black_key, black_val in black_state.items():
                if black_key.startswith(black_prefix):
                    black_sub_state[black_key[len(black_prefix):]] = black_val
            black_mod.black_load_state_dict(black_sub_state, black_strict)

    def black_to(self, black_device):
        return self

    def black_register_parameter(self, black_name, black_param):
        self._black_parameters[black_name] = black_param

    def black_register_buffer(self, black_name, black_buf):
        self._black_buffers[black_name] = black_buf

    def black_register_submodule(self, black_name, black_module):
        self._black_submodules[black_name] = black_module

    def __setattr__(self, black_name, black_value):
        if isinstance(black_value, BlackModule) and black_name not in ('_black_parameters', '_black_buffers', '_black_submodules'):
            if hasattr(self, '_black_submodules'):
                self._black_submodules[black_name] = black_value
        super().__setattr__(black_name, black_value)

    def black_num_parameters(self):
        return sum(1 for _ in self.black_parameters())

    def __repr__(self):
        black_lines = [f"{self.__class__.__name__}("]
        for black_name, black_mod in self._black_submodules.items():
            black_lines.append(f"  ({black_name}): {repr(black_mod)}")
        black_lines.append(")")
        return "\n".join(black_lines)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, black_state):
        self.__dict__.update(black_state)


class BlackLinear(BlackModule):
    def __init__(self, black_in_features, black_out_features, black_bias=True):
        super().__init__()
        self.black_in_features = black_in_features
        self.black_out_features = black_out_features
        self.black_has_bias = black_bias
        black_k = 1.0 / math.sqrt(black_in_features)
        self._black_parameters['black_weight'] = [
            [random.uniform(-black_k, black_k) for _ in range(black_in_features)]
            for _ in range(black_out_features)
        ]
        if black_bias:
            self._black_parameters['black_bias'] = [random.uniform(-black_k, black_k) for _ in range(black_out_features)]

    def black_forward(self, black_x):
        return {"black_op": "linear", "black_input": black_x, "black_weight": self._black_parameters['black_weight'],
                "black_bias": self._black_parameters.get('black_bias')}


class BlackEmbedding(BlackModule):
    def __init__(self, black_num_embeddings, black_embedding_dim, black_padding_idx=None):
        super().__init__()
        self.black_num_embeddings = black_num_embeddings
        self.black_embedding_dim = black_embedding_dim
        self.black_padding_idx = black_padding_idx
        self._black_parameters['black_weight'] = [
            [random.gauss(0, 1) for _ in range(black_embedding_dim)]
            for _ in range(black_num_embeddings)
        ]

    def black_forward(self, black_indices):
        return {"black_op": "embedding", "black_indices": black_indices, "black_weight": self._black_parameters['black_weight']}


class BlackLayerNorm(BlackModule):
    def __init__(self, black_normalized_shape, black_eps=1e-5, black_elementwise_affine=True):
        super().__init__()
        self.black_normalized_shape = black_normalized_shape if isinstance(black_normalized_shape, (list, tuple)) else [black_normalized_shape]
        self.black_eps = black_eps
        self.black_elementwise_affine = black_elementwise_affine
        if black_elementwise_affine:
            black_size = 1
            for black_d in self.black_normalized_shape:
                black_size *= black_d
            self._black_parameters['black_weight'] = [1.0] * black_size
            self._black_parameters['black_bias'] = [0.0] * black_size

    def black_forward(self, black_x):
        return {"black_op": "layer_norm", "black_input": black_x, "black_eps": self.black_eps}


class BlackRMSNorm(BlackModule):
    def __init__(self, black_normalized_shape, black_eps=1e-8):
        super().__init__()
        self.black_normalized_shape = black_normalized_shape
        self.black_eps = black_eps
        self._black_parameters['black_weight'] = [1.0] * black_normalized_shape

    def black_forward(self, black_x):
        return {"black_op": "rms_norm", "black_input": black_x, "black_eps": self.black_eps}


class BlackGroupNorm(BlackModule):
    def __init__(self, black_num_groups, black_num_channels, black_eps=1e-5):
        super().__init__()
        self.black_num_groups = black_num_groups
        self.black_num_channels = black_num_channels
        self.black_eps = black_eps
        self._black_parameters['black_weight'] = [1.0] * black_num_channels
        self._black_parameters['black_bias'] = [0.0] * black_num_channels

    def black_forward(self, black_x):
        return {"black_op": "group_norm", "black_input": black_x}


class BlackBatchNorm1d(BlackModule):
    def __init__(self, black_num_features, black_eps=1e-5, black_momentum=0.1):
        super().__init__()
        self.black_num_features = black_num_features
        self.black_eps = black_eps
        self.black_momentum = black_momentum
        self._black_parameters['black_weight'] = [1.0] * black_num_features
        self._black_parameters['black_bias'] = [0.0] * black_num_features
        self._black_buffers['black_running_mean'] = [0.0] * black_num_features
        self._black_buffers['black_running_var'] = [1.0] * black_num_features

    def black_forward(self, black_x):
        return {"black_op": "batch_norm_1d", "black_input": black_x, "black_training": self.black_training}


class BlackBatchNorm2d(BlackModule):
    def __init__(self, black_num_features, black_eps=1e-5, black_momentum=0.1):
        super().__init__()
        self.black_num_features = black_num_features
        self.black_eps = black_eps
        self.black_momentum = black_momentum
        self._black_parameters['black_weight'] = [1.0] * black_num_features
        self._black_parameters['black_bias'] = [0.0] * black_num_features
        self._black_buffers['black_running_mean'] = [0.0] * black_num_features
        self._black_buffers['black_running_var'] = [1.0] * black_num_features

    def black_forward(self, black_x):
        return {"black_op": "batch_norm_2d", "black_input": black_x, "black_training": self.black_training}


class BlackDropout(BlackModule):
    def __init__(self, black_p=0.5):
        super().__init__()
        self.black_p = black_p

    def black_forward(self, black_x):
        if self.black_training and self.black_p > 0:
            return {"black_op": "dropout", "black_input": black_x, "black_p": self.black_p}
        return black_x


class BlackMultiheadAttention(BlackModule):
    def __init__(self, black_embed_dim, black_num_heads, black_dropout=0.0, black_bias=True,
                 black_add_bias_kv=False, black_kdim=None, black_vdim=None, black_batch_first=False):
        super().__init__()
        self.black_embed_dim = black_embed_dim
        self.black_num_heads = black_num_heads
        self.black_head_dim = black_embed_dim // black_num_heads
        self.black_dropout = black_dropout
        self.black_batch_first = black_batch_first
        black_kdim = black_kdim or black_embed_dim
        black_vdim = black_vdim or black_embed_dim
        self.black_q_proj = BlackLinear(black_embed_dim, black_embed_dim, black_bias)
        self.black_k_proj = BlackLinear(black_kdim, black_embed_dim, black_bias)
        self.black_v_proj = BlackLinear(black_vdim, black_embed_dim, black_bias)
        self.black_out_proj = BlackLinear(black_embed_dim, black_embed_dim, black_bias)

    def black_forward(self, black_query, black_key=None, black_value=None, black_mask=None):
        return {"black_op": "multihead_attention", "black_query": black_query,
                "black_key": black_key, "black_value": black_value, "black_mask": black_mask}


class BlackGroupedQueryAttention(BlackModule):
    def __init__(self, black_embed_dim, black_num_heads, black_num_kv_heads, black_head_dim):
        super().__init__()
        self.black_embed_dim = black_embed_dim
        self.black_num_heads = black_num_heads
        self.black_num_kv_heads = black_num_kv_heads
        self.black_head_dim = black_head_dim
        self.black_q_proj = BlackLinear(black_embed_dim, black_num_heads * black_head_dim)
        self.black_k_proj = BlackLinear(black_embed_dim, black_num_kv_heads * black_head_dim)
        self.black_v_proj = BlackLinear(black_embed_dim, black_num_kv_heads * black_head_dim)
        self.black_o_proj = BlackLinear(black_num_heads * black_head_dim, black_embed_dim)

    def black_forward(self, black_x, black_mask=None):
        return {"black_op": "gqa", "black_input": black_x, "black_mask": black_mask}


class BlackSlidingWindowAttention(BlackModule):
    def __init__(self, black_embed_dim, black_num_heads, black_window_size):
        super().__init__()
        self.black_embed_dim = black_embed_dim
        self.black_num_heads = black_num_heads
        self.black_window_size = black_window_size
        self.black_q_proj = BlackLinear(black_embed_dim, black_embed_dim)
        self.black_k_proj = BlackLinear(black_embed_dim, black_embed_dim)
        self.black_v_proj = BlackLinear(black_embed_dim, black_embed_dim)
        self.black_out_proj = BlackLinear(black_embed_dim, black_embed_dim)

    def black_forward(self, black_x, black_mask=None):
        return {"black_op": "sliding_window_attn", "black_input": black_x, "black_window_size": self.black_window_size}


class BlackConv1d(BlackModule):
    def __init__(self, black_in_channels, black_out_channels, black_kernel_size,
                 black_stride=1, black_padding=0, black_dilation=1, black_groups=1, black_bias=True):
        super().__init__()
        self.black_in_channels = black_in_channels
        self.black_out_channels = black_out_channels
        self.black_kernel_size = black_kernel_size
        self.black_stride = black_stride
        self.black_padding = black_padding
        self.black_dilation = black_dilation
        self.black_groups = black_groups
        black_k = 1.0 / math.sqrt(black_in_channels * black_kernel_size)
        self._black_parameters['black_weight'] = [
            random.uniform(-black_k, black_k)
            for _ in range(black_out_channels * (black_in_channels // black_groups) * black_kernel_size)
        ]
        if black_bias:
            self._black_parameters['black_bias'] = [random.uniform(-black_k, black_k) for _ in range(black_out_channels)]

    def black_forward(self, black_x):
        return {"black_op": "conv1d", "black_input": black_x}


class BlackConv2d(BlackModule):
    def __init__(self, black_in_channels, black_out_channels, black_kernel_size,
                 black_stride=1, black_padding=0, black_dilation=1, black_groups=1, black_bias=True):
        super().__init__()
        self.black_in_channels = black_in_channels
        self.black_out_channels = black_out_channels
        black_kh = black_kernel_size if isinstance(black_kernel_size, int) else black_kernel_size[0]
        black_kw = black_kernel_size if isinstance(black_kernel_size, int) else black_kernel_size[1]
        self.black_kernel_size = (black_kh, black_kw)
        self.black_stride = black_stride if isinstance(black_stride, tuple) else (black_stride, black_stride)
        self.black_padding = black_padding if isinstance(black_padding, tuple) else (black_padding, black_padding)
        self.black_dilation = black_dilation if isinstance(black_dilation, tuple) else (black_dilation, black_dilation)
        self.black_groups = black_groups
        black_k = 1.0 / math.sqrt(black_in_channels * black_kh * black_kw)
        self._black_parameters['black_weight'] = [
            random.uniform(-black_k, black_k)
            for _ in range(black_out_channels * (black_in_channels // black_groups) * black_kh * black_kw)
        ]
        if black_bias:
            self._black_parameters['black_bias'] = [random.uniform(-black_k, black_k) for _ in range(black_out_channels)]

    def black_forward(self, black_x):
        return {"black_op": "conv2d", "black_input": black_x}


class BlackConv3d(BlackModule):
    def __init__(self, black_in_channels, black_out_channels, black_kernel_size,
                 black_stride=1, black_padding=0, black_dilation=1, black_groups=1, black_bias=True):
        super().__init__()
        self.black_in_channels = black_in_channels
        self.black_out_channels = black_out_channels
        self.black_kernel_size = black_kernel_size
        self.black_stride = black_stride
        self.black_padding = black_padding
        self.black_dilation = black_dilation
        self.black_groups = black_groups

    def black_forward(self, black_x):
        return {"black_op": "conv3d", "black_input": black_x}


class BlackConvTranspose2d(BlackModule):
    def __init__(self, black_in_channels, black_out_channels, black_kernel_size,
                 black_stride=1, black_padding=0, black_output_padding=0, black_bias=True):
        super().__init__()
        self.black_in_channels = black_in_channels
        self.black_out_channels = black_out_channels
        self.black_kernel_size = black_kernel_size
        self.black_stride = black_stride
        self.black_padding = black_padding
        self.black_output_padding = black_output_padding

    def black_forward(self, black_x):
        return {"black_op": "conv_transpose2d", "black_input": black_x}


class BlackSequential(BlackModule):
    def __init__(self, *black_modules):
        super().__init__()
        for black_idx, black_mod in enumerate(black_modules):
            self._black_submodules[f"black_{black_idx}"] = black_mod

    def black_forward(self, black_x):
        black_out = black_x
        for black_mod in self._black_submodules.values():
            black_out = black_mod(black_out)
        return black_out


class BlackResidual(BlackModule):
    def __init__(self, black_module):
        super().__init__()
        self.black_module = black_module

    def black_forward(self, black_x):
        return {"black_op": "residual", "black_input": black_x, "black_output": self.black_module(black_x)}


class BlackMLP(BlackModule):
    def __init__(self, black_in_features, black_hidden_features, black_out_features,
                 black_act='gelu', black_dropout=0.0):
        super().__init__()
        self.black_fc1 = BlackLinear(black_in_features, black_hidden_features)
        self.black_fc2 = BlackLinear(black_hidden_features, black_out_features)
        self.black_act = black_act
        self.black_drop = BlackDropout(black_dropout)

    def black_forward(self, black_x):
        black_h = self.black_fc1(black_x)
        black_h = {"black_op": self.black_act, "black_input": black_h}
        black_h = self.black_drop(black_h)
        black_h = self.black_fc2(black_h)
        black_h = self.black_drop(black_h)
        return black_h


class BlackSwiGLU(BlackModule):
    def __init__(self, black_in_features, black_hidden_features):
        super().__init__()
        self.black_w1 = BlackLinear(black_in_features, black_hidden_features, black_bias=False)
        self.black_w2 = BlackLinear(black_hidden_features, black_in_features, black_bias=False)
        self.black_w3 = BlackLinear(black_in_features, black_hidden_features, black_bias=False)

    def black_forward(self, black_x):
        black_gate = self.black_w1(black_x)
        black_up = self.black_w3(black_x)
        return self.black_w2({"black_op": "silu_gate", "black_gate": black_gate, "black_up": black_up})


from black_ferox.black_nn import black_transformers
