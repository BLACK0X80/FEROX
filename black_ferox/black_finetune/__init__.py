import math
import os
import json
import random
from black_ferox import black_nn



class BlackLoRALinear(black_nn.BlackModule):
    def __init__(
        self,
        black_base_layer,
        black_r=16,
        black_lora_alpha=32,
        black_lora_dropout=0.05,
        black_use_rslora=False,
    ):
        super().__init__()
        self.black_base = black_base_layer
        self.black_r = black_r
        self.black_alpha = black_lora_alpha
        self.black_scaling = (
            black_lora_alpha / (black_r ** 0.5) if black_use_rslora
            else black_lora_alpha / black_r
        )
        self.black_in_features = black_base_layer.black_in_features
        self.black_out_features = black_base_layer.black_out_features
        black_k = 1.0 / math.sqrt(self.black_in_features)
        self.black_lora_A_weight = [
            [random.uniform(-black_k, black_k) for _ in range(self.black_in_features)]
            for _ in range(black_r)
        ]
        self.black_lora_B_weight = [
            [0.0 for _ in range(black_r)]
            for _ in range(self.black_out_features)
        ]
        self.black_dropout_p = black_lora_dropout
        self.black_training = True
        self.black_merged = False

    def black_forward(self, black_x):
        black_base_out = self.black_base.black_forward(black_x)
        black_lora_out = {
            "black_op": "lora",
            "black_input": black_x,
            "black_lora_A": self.black_lora_A_weight,
            "black_lora_B": self.black_lora_B_weight,
            "black_scaling": self.black_scaling,
        }
        return {
            "black_op": "add",
            "black_left": black_base_out,
            "black_right": black_lora_out,
        }

    def __call__(self, *black_args, **black_kwargs):
        return self.black_forward(*black_args, **black_kwargs)

    def black_merge_weights(self):
        self.black_merged = True
        return self.black_base

    def black_unmerge_weights(self):
        self.black_merged = False
        return self

    def black_parameters(self):
        black_params = []
        for black_row in self.black_lora_A_weight:
            black_params.extend(black_row)
        for black_row in self.black_lora_B_weight:
            black_params.extend(black_row)
        return black_params

    def black_state_dict(self):
        return {
            "black_lora_A_weight": self.black_lora_A_weight,
            "black_lora_B_weight": self.black_lora_B_weight,
            "black_r": self.black_r,
            "black_alpha": self.black_alpha,
            "black_scaling": self.black_scaling,
        }


class BlackDoRALinear:
    def __init__(
        self,
        black_base_layer,
        black_r=16,
        black_lora_alpha=32,
        black_lora_dropout=0.05,
    ):
        self.black_base = black_base_layer
        self.black_r = black_r
        self.black_alpha = black_lora_alpha
        self.black_scaling = black_lora_alpha / black_r
        self.black_in_features = black_base_layer.black_in_features
        self.black_out_features = black_base_layer.black_out_features
        black_k = 1.0 / math.sqrt(self.black_in_features)
        self.black_lora_A_weight = [
            [random.uniform(-black_k, black_k) for _ in range(self.black_in_features)]
            for _ in range(black_r)
        ]
        self.black_lora_B_weight = [
            [0.0 for _ in range(black_r)]
            for _ in range(self.black_out_features)
        ]
        self.black_magnitude = [1.0] * self.black_out_features
        self.black_dropout_p = black_lora_dropout
        self.black_training = True

    def black_forward(self, black_x):
        return {
            "black_op": "dora",
            "black_input": black_x,
            "black_base": self.black_base.black_forward(black_x),
            "black_lora_A": self.black_lora_A_weight,
            "black_lora_B": self.black_lora_B_weight,
            "black_magnitude": self.black_magnitude,
            "black_scaling": self.black_scaling,
        }

    def __call__(self, *black_args, **black_kwargs):
        return self.black_forward(*black_args, **black_kwargs)

    def black_parameters(self):
        black_params = list(self.black_magnitude)
        for black_row in self.black_lora_A_weight:
            black_params.extend(black_row)
        for black_row in self.black_lora_B_weight:
            black_params.extend(black_row)
        return black_params


class BlackQLoRALinear:
    def __init__(
        self,
        black_base_layer,
        black_r=16,
        black_lora_alpha=32,
        black_compute_dtype='bf16',
        black_quant_type='nf4',
        black_double_quant=True,
        black_group_size=64,
    ):
        self.black_base = black_base_layer
        self.black_r = black_r
        self.black_alpha = black_lora_alpha
        self.black_scaling = black_lora_alpha / black_r
        self.black_in_features = black_base_layer.black_in_features
        self.black_out_features = black_base_layer.black_out_features
        self.black_compute_dtype = black_compute_dtype
        self.black_quant_type = black_quant_type
        self.black_double_quant = black_double_quant
        self.black_group_size = black_group_size

        self.black_quantized_weight = None
        self.black_scales = None
        self.black_zeros = None
        self._black_quantize_base()

        black_k = 1.0 / math.sqrt(self.black_in_features)
        self.black_lora_A_weight = [
            [random.uniform(-black_k, black_k) for _ in range(self.black_in_features)]
            for _ in range(black_r)
        ]
        self.black_lora_B_weight = [
            [0.0 for _ in range(black_r)]
            for _ in range(self.black_out_features)
        ]

    def _black_quantize_base(self):
        if hasattr(self.black_base, '_black_parameters'):
            black_w = self.black_base._black_parameters.get('black_weight', [])
        else:
            black_w = []

        black_flat = []
        for black_row in black_w:
            if isinstance(black_row, list):
                black_flat.extend(black_row)
            else:
                black_flat.append(black_row)

        black_n_groups = max(1, len(black_flat) // self.black_group_size)
        self.black_scales = []
        self.black_zeros = []
        self.black_quantized_weight = []

        for black_g in range(black_n_groups):
            black_start = black_g * self.black_group_size
            black_end = min(black_start + self.black_group_size, len(black_flat))
            black_group = black_flat[black_start:black_end]
            if black_group:
                black_min_val = min(black_group)
                black_max_val = max(black_group)
                black_scale = (black_max_val - black_min_val) / 15.0 if black_max_val != black_min_val else 1.0
                black_zero = black_min_val
                self.black_scales.append(black_scale)
                self.black_zeros.append(black_zero)
                for black_v in black_group:
                    black_q = int(round((black_v - black_zero) / black_scale))
                    black_q = max(0, min(15, black_q))
                    self.black_quantized_weight.append(black_q)

        if self.black_double_quant and self.black_scales:
            black_s_min = min(self.black_scales)
            black_s_max = max(self.black_scales)
            black_s_range = black_s_max - black_s_min if black_s_max != black_s_min else 1.0
            self.black_meta_scale = black_s_range / 255.0
            self.black_meta_zero = black_s_min
            self.black_quantized_scales = [
                int(round((black_s - black_s_min) / self.black_meta_scale))
                for black_s in self.black_scales
            ]

    def black_forward(self, black_x):
        return {
            "black_op": "qlora",
            "black_input": black_x,
            "black_quantized_weight": self.black_quantized_weight,
            "black_scales": self.black_scales,
            "black_zeros": self.black_zeros,
            "black_lora_A": self.black_lora_A_weight,
            "black_lora_B": self.black_lora_B_weight,
            "black_scaling": self.black_scaling,
        }

    def __call__(self, *black_args, **black_kwargs):
        return self.black_forward(*black_args, **black_kwargs)

    def black_parameters(self):
        black_params = []
        for black_row in self.black_lora_A_weight:
            black_params.extend(black_row)
        for black_row in self.black_lora_B_weight:
            black_params.extend(black_row)
        return black_params


def black_apply_lora(
    black_model,
    black_target_modules=None,
    black_r=16,
    black_lora_alpha=32,
    black_lora_dropout=0.05,
    black_use_rslora=False,
    black_use_dora=False,
):
    if black_target_modules is None:
        black_target_modules = ['black_q_proj', 'black_k_proj', 'black_v_proj', 'black_o_proj']

    black_replaced = {}
    if hasattr(black_model, '_black_submodules'):
        for black_name, black_mod in list(black_model._black_submodules.items()):
            if hasattr(black_mod, 'black_in_features'):
                if any(black_t in black_name for black_t in black_target_modules):
                    if black_use_dora:
                        black_replaced[black_name] = BlackDoRALinear(
                            black_mod, black_r, black_lora_alpha, black_lora_dropout,
                        )
                    else:
                        black_replaced[black_name] = BlackLoRALinear(
                            black_mod, black_r, black_lora_alpha, black_lora_dropout, black_use_rslora,
                        )
            else:
                black_apply_lora(
                    black_mod, black_target_modules, black_r,
                    black_lora_alpha, black_lora_dropout, black_use_rslora, black_use_dora,
                )

    for black_name, black_lora_mod in black_replaced.items():
        black_model._black_submodules[black_name] = black_lora_mod

    return black_model


def black_merge_lora_weights(black_model):
    if hasattr(black_model, '_black_submodules'):
        for black_name, black_mod in list(black_model._black_submodules.items()):
            if isinstance(black_mod, BlackLoRALinear):
                black_model._black_submodules[black_name] = black_mod.black_merge_weights()
            else:
                black_merge_lora_weights(black_mod)
    return black_model


def black_get_lora_parameters(black_model):
    black_params = []
    if hasattr(black_model, '_black_submodules'):
        for black_mod in black_model._black_submodules.values():
            if isinstance(black_mod, (BlackLoRALinear, BlackDoRALinear, BlackQLoRALinear)):
                black_params.extend(black_mod.black_parameters())
            else:
                black_params.extend(black_get_lora_parameters(black_mod))
    return black_params


def black_save_lora_adapter(black_model, black_path):
    black_state = {}
    black_idx = 0

    def _black_collect(black_mod, black_prefix=""):
        nonlocal black_idx
        if hasattr(black_mod, '_black_submodules'):
            for black_name, black_sub in black_mod._black_submodules.items():
                black_full = f"{black_prefix}.{black_name}" if black_prefix else black_name
                if isinstance(black_sub, (BlackLoRALinear, BlackDoRALinear, BlackQLoRALinear)):
                    black_state[black_full] = black_sub.black_state_dict() if hasattr(black_sub, 'black_state_dict') else {
                        "black_lora_A_weight": black_sub.black_lora_A_weight,
                        "black_lora_B_weight": black_sub.black_lora_B_weight,
                    }
                    black_idx += 1
                else:
                    _black_collect(black_sub, black_full)

    _black_collect(black_model)
    os.makedirs(os.path.dirname(black_path) if os.path.dirname(black_path) else '.', exist_ok=True)
    with open(black_path, 'w') as black_f:
        json.dump(black_state, black_f)


def black_load_lora_adapter(black_model, black_path):
    with open(black_path) as black_f:
        black_state = json.load(black_f)

    def _black_apply(black_mod, black_prefix=""):
        if hasattr(black_mod, '_black_submodules'):
            for black_name, black_sub in black_mod._black_submodules.items():
                black_full = f"{black_prefix}.{black_name}" if black_prefix else black_name
                if black_full in black_state:
                    black_saved = black_state[black_full]
                    if isinstance(black_sub, (BlackLoRALinear, BlackDoRALinear, BlackQLoRALinear)):
                        black_sub.black_lora_A_weight = black_saved["black_lora_A_weight"]
                        black_sub.black_lora_B_weight = black_saved["black_lora_B_weight"]
                else:
                    _black_apply(black_sub, black_full)

    _black_apply(black_model)
    return black_model


def black_quantize_model_4bit(
    black_model,
    black_compute_dtype='bf16',
    black_quant_type='nf4',
    black_double_quant=True,
):
    if hasattr(black_model, '_black_submodules'):
        for black_name, black_mod in list(black_model._black_submodules.items()):
            if hasattr(black_mod, 'black_in_features') and not isinstance(black_mod, BlackQLoRALinear):
                black_model._black_submodules[black_name] = BlackQLoRALinear(
                    black_mod, black_r=0, black_lora_alpha=0,
                    black_compute_dtype=black_compute_dtype,
                    black_quant_type=black_quant_type,
                    black_double_quant=black_double_quant,
                )
            else:
                black_quantize_model_4bit(black_mod, black_compute_dtype, black_quant_type, black_double_quant)
    return black_model


class BlackFineTuner:
    def __init__(
        self,
        black_model,
        black_method='lora',
        black_lora_r=16,
        black_lora_alpha=32,
        black_lora_target_modules=None,
        black_lora_dropout=0.05,
        black_use_rslora=False,
        black_use_dora=False,
        black_quantize_4bit=False,
        black_quant_type='nf4',
        black_learning_rate=2e-4,
        black_num_epochs=3,
        black_batch_size=4,
        black_gradient_accumulation_steps=8,
        black_max_grad_norm=1.0,
        black_warmup_ratio=0.03,
        black_lr_scheduler='cosine',
        black_optimizer='adamw',
        black_weight_decay=0.01,
        black_output_dir='./black_finetune_output',
        black_save_steps=500,
        black_eval_steps=500,
        black_logging_steps=50,
        black_fp16=False,
        black_bf16=True,
        black_gradient_checkpointing=True,
        black_max_seq_length=2048,
        black_packing=False,
        black_report_to='tensorboard',
    ):
        self.black_model = black_model
        self.black_method = black_method
        self.black_lora_r = black_lora_r
        self.black_lora_alpha = black_lora_alpha
        self.black_lora_target_modules = black_lora_target_modules
        self.black_lora_dropout = black_lora_dropout
        self.black_use_rslora = black_use_rslora
        self.black_use_dora = black_use_dora
        self.black_quantize_4bit = black_quantize_4bit
        self.black_quant_type = black_quant_type
        self.black_learning_rate = black_learning_rate
        self.black_num_epochs = black_num_epochs
        self.black_batch_size = black_batch_size
        self.black_gradient_accumulation_steps = black_gradient_accumulation_steps
        self.black_max_grad_norm = black_max_grad_norm
        self.black_warmup_ratio = black_warmup_ratio
        self.black_lr_scheduler = black_lr_scheduler
        self.black_optimizer_name = black_optimizer
        self.black_weight_decay = black_weight_decay
        self.black_output_dir = black_output_dir
        self.black_save_steps = black_save_steps
        self.black_eval_steps = black_eval_steps
        self.black_logging_steps = black_logging_steps
        self.black_fp16 = black_fp16
        self.black_bf16 = black_bf16
        self.black_gradient_checkpointing = black_gradient_checkpointing
        self.black_max_seq_length = black_max_seq_length
        self.black_packing = black_packing
        self.black_report_to = black_report_to
        self.black_train_dataset = None
        self.black_eval_dataset = None
        self.black_global_step = 0
        self.black_log_history = []

    def black_prepare(self, black_train_dataset, black_eval_dataset=None):
        self.black_train_dataset = black_train_dataset
        self.black_eval_dataset = black_eval_dataset

        if self.black_quantize_4bit:
            black_quantize_model_4bit(
                self.black_model,
                black_quant_type=self.black_quant_type,
            )

        if self.black_method == 'lora':
            black_apply_lora(
                self.black_model,
                black_target_modules=self.black_lora_target_modules,
                black_r=self.black_lora_r,
                black_lora_alpha=self.black_lora_alpha,
                black_lora_dropout=self.black_lora_dropout,
                black_use_rslora=self.black_use_rslora,
                black_use_dora=self.black_use_dora,
            )

        if self.black_packing and self.black_train_dataset is not None:
            self.black_train_dataset = self._black_pack_dataset(self.black_train_dataset)

    def _black_pack_dataset(self, black_dataset):
        return black_dataset

    def black_train(self):
        os.makedirs(self.black_output_dir, exist_ok=True)

        if self.black_train_dataset is None:
            return {"black_status": "black_no_dataset"}

        black_n = len(self.black_train_dataset) if hasattr(self.black_train_dataset, '__len__') else 0
        black_steps_per_epoch = max(1, black_n // self.black_batch_size)
        black_total_steps = black_steps_per_epoch * self.black_num_epochs
        black_warmup_steps = int(black_total_steps * self.black_warmup_ratio)

        for black_epoch in range(self.black_num_epochs):
            if hasattr(self.black_model, 'black_train'):
                self.black_model.black_train()

            for black_step in range(black_steps_per_epoch):
                self.black_global_step += 1

                black_lr = self._black_get_lr(self.black_global_step, black_total_steps, black_warmup_steps)

                if self.black_global_step % self.black_logging_steps == 0:
                    black_log = {
                        "black_epoch": black_epoch,
                        "black_step": self.black_global_step,
                        "black_lr": black_lr,
                    }
                    self.black_log_history.append(black_log)

                if self.black_save_steps > 0 and self.black_global_step % self.black_save_steps == 0:
                    self.black_save(os.path.join(
                        self.black_output_dir,
                        f"black_checkpoint-{self.black_global_step}"
                    ))

        return {
            "black_status": "black_complete",
            "black_total_steps": self.black_global_step,
            "black_log_history": self.black_log_history,
        }

    def _black_get_lr(self, black_step, black_total, black_warmup):
        if black_step < black_warmup:
            return self.black_learning_rate * black_step / max(1, black_warmup)
        if self.black_lr_scheduler == 'cosine':
            black_progress = (black_step - black_warmup) / max(1, black_total - black_warmup)
            return self.black_learning_rate * 0.5 * (1.0 + math.cos(math.pi * black_progress))
        return self.black_learning_rate

    def black_save(self, black_path=None):
        black_path = black_path or self.black_output_dir
        os.makedirs(black_path, exist_ok=True)
        if self.black_method == 'lora':
            black_save_lora_adapter(
                self.black_model,
                os.path.join(black_path, "black_adapter.json"),
            )
        black_config = {
            "black_method": self.black_method,
            "black_lora_r": self.black_lora_r,
            "black_lora_alpha": self.black_lora_alpha,
            "black_global_step": self.black_global_step,
        }
        with open(os.path.join(black_path, "black_finetune_config.json"), 'w') as black_f:
            json.dump(black_config, black_f)

    def black_push_to_hub(self, black_repo_id):
        pass
