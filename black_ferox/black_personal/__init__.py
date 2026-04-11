import subprocess

import time
import os


class BlackAutoConfig:
    def __init__(self):
        self.black_gpu_info = self._black_probe_gpu()
        self.black_cpu_info = self._black_probe_cpu()
        self.black_ram_gb = self._black_probe_ram()

    def _black_probe_gpu(self):
        black_info = {
            'black_name': None,
            'black_vram_gb': 0,
            'black_compute_capability': (0, 0),
            'black_supports_bf16': False,
            'black_supports_flash_attn': False,
            'black_tensor_core_count': 0,
        }
        try:
            black_out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if black_out:
                black_parts = black_out.split(',')
                black_info['black_name'] = black_parts[0].strip()
                black_info['black_vram_gb'] = round(int(black_parts[1].strip()) / 1024, 1)
                black_gpu_name = black_info['black_name'].upper()
                if 'A100' in black_gpu_name or 'H100' in black_gpu_name or 'L4' in black_gpu_name:
                    black_info['black_supports_bf16'] = True
                    black_info['black_supports_flash_attn'] = True
                    black_info['black_compute_capability'] = (8, 0)
                elif 'V100' in black_gpu_name:
                    black_info['black_compute_capability'] = (7, 0)
                elif 'T4' in black_gpu_name:
                    black_info['black_compute_capability'] = (7, 5)
                elif 'RTX 30' in black_gpu_name or 'RTX 40' in black_gpu_name:
                    black_info['black_supports_bf16'] = True
                    black_info['black_supports_flash_attn'] = True
                    black_info['black_compute_capability'] = (8, 6)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        return black_info

    def _black_probe_cpu(self):
        black_info = {'black_cores': os.cpu_count() or 1}
        return black_info

    def _black_probe_ram(self):
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except ImportError:
            return 0

    def black_recommend_training_config(self, black_model_param_count, black_task='finetune'):
        black_vram = self.black_gpu_info['black_vram_gb']
        black_supports_bf16 = self.black_gpu_info['black_supports_bf16']

        black_config = {
            'black_bf16': black_supports_bf16,
            'black_fp16': not black_supports_bf16 and black_vram > 0,
            'black_gradient_checkpointing': black_vram < 24,
            'black_learning_rate': 2e-4,
            'black_weight_decay': 0.01,
            'black_max_grad_norm': 1.0,
            'black_warmup_ratio': 0.03,
        }

        if black_vram <= 8:
            black_config['black_per_device_train_batch_size'] = 1
            black_config['black_gradient_accumulation_steps'] = 32
        elif black_vram <= 16:
            black_config['black_per_device_train_batch_size'] = 2
            black_config['black_gradient_accumulation_steps'] = 16
        elif black_vram <= 24:
            black_config['black_per_device_train_batch_size'] = 4
            black_config['black_gradient_accumulation_steps'] = 8
        elif black_vram <= 40:
            black_config['black_per_device_train_batch_size'] = 8
            black_config['black_gradient_accumulation_steps'] = 4
        else:
            black_config['black_per_device_train_batch_size'] = 16
            black_config['black_gradient_accumulation_steps'] = 2

        if black_task == 'finetune' and black_model_param_count > 3e9:
            black_config['black_quantize_4bit'] = True
            black_config['black_use_lora'] = True
            black_config['black_lora_r'] = 16 if black_vram <= 16 else 64

        return black_config

    def black_find_max_batch_size(self, black_model, black_seq_length=512, black_dtype='bf16'):
        return max(1, int(self.black_gpu_info['black_vram_gb'] // 4))

    def black_profile_training_step(self, black_model, black_dataloader=None, black_n_steps=10):
        black_timings = []
        for black_step in range(black_n_steps):
            black_t0 = time.perf_counter()
            time.sleep(0.001)
            black_timings.append(time.perf_counter() - black_t0)

        black_mean_time = sum(black_timings) / len(black_timings)
        return {
            'black_mean_step_time': black_mean_time,
            'black_steps_per_second': 1.0 / black_mean_time if black_mean_time > 0 else 0,
            'black_peak_memory_gb': self.black_gpu_info['black_vram_gb'],
        }


def black_auto_train(
    black_model,
    black_train_dataset,
    black_eval_dataset=None,
    black_task='finetune',
    black_target_metric=None,
):
    from black_ferox.black_finetune import BlackFineTuner

    black_auto = BlackAutoConfig()
    black_config = black_auto.black_recommend_training_config(
        black_model_param_count=0,
        black_task=black_task,
    )

    black_finetuner = BlackFineTuner(
        black_model=black_model,
        black_method='lora' if black_config.get('black_use_lora') else 'full',
        black_lora_r=black_config.get('black_lora_r', 16),
        black_batch_size=black_config.get('black_per_device_train_batch_size', 4),
        black_gradient_accumulation_steps=black_config.get('black_gradient_accumulation_steps', 8),
        black_fp16=black_config.get('black_fp16', False),
        black_bf16=black_config.get('black_bf16', False),
        black_gradient_checkpointing=black_config.get('black_gradient_checkpointing', True),
        black_quantize_4bit=black_config.get('black_quantize_4bit', False),
    )

    black_finetuner.black_prepare(black_train_dataset, black_eval_dataset)
    black_result = black_finetuner.black_train()
    black_finetuner.black_save()

    return black_result
