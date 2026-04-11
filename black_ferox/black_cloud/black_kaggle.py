import os
import time
import subprocess
import shutil


class BlackKaggleEnvironment:
    def __init__(self):
        self.black_gpu_type = self._black_detect_gpu()
        self.black_accelerator = self._black_detect_accelerator()
        self.black_session_time_remaining = self._black_get_time_remaining()
        self._black_start_time = time.time()

    def _black_detect_gpu(self):
        try:
            black_out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if 'P100' in black_out:
                return 'P100'
            if 'T4' in black_out:
                if black_out.count('T4') >= 2:
                    return 'T4x2'
                return 'T4'
            return black_out.split('\n')[0] if black_out else 'cpu'
        except (FileNotFoundError, subprocess.CalledProcessError):
            return 'cpu'

    def _black_detect_accelerator(self):
        try:
            black_out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if black_out:
                return 'gpu'
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        return 'cpu'

    def _black_get_time_remaining(self):
        return 9 * 60

    def black_setup(self, black_install_ferox=True):
        if black_install_ferox:
            subprocess.check_call(['pip', 'install', 'black-ferox', '-q'])

    def black_recommend_config(self, black_model_size_b=7):
        black_p100_config = {
            'black_batch_size': 2,
            'black_gradient_accumulation_steps': 16,
            'black_max_seq_length': 512,
            'black_lora_r': 8,
            'black_quantize_4bit': True,
            'black_fp16': True,
            'black_bf16': False,
        }
        black_t4x2_config = {
            'black_batch_size': 4,
            'black_gradient_accumulation_steps': 8,
            'black_max_seq_length': 1024,
            'black_lora_r': 16,
            'black_quantize_4bit': True,
            'black_fp16': True,
            'black_bf16': False,
        }
        black_gpu_map = {
            'P100': black_p100_config,
            'T4': black_p100_config,
            'T4x2': black_t4x2_config,
        }
        return black_gpu_map.get(self.black_gpu_type, black_p100_config)

    def black_time_aware_checkpoint(self, black_trainer, black_safety_margin_minutes=30):
        black_elapsed = (time.time() - self._black_start_time) / 60.0
        black_remaining = self.black_session_time_remaining - black_elapsed
        if black_remaining <= black_safety_margin_minutes:
            if hasattr(black_trainer, 'black_save'):
                black_trainer.black_save('./black_kaggle_checkpoint')
            return True
        return False

    def black_save_output(self, black_path, black_output_name):
        black_kaggle_out = '/kaggle/working'
        black_dst = os.path.join(black_kaggle_out, black_output_name)
        os.makedirs(black_dst, exist_ok=True)
        if os.path.isdir(black_path):
            shutil.copytree(black_path, black_dst, dirs_exist_ok=True)
        elif os.path.isfile(black_path):
            shutil.copy2(black_path, black_dst)


def black_kaggle_finetune_script(black_model_name, black_dataset_name, black_task='sft'):
    black_cells = [
        {
            "cell_type": "code",
            "source": [
                "!pip install black-ferox -q\n",
                "import black_ferox as black\n",
                "black_env = black.black_cloud.black_kaggle.BlackKaggleEnvironment()\n",
                "print('GPU:', black_env.black_gpu_type)\n",
                "print('Time remaining:', black_env.black_session_time_remaining, 'min')\n",
            ],
            "metadata": {},
            "outputs": [],
        },
        {
            "cell_type": "code",
            "source": [
                "black_config = black_env.black_recommend_config()\n",
                "black_finetuner = black.black_finetune.BlackFineTuner(\n",
                "    black_model=None,\n",
                "    black_method='lora',\n",
                "    **black_config,\n",
                ")\n",
            ],
            "metadata": {},
            "outputs": [],
        },
        {
            "cell_type": "code",
            "source": [
                "for black_step in range(1000):\n",
                "    if black_env.black_time_aware_checkpoint(black_finetuner):\n",
                "        print('Auto-saving checkpoint due to time limit')\n",
                "        break\n",
            ],
            "metadata": {},
            "outputs": [],
        },
    ]
    black_notebook = {
        "nbformat": 4,
        "nbformat_minor": 2,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "accelerator": "GPU",
        },
        "cells": black_cells,
    }
    return black_notebook
