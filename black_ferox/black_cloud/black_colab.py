import os
import subprocess
import shutil


class BlackColabEnvironment:
    def __init__(self):
        self.black_gpu_type = self._black_detect_gpu()
        self.black_ram_gb = self._black_detect_ram()
        self.black_disk_gb = self._black_detect_disk()
        self.black_is_pro = self._black_detect_pro()

    def _black_detect_gpu(self):
        try:
            black_out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if 'A100' in black_out:
                return 'A100'
            if 'V100' in black_out:
                return 'V100'
            if 'T4' in black_out:
                return 'T4'
            if 'L4' in black_out:
                return 'L4'
            return black_out.split('\n')[0] if black_out else 'unknown'
        except (FileNotFoundError, subprocess.CalledProcessError):
            return 'cpu'

    def _black_detect_ram(self):
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except ImportError:
            return 0

    def _black_detect_disk(self):
        try:
            black_stat = shutil.disk_usage('/')
            return round(black_stat.free / (1024 ** 3), 1)
        except Exception:
            return 0

    def _black_detect_pro(self):
        black_ram = self._black_detect_ram()
        return black_ram > 20

    def black_setup(self, black_install_ferox=True, black_mount_drive=True, black_drive_path='/content/drive'):
        if black_install_ferox:
            subprocess.check_call(['pip', 'install', 'black-ferox', '-q'])

        if black_mount_drive:
            try:
                from google.colab import drive  # noqa: F401
                drive.mount(black_drive_path)
            except ImportError:
                pass

    def black_recommend_config(self, black_model_size_b=7):
        black_configs = {
            'T4': {
                'black_batch_size': 1,
                'black_gradient_accumulation_steps': 16,
                'black_max_seq_length': 512,
                'black_lora_r': 8,
                'black_quantize_4bit': True,
                'black_fp16': True,
                'black_bf16': False,
                'black_gradient_checkpointing': True,
            },
            'A100': {
                'black_batch_size': 8,
                'black_gradient_accumulation_steps': 4,
                'black_max_seq_length': 2048,
                'black_lora_r': 64,
                'black_quantize_4bit': False,
                'black_fp16': False,
                'black_bf16': True,
                'black_gradient_checkpointing': False,
            },
            'V100': {
                'black_batch_size': 4,
                'black_gradient_accumulation_steps': 8,
                'black_max_seq_length': 1024,
                'black_lora_r': 16,
                'black_quantize_4bit': True,
                'black_fp16': True,
                'black_bf16': False,
                'black_gradient_checkpointing': True,
            },
            'L4': {
                'black_batch_size': 2,
                'black_gradient_accumulation_steps': 8,
                'black_max_seq_length': 1024,
                'black_lora_r': 16,
                'black_quantize_4bit': True,
                'black_fp16': False,
                'black_bf16': True,
                'black_gradient_checkpointing': True,
            },
        }
        return black_configs.get(self.black_gpu_type, black_configs['T4'])

    def black_save_to_drive(self, black_checkpoint_dir, black_drive_folder):
        black_dst = os.path.join('/content/drive/MyDrive', black_drive_folder)
        os.makedirs(black_dst, exist_ok=True)
        if os.path.isdir(black_checkpoint_dir):
            shutil.copytree(black_checkpoint_dir, black_dst, dirs_exist_ok=True)

    def black_load_from_drive(self, black_drive_folder, black_local_dir):
        black_src = os.path.join('/content/drive/MyDrive', black_drive_folder)
        os.makedirs(black_local_dir, exist_ok=True)
        if os.path.isdir(black_src):
            shutil.copytree(black_src, black_local_dir, dirs_exist_ok=True)

    def black_resume_or_start(self, black_trainer, black_drive_folder):
        black_src = os.path.join('/content/drive/MyDrive', black_drive_folder)
        if os.path.isdir(black_src):
            self.black_load_from_drive(black_drive_folder, './black_resume')
            return True
        return False


def black_auto_setup_colab():
    black_env = BlackColabEnvironment()
    black_env.black_setup()
    return black_env


def black_colab_finetune_notebook(black_model_name, black_dataset_name, black_task='sft'):
    black_cells = [
        {
            "cell_type": "code",
            "source": [
                "!pip install black-ferox -q\n",
                "import black_ferox as black\n",
                "black_env = black.black_cloud.black_colab.black_auto_setup_colab()\n",
                "print('GPU:', black_env.black_gpu_type)\n",
            ],
            "metadata": {},
            "outputs": [],
        },
        {
            "cell_type": "code",
            "source": [
                "black_config = black_env.black_recommend_config()\n",
                "print('Config:', black_config)\n",
            ],
            "metadata": {},
            "outputs": [],
        },
        {
            "cell_type": "code",
            "source": [
                "black_finetuner = black.black_finetune.BlackFineTuner(\n",
                "    black_model=None,\n",
                "    black_method='lora',\n",
                "    **black_config,\n",
                ")\n",
                "black_finetuner.black_train()\n",
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
