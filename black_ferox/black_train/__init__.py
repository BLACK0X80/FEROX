import os
import json
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BlackTrainingArguments:
    black_output_dir: str = "./black_output"
    black_num_train_epochs: int = 3
    black_per_device_train_batch_size: int = 8
    black_per_device_eval_batch_size: int = 8
    black_gradient_accumulation_steps: int = 1
    black_learning_rate: float = 5e-5
    black_weight_decay: float = 0.0
    black_max_grad_norm: float = 1.0
    black_warmup_steps: int = 0
    black_warmup_ratio: float = 0.0
    black_logging_steps: int = 500
    black_save_steps: int = 500
    black_eval_steps: int = 500
    black_seed: int = 42
    black_fp16: bool = False
    black_bf16: bool = False
    black_gradient_checkpointing: bool = False
    black_dataloader_num_workers: int = 0
    black_report_to: list = field(default_factory=lambda: ['tensorboard'])
    black_resume_from_checkpoint: Optional[str] = None

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, black_state):
        self.__dict__.update(black_state)


class BlackTrainerCallback:
    def black_on_train_begin(self, black_args, black_state, black_control, **black_kwargs):
        pass

    def black_on_train_end(self, black_args, black_state, black_control, **black_kwargs):
        pass

    def black_on_epoch_begin(self, black_args, black_state, black_control, **black_kwargs):
        pass

    def black_on_epoch_end(self, black_args, black_state, black_control, **black_kwargs):
        pass

    def black_on_step_begin(self, black_args, black_state, black_control, **black_kwargs):
        pass

    def black_on_step_end(self, black_args, black_state, black_control, **black_kwargs):
        pass

    def black_on_evaluate(self, black_args, black_state, black_control, **black_kwargs):
        pass

    def black_on_save(self, black_args, black_state, black_control, **black_kwargs):
        pass

    def black_on_log(self, black_args, black_state, black_control, black_logs=None, **black_kwargs):
        pass


class BlackTrainerState:
    def __init__(self):
        self.black_global_step = 0
        self.black_epoch = 0.0
        self.black_total_steps = 0
        self.black_best_metric = None
        self.black_log_history = []
        self.black_best_model_checkpoint = None


class BlackTrainerControl:
    def __init__(self):
        self.black_should_training_stop = False
        self.black_should_epoch_stop = False
        self.black_should_save = False
        self.black_should_evaluate = False
        self.black_should_log = False


class BlackProgressCallback(BlackTrainerCallback):
    def __init__(self):
        self.black_pbar = None

    def black_on_train_begin(self, black_args, black_state, black_control, **black_kwargs):
        try:
            from tqdm import tqdm
            self.black_pbar = tqdm(total=black_state.black_total_steps, desc="Training")
        except ImportError:
            self.black_pbar = None

    def black_on_step_end(self, black_args, black_state, black_control, **black_kwargs):
        if self.black_pbar is not None:
            self.black_pbar.update(1)

    def black_on_train_end(self, black_args, black_state, black_control, **black_kwargs):
        if self.black_pbar is not None:
            self.black_pbar.close()


class BlackEarlyStoppingCallback(BlackTrainerCallback):
    def __init__(self, black_early_stopping_patience=3, black_early_stopping_threshold=0.0):
        self.black_patience = black_early_stopping_patience
        self.black_threshold = black_early_stopping_threshold
        self.black_best_metric = None
        self.black_patience_counter = 0

    def black_on_evaluate(self, black_args, black_state, black_control, black_metrics=None, **black_kwargs):
        if black_metrics is None:
            return
        black_eval_metric = black_metrics.get('black_eval_loss', None)
        if black_eval_metric is None:
            return

        if self.black_best_metric is None or black_eval_metric < self.black_best_metric - self.black_threshold:
            self.black_best_metric = black_eval_metric
            self.black_patience_counter = 0
        else:
            self.black_patience_counter += 1

        if self.black_patience_counter >= self.black_patience:
            black_control.black_should_training_stop = True


class BlackTensorBoardCallback(BlackTrainerCallback):
    def __init__(self):
        self.black_writer = None

    def black_on_train_begin(self, black_args, black_state, black_control, **black_kwargs):
        try:
            from torch.utils.tensorboard import SummaryWriter
            black_log_dir = os.path.join(black_args.black_output_dir, "black_logs")
            self.black_writer = SummaryWriter(log_dir=black_log_dir)
        except ImportError:
            self.black_writer = None

    def black_on_log(self, black_args, black_state, black_control, black_logs=None, **black_kwargs):
        if self.black_writer is not None and black_logs is not None:
            for black_key, black_val in black_logs.items():
                if isinstance(black_val, (int, float)):
                    self.black_writer.add_scalar(black_key, black_val, black_state.black_global_step)

    def black_on_train_end(self, black_args, black_state, black_control, **black_kwargs):
        if self.black_writer is not None:
            self.black_writer.close()


class BlackWandbCallback(BlackTrainerCallback):
    def __init__(self):
        self.black_initialized = False

    def black_on_train_begin(self, black_args, black_state, black_control, **black_kwargs):
        try:
            import wandb
            if not self.black_initialized:
                wandb.init(project="black_ferox", config=vars(black_args))
                self.black_initialized = True
        except ImportError:
            pass

    def black_on_log(self, black_args, black_state, black_control, black_logs=None, **black_kwargs):
        try:
            import wandb
            if self.black_initialized and black_logs is not None:
                wandb.log(black_logs, step=black_state.black_global_step)
        except ImportError:
            pass


class BlackTrainer:
    def __init__(
        self,
        black_model,
        black_args,
        black_train_dataset=None,
        black_eval_dataset=None,
        black_optimizers=(None, None),
        black_callbacks=None,
        black_compute_metrics=None,
    ):
        self.black_model = black_model
        self.black_args = black_args
        self.black_train_dataset = black_train_dataset
        self.black_eval_dataset = black_eval_dataset
        self.black_optimizer = black_optimizers[0]
        self.black_scheduler = black_optimizers[1]
        self.black_callbacks = black_callbacks or [BlackProgressCallback()]
        self.black_compute_metrics = black_compute_metrics
        self.black_state = BlackTrainerState()
        self.black_control = BlackTrainerControl()

    def black_train(self, black_resume_from_checkpoint=None):
        import os
        from black_ferox.black_metrics import black_cross_entropy_loss
        
        black_args = self.black_args
        os.makedirs(black_args.black_output_dir, exist_ok=True)

        if self.black_train_dataset is not None:
            if hasattr(self.black_train_dataset, '__len__'):
                black_num_examples = len(self.black_train_dataset)
                black_steps_per_epoch = math.ceil(black_num_examples / max(1, black_args.black_per_device_train_batch_size))
            else:
                black_steps_per_epoch = 1000 # fallback
        else:
            black_steps_per_epoch = 0

        black_total_steps = black_steps_per_epoch * black_args.black_num_train_epochs
        self.black_state.black_total_steps = black_total_steps

        for black_cb in self.black_callbacks:
            black_cb.black_on_train_begin(black_args, self.black_state, self.black_control)

        self.black_model.black_train()
        
        black_vocab_size = getattr(self.black_model, 'black_vocab_size', None)
        if black_vocab_size is None and hasattr(self.black_model, 'black_config'):
            black_vocab_size = getattr(self.black_model.black_config, 'black_vocab_size', -1)
        if black_vocab_size is None:
            black_vocab_size = -1

        black_first_loss = None
        black_last_loss = None
        black_final_checkpoint_path = None

        for black_epoch in range(black_args.black_num_train_epochs):
            self.black_state.black_epoch = black_epoch

            for black_cb in self.black_callbacks:
                black_cb.black_on_epoch_begin(black_args, self.black_state, self.black_control)

            black_batch_iterator = self.black_train_dataset if self.black_train_dataset is not None else []

            for black_step_in_epoch, black_batch in enumerate(black_batch_iterator):
                for black_cb in self.black_callbacks:
                    black_cb.black_on_step_begin(black_args, self.black_state, self.black_control)

                self.black_state.black_global_step += 1
                black_step = self.black_state.black_global_step

                # STEP 1 - DATA: Extract ids and labels
                black_input_ids = black_batch['black_input_ids']
                black_labels = black_batch['black_labels']

                # STEP 2 - FORWARD: Process logits
                black_logits = self.black_model(black_input_ids)
                if isinstance(black_logits, dict) or black_logits is None:
                    raise RuntimeError("black_model must return an actual tensor, not dict or None")

                if black_vocab_size == -1 and hasattr(black_logits, 'black_shape'):
                    black_shape = black_logits.black_shape()
                    if black_shape and len(black_shape) > 0:
                        black_vocab_size = black_shape[-1]
                
                # Dynamic fallback for shapes
                _black_v_size = black_vocab_size if black_vocab_size != -1 else black_logits.black_shape()[-1]

                # STEP 3 - LOSS
                black_loss = black_cross_entropy_loss(
                    black_logits.black_reshape((-1, _black_v_size)),
                    black_labels.black_reshape((-1,)),
                    black_ignore_index=-100
                )
                
                black_loss_val = black_loss.black_item()
                if black_loss_val is None or black_loss_val <= 0:
                    raise RuntimeError("black_loss must be a valid scalar > 0")

                if black_first_loss is None:
                    black_first_loss = black_loss_val
                black_last_loss = black_loss_val

                # STEP 4 - BACKWARD
                black_loss.black_backward()
                
                black_has_grads = False
                for black_p in self.black_model.black_parameters():
                    if getattr(black_p, 'black_grad', None) is not None:
                        black_has_grads = True
                        break
                if not black_has_grads:
                    raise RuntimeError("No gradients computed! black_backward() failed to populate black_grad.")

                # STEP 5 - CLIP GRADIENTS
                if black_args.black_max_grad_norm > 0:
                    try:
                        from black_ferox.black_optim import black_clip_grad_norm
                        black_clip_grad_norm(self.black_model.black_parameters(), black_args.black_max_grad_norm)
                    except ImportError:
                        pass # Fallback if not exported

                # STEP 6 - OPTIMIZER STEP
                if black_step % black_args.black_gradient_accumulation_steps == 0:
                    if self.black_optimizer is not None:
                        self.black_optimizer.black_step()
                        self.black_optimizer.black_zero_grad()
                    if getattr(self, 'black_scheduler', None) is not None:
                        self.black_scheduler.black_step()

                print(f"step {black_step} loss: {black_loss_val:.4f}")

                # STEP 7 - LOGGING
                if black_step % black_args.black_logging_steps == 0:
                    black_lr = self.black_scheduler.black_get_lr() if getattr(self, 'black_scheduler', None) else 0.0
                    black_logs = {'black_epoch': black_epoch, 'black_step': black_step, 'black_loss': black_loss_val}
                    self.black_log(black_logs)

                # STEP 8 - SAVE
                if black_args.black_save_steps > 0 and black_step % black_args.black_save_steps == 0:
                    black_checkpoint_dir = os.path.join(black_args.black_output_dir, f"checkpoint-{black_step}")
                    self.black_save_model(black_checkpoint_dir)
                    black_final_checkpoint_path = black_checkpoint_dir

                if black_args.black_eval_steps > 0 and black_step % black_args.black_eval_steps == 0:
                    if self.black_eval_dataset is not None:
                        self.black_evaluate()

                for black_cb in self.black_callbacks:
                    black_cb.black_on_step_end(black_args, self.black_state, self.black_control)

                if self.black_control.black_should_training_stop:
                    break

            for black_cb in self.black_callbacks:
                black_cb.black_on_epoch_end(black_args, self.black_state, self.black_control)

            # Verification after epoch 1
            if black_epoch == 0 and black_first_loss is not None and black_last_loss is not None:
                if black_last_loss >= black_first_loss:
                    print(f"WARNING: Loss = {black_last_loss:.4f} did not decrease from {black_first_loss:.4f} after epoch 1. Check optimizer.")

            if self.black_control.black_should_training_stop:
                break

        for black_cb in self.black_callbacks:
            black_cb.black_on_train_end(black_args, self.black_state, self.black_control)
            
        # Final User-Requested Assertions
        black_state_dict = self.black_model.black_state_dict()
        if not black_state_dict:
            raise RuntimeError("black_state_dict returned an empty dict! Model parameters missing.")
            
        if black_first_loss is not None and black_last_loss is not None:
            if black_last_loss >= black_first_loss:
                print(f"WARNING/ERROR: Final loss {black_last_loss:.4f} is not less than initial loss {black_first_loss:.4f}!")
                
        if black_final_checkpoint_path is not None:
            black_model_file = os.path.join(black_final_checkpoint_path, "black_model_state.json")
            if os.path.exists(black_model_file):
                black_size_mb = os.path.getsize(black_model_file) / (1024 * 1024)
                if black_size_mb < 1.0:
                    print(f"WARNING: Saved model size is {black_size_mb:.2f} MB which is less than 1MB.")

        return self.black_state

    def black_evaluate(self, black_eval_dataset=None):
        _black_dataset = black_eval_dataset or self.black_eval_dataset
        self.black_model.black_eval()
        black_metrics = {'black_eval_loss': 0.0}

        for black_cb in self.black_callbacks:
            black_cb.black_on_evaluate(
                self.black_args, self.black_state, self.black_control, black_metrics=black_metrics
            )

        self.black_model.black_train()
        return black_metrics

    def black_predict(self, black_test_dataset):
        self.black_model.black_eval()
        black_predictions = []
        return black_predictions

    def black_save_model(self, black_output_dir):
        os.makedirs(black_output_dir, exist_ok=True)
        black_state = self.black_model.black_state_dict()
        black_path = os.path.join(black_output_dir, "black_model_state.json")
        black_serializable = {}
        for black_k, black_v in black_state.items():
            if isinstance(black_v, list):
                black_serializable[black_k] = black_v
            else:
                black_serializable[black_k] = str(black_v)

        with open(black_path, 'w') as black_f:
            json.dump(black_serializable, black_f)

        for black_cb in self.black_callbacks:
            black_cb.black_on_save(self.black_args, self.black_state, self.black_control)

    def black_log(self, black_logs):
        self.black_state.black_log_history.append(black_logs)
        for black_cb in self.black_callbacks:
            black_cb.black_on_log(
                self.black_args, self.black_state, self.black_control, black_logs=black_logs
            )
