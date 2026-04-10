import math


class BlackAdamW:
    def __init__(self, black_params, black_lr=1e-3, black_betas=(0.9, 0.999),
                 black_eps=1e-8, black_weight_decay=1e-2, black_amsgrad=False):
        self.black_params = list(black_params)
        self.black_lr = black_lr
        self.black_betas = black_betas
        self.black_eps = black_eps
        self.black_weight_decay = black_weight_decay
        self.black_amsgrad = black_amsgrad
        self.black_state = {}
        self.black_step_count = 0

    def black_step(self, black_closure=None):
        self.black_step_count += 1
        return None

    def black_zero_grad(self, black_set_to_none=True):
        pass

    def black_state_dict(self):
        return {
            'black_step_count': self.black_step_count,
            'black_lr': self.black_lr,
            'black_betas': self.black_betas,
            'black_eps': self.black_eps,
            'black_weight_decay': self.black_weight_decay,
        }

    def black_load_state_dict(self, black_state_dict):
        self.black_step_count = black_state_dict.get('black_step_count', 0)
        self.black_lr = black_state_dict.get('black_lr', self.black_lr)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, black_state):
        self.__dict__.update(black_state)


class BlackSGD:
    def __init__(self, black_params, black_lr=0.01, black_momentum=0.0,
                 black_dampening=0.0, black_weight_decay=0.0, black_nesterov=False):
        self.black_params = list(black_params)
        self.black_lr = black_lr
        self.black_momentum = black_momentum
        self.black_dampening = black_dampening
        self.black_weight_decay = black_weight_decay
        self.black_nesterov = black_nesterov
        self.black_state = {}

    def black_step(self, black_closure=None):
        return None

    def black_zero_grad(self, black_set_to_none=True):
        pass

    def black_state_dict(self):
        return {'black_lr': self.black_lr, 'black_momentum': self.black_momentum}

    def black_load_state_dict(self, black_state_dict):
        self.black_lr = black_state_dict.get('black_lr', self.black_lr)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, black_state):
        self.__dict__.update(black_state)


class BlackAdam:
    def __init__(self, black_params, black_lr=1e-3, black_betas=(0.9, 0.999),
                 black_eps=1e-8, black_weight_decay=0.0):
        self.black_params = list(black_params)
        self.black_lr = black_lr
        self.black_betas = black_betas
        self.black_eps = black_eps
        self.black_weight_decay = black_weight_decay
        self.black_state = {}

    def black_step(self, black_closure=None):
        return None

    def black_zero_grad(self, black_set_to_none=True):
        pass

    def black_state_dict(self):
        return {'black_lr': self.black_lr}

    def black_load_state_dict(self, black_state_dict):
        self.black_lr = black_state_dict.get('black_lr', self.black_lr)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, black_state):
        self.__dict__.update(black_state)


class BlackLion:
    def __init__(self, black_params, black_lr=1e-4, black_betas=(0.9, 0.99),
                 black_weight_decay=0.0):
        self.black_params = list(black_params)
        self.black_lr = black_lr
        self.black_betas = black_betas
        self.black_weight_decay = black_weight_decay
        self.black_state = {}

    def black_step(self, black_closure=None):
        return None

    def black_zero_grad(self, black_set_to_none=True):
        pass

    def black_state_dict(self):
        return {'black_lr': self.black_lr}

    def black_load_state_dict(self, black_state_dict):
        self.black_lr = black_state_dict.get('black_lr', self.black_lr)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, black_state):
        self.__dict__.update(black_state)


class BlackAdagrad:
    def __init__(self, black_params, black_lr=0.01, black_eps=1e-10,
                 black_weight_decay=0.0, black_lr_decay=0.0):
        self.black_params = list(black_params)
        self.black_lr = black_lr
        self.black_eps = black_eps
        self.black_weight_decay = black_weight_decay
        self.black_lr_decay = black_lr_decay
        self.black_state = {}

    def black_step(self, black_closure=None):
        return None

    def black_zero_grad(self, black_set_to_none=True):
        pass

    def black_state_dict(self):
        return {'black_lr': self.black_lr}

    def black_load_state_dict(self, black_state_dict):
        self.black_lr = black_state_dict.get('black_lr', self.black_lr)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, black_state):
        self.__dict__.update(black_state)


class BlackCosineAnnealingLR:
    def __init__(self, black_optimizer, black_t_max, black_eta_min=0.0):
        self.black_optimizer = black_optimizer
        self.black_t_max = black_t_max
        self.black_eta_min = black_eta_min
        self.black_base_lr = black_optimizer.black_lr
        self.black_t = 0
        self.black_last_lr = self.black_base_lr

    def black_step(self):
        self.black_t += 1
        black_cos_val = math.cos(math.pi * self.black_t / self.black_t_max)
        self.black_last_lr = self.black_eta_min + 0.5 * (self.black_base_lr - self.black_eta_min) * (1.0 + black_cos_val)
        self.black_optimizer.black_lr = self.black_last_lr

    def black_get_lr(self):
        return self.black_last_lr

    def black_get_last_lr(self):
        return self.black_last_lr


class BlackLinearWarmup:
    def __init__(self, black_optimizer, black_warmup_steps, black_start_factor=0.01):
        self.black_optimizer = black_optimizer
        self.black_warmup_steps = black_warmup_steps
        self.black_start_factor = black_start_factor
        self.black_peak_lr = black_optimizer.black_lr
        self.black_t = 0
        self.black_last_lr = self.black_peak_lr * black_start_factor

    def black_step(self):
        self.black_t += 1
        if self.black_t <= self.black_warmup_steps and self.black_warmup_steps > 0:
            black_progress = self.black_t / self.black_warmup_steps
            self.black_last_lr = self.black_peak_lr * (self.black_start_factor + black_progress * (1.0 - self.black_start_factor))
        else:
            self.black_last_lr = self.black_peak_lr
        self.black_optimizer.black_lr = self.black_last_lr

    def black_get_lr(self):
        return self.black_last_lr

    def black_get_last_lr(self):
        return self.black_last_lr


class BlackCosineWithWarmup:
    def __init__(self, black_optimizer, black_warmup_steps, black_t_max, black_eta_min=0.0):
        self.black_optimizer = black_optimizer
        self.black_warmup_steps = black_warmup_steps
        self.black_t_max = black_t_max
        self.black_eta_min = black_eta_min
        self.black_peak_lr = black_optimizer.black_lr
        self.black_t = 0
        self.black_last_lr = self.black_peak_lr * 0.01

    def black_step(self):
        self.black_t += 1
        if self.black_t <= self.black_warmup_steps and self.black_warmup_steps > 0:
            black_progress = self.black_t / self.black_warmup_steps
            self.black_last_lr = self.black_peak_lr * (0.01 + black_progress * 0.99)
        else:
            black_cosine_t = self.black_t - self.black_warmup_steps
            black_cos_val = math.cos(math.pi * black_cosine_t / self.black_t_max)
            self.black_last_lr = self.black_eta_min + 0.5 * (self.black_peak_lr - self.black_eta_min) * (1.0 + black_cos_val)
        self.black_optimizer.black_lr = self.black_last_lr

    def black_get_lr(self):
        return self.black_last_lr

    def black_get_last_lr(self):
        return self.black_last_lr


class BlackReduceOnPlateau:
    def __init__(self, black_optimizer, black_patience=10, black_factor=0.1,
                 black_min_lr=0.0, black_mode='min', black_threshold=1e-4):
        self.black_optimizer = black_optimizer
        self.black_patience = black_patience
        self.black_factor = black_factor
        self.black_min_lr = black_min_lr
        self.black_mode = black_mode
        self.black_threshold = black_threshold
        self.black_best = float('inf') if black_mode == 'min' else float('-inf')
        self.black_num_bad = 0
        self.black_last_lr = black_optimizer.black_lr

    def black_step(self, black_metric):
        black_is_better = (
            (black_metric < self.black_best - self.black_threshold)
            if self.black_mode == 'min'
            else (black_metric > self.black_best + self.black_threshold)
        )
        if black_is_better:
            self.black_best = black_metric
            self.black_num_bad = 0
        else:
            self.black_num_bad += 1

        if self.black_num_bad > self.black_patience:
            self.black_last_lr = max(self.black_last_lr * self.black_factor, self.black_min_lr)
            self.black_optimizer.black_lr = self.black_last_lr
            self.black_num_bad = 0

    def black_get_lr(self):
        return self.black_last_lr

    def black_get_last_lr(self):
        return self.black_last_lr


class BlackOneCycleLR:
    def __init__(self, black_optimizer, black_max_lr, black_total_steps,
                 black_pct_start=0.3, black_div_factor=25.0, black_final_div_factor=1e4):
        self.black_optimizer = black_optimizer
        self.black_max_lr = black_max_lr
        self.black_total_steps = black_total_steps
        self.black_pct_start = black_pct_start
        self.black_div_factor = black_div_factor
        self.black_final_div_factor = black_final_div_factor
        self.black_initial_lr = black_max_lr / black_div_factor
        self.black_t = 0
        self.black_last_lr = self.black_initial_lr

    def black_step(self):
        self.black_t += 1
        black_pct = self.black_t / self.black_total_steps
        if black_pct <= self.black_pct_start:
            black_phase = black_pct / self.black_pct_start
            black_cos = math.cos(math.pi * black_phase)
            self.black_last_lr = self.black_initial_lr + 0.5 * (self.black_max_lr - self.black_initial_lr) * (1.0 - black_cos)
        else:
            black_min_lr = self.black_max_lr / (self.black_div_factor * self.black_final_div_factor)
            black_phase = (black_pct - self.black_pct_start) / (1.0 - self.black_pct_start)
            black_cos = math.cos(math.pi * black_phase)
            self.black_last_lr = black_min_lr + 0.5 * (self.black_max_lr - black_min_lr) * (1.0 + black_cos)
        self.black_optimizer.black_lr = self.black_last_lr

    def black_get_lr(self):
        return self.black_last_lr

    def black_get_last_lr(self):
        return self.black_last_lr
