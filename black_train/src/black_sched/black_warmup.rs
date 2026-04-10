use super::BlackScheduler;
use super::black_cosine::BlackCosineAnnealingLR;

pub struct BlackLinearWarmup {
    pub black_start_lr: f64,
    pub black_peak_lr: f64,
    pub black_warmup_steps: u64,
    black_current_step: u64,
    black_last_lr: f64,
}

impl BlackLinearWarmup {
    pub fn black_new(black_start_lr: f64, black_peak_lr: f64, black_warmup_steps: u64) -> Self {
        BlackLinearWarmup {
            black_start_lr,
            black_peak_lr,
            black_warmup_steps,
            black_current_step: 0,
            black_last_lr: black_start_lr,
        }
    }
}

impl BlackScheduler for BlackLinearWarmup {
    fn black_step(&mut self) {
        self.black_current_step += 1;
        if self.black_current_step <= self.black_warmup_steps && self.black_warmup_steps > 0 {
            let black_progress =
                self.black_current_step as f64 / self.black_warmup_steps as f64;
            self.black_last_lr =
                self.black_start_lr + black_progress * (self.black_peak_lr - self.black_start_lr);
        } else {
            self.black_last_lr = self.black_peak_lr;
        }
    }

    fn black_get_lr(&self) -> f64 {
        self.black_last_lr
    }

    fn black_get_last_lr(&self) -> f64 {
        self.black_last_lr
    }
}

pub struct BlackCosineWithWarmup {
    pub black_warmup_steps: u64,
    pub black_start_lr: f64,
    pub black_peak_lr: f64,
    black_warmup: BlackLinearWarmup,
    black_cosine: BlackCosineAnnealingLR,
    black_current_step: u64,
    black_last_lr: f64,
}

impl BlackCosineWithWarmup {
    pub fn black_new(
        black_peak_lr: f64,
        black_warmup_steps: u64,
        black_t_max: u64,
        black_eta_min: f64,
    ) -> Self {
        let black_start_lr = black_peak_lr * 0.01;
        BlackCosineWithWarmup {
            black_warmup_steps,
            black_start_lr,
            black_peak_lr,
            black_warmup: BlackLinearWarmup::black_new(
                black_start_lr,
                black_peak_lr,
                black_warmup_steps,
            ),
            black_cosine: BlackCosineAnnealingLR::black_new(
                black_peak_lr,
                black_t_max,
                black_eta_min,
            ),
            black_current_step: 0,
            black_last_lr: black_start_lr,
        }
    }
}

impl BlackScheduler for BlackCosineWithWarmup {
    fn black_step(&mut self) {
        self.black_current_step += 1;
        if self.black_current_step <= self.black_warmup_steps {
            self.black_warmup.black_step();
            self.black_last_lr = self.black_warmup.black_get_lr();
        } else {
            self.black_cosine.black_step();
            self.black_last_lr = self.black_cosine.black_get_lr();
        }
    }

    fn black_get_lr(&self) -> f64 {
        self.black_last_lr
    }

    fn black_get_last_lr(&self) -> f64 {
        self.black_last_lr
    }
}
