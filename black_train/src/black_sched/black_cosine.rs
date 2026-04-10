use super::BlackScheduler;

pub struct BlackCosineAnnealingLR {
    pub black_base_lr: f64,
    pub black_t_max: u64,
    pub black_eta_min: f64,
    black_t: u64,
    black_last_lr: f64,
}

impl BlackCosineAnnealingLR {
    pub fn black_new(black_base_lr: f64, black_t_max: u64, black_eta_min: f64) -> Self {
        BlackCosineAnnealingLR {
            black_base_lr,
            black_t_max,
            black_eta_min,
            black_t: 0,
            black_last_lr: black_base_lr,
        }
    }
}

impl BlackScheduler for BlackCosineAnnealingLR {
    fn black_step(&mut self) {
        self.black_t += 1;
        let black_pi = std::f64::consts::PI;
        let black_cos_val =
            (black_pi * self.black_t as f64 / self.black_t_max as f64).cos();
        self.black_last_lr = self.black_eta_min
            + 0.5 * (self.black_base_lr - self.black_eta_min) * (1.0 + black_cos_val);
    }

    fn black_get_lr(&self) -> f64 {
        self.black_last_lr
    }

    fn black_get_last_lr(&self) -> f64 {
        self.black_last_lr
    }
}
