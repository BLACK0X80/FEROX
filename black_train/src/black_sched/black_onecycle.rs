use super::BlackScheduler;

pub struct BlackOneCycleLR {
    pub black_max_lr: f64,
    pub black_total_steps: u64,
    pub black_pct_start: f64,
    pub black_div_factor: f64,
    pub black_final_div_factor: f64,
    black_current_step: u64,
    black_last_lr: f64,
}

impl BlackOneCycleLR {
    pub fn black_new(
        black_max_lr: f64,
        black_total_steps: u64,
        black_pct_start: f64,
        black_div_factor: f64,
        black_final_div_factor: f64,
    ) -> Self {
        let black_initial_lr = black_max_lr / black_div_factor;
        BlackOneCycleLR {
            black_max_lr,
            black_total_steps,
            black_pct_start,
            black_div_factor,
            black_final_div_factor,
            black_current_step: 0,
            black_last_lr: black_initial_lr,
        }
    }
}

impl BlackScheduler for BlackOneCycleLR {
    fn black_step(&mut self) {
        self.black_current_step += 1;
        let black_pct = self.black_current_step as f64 / self.black_total_steps as f64;
        let black_initial_lr = self.black_max_lr / self.black_div_factor;
        let black_min_lr = self.black_max_lr / (self.black_div_factor * self.black_final_div_factor);

        if black_pct <= self.black_pct_start {
            let black_phase_pct = black_pct / self.black_pct_start;
            let black_cos_val = (std::f64::consts::PI * black_phase_pct).cos();
            self.black_last_lr = black_initial_lr
                + 0.5 * (self.black_max_lr - black_initial_lr) * (1.0 - black_cos_val);
        } else {
            let black_phase_pct =
                (black_pct - self.black_pct_start) / (1.0 - self.black_pct_start);
            let black_cos_val = (std::f64::consts::PI * black_phase_pct).cos();
            self.black_last_lr =
                black_min_lr + 0.5 * (self.black_max_lr - black_min_lr) * (1.0 + black_cos_val);
        }
    }

    fn black_get_lr(&self) -> f64 {
        self.black_last_lr
    }

    fn black_get_last_lr(&self) -> f64 {
        self.black_last_lr
    }
}
