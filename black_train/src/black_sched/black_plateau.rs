use super::BlackScheduler;

pub struct BlackReduceOnPlateau {
    pub black_patience: u64,
    pub black_factor: f64,
    pub black_min_lr: f64,
    pub black_mode: BlackPlateauMode,
    pub black_threshold: f64,
    black_best: f64,
    black_num_bad_epochs: u64,
    black_last_lr: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlackPlateauMode {
    BlackMin,
    BlackMax,
}

impl BlackReduceOnPlateau {
    pub fn black_new(
        black_lr: f64,
        black_patience: u64,
        black_factor: f64,
        black_min_lr: f64,
        black_mode: BlackPlateauMode,
        black_threshold: f64,
    ) -> Self {
        let black_best = match black_mode {
            BlackPlateauMode::BlackMin => f64::INFINITY,
            BlackPlateauMode::BlackMax => f64::NEG_INFINITY,
        };

        BlackReduceOnPlateau {
            black_patience,
            black_factor,
            black_min_lr,
            black_mode,
            black_threshold,
            black_best,
            black_num_bad_epochs: 0,
            black_last_lr: black_lr,
        }
    }

    fn black_is_better(&self, black_metric: f64) -> bool {
        match self.black_mode {
            BlackPlateauMode::BlackMin => {
                black_metric < self.black_best - self.black_threshold
            }
            BlackPlateauMode::BlackMax => {
                black_metric > self.black_best + self.black_threshold
            }
        }
    }
}

impl BlackScheduler for BlackReduceOnPlateau {
    fn black_step(&mut self) {}

    fn black_step_with_metric(&mut self, black_metric: f64) {
        if self.black_is_better(black_metric) {
            self.black_best = black_metric;
            self.black_num_bad_epochs = 0;
        } else {
            self.black_num_bad_epochs += 1;
        }

        if self.black_num_bad_epochs > self.black_patience {
            let black_new_lr = self.black_last_lr * self.black_factor;
            self.black_last_lr = black_new_lr.max(self.black_min_lr);
            self.black_num_bad_epochs = 0;
        }
    }

    fn black_get_lr(&self) -> f64 {
        self.black_last_lr
    }

    fn black_get_last_lr(&self) -> f64 {
        self.black_last_lr
    }
}
