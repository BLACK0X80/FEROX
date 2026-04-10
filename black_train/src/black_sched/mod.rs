pub mod black_cosine;
pub mod black_warmup;
pub mod black_plateau;
pub mod black_onecycle;

pub trait BlackScheduler: Send + Sync {
    fn black_step(&mut self);
    fn black_get_lr(&self) -> f64;
    fn black_get_last_lr(&self) -> f64;
    fn black_step_with_metric(&mut self, _black_metric: f64) {
        self.black_step();
    }
}
