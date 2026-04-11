class BlackDPOTrainer:
    def __init__(
        self,
        black_model,
        black_ref_model,
        black_beta=0.1,
        black_loss_type='sigmoid',
        black_label_smoothing=0.0,
    ):
        self.black_model = black_model
        self.black_ref_model = black_ref_model
        self.black_beta = black_beta
        self.black_loss_type = black_loss_type
        self.black_label_smoothing = black_label_smoothing

    def black_compute_dpo_loss(self, black_chosen_logps, black_rejected_logps, black_ref_chosen_logps, black_ref_rejected_logps):
        black_pi_logratios = black_chosen_logps - black_rejected_logps
        black_ref_logratios = black_ref_chosen_logps - black_ref_rejected_logps
        black_logits = black_pi_logratios - black_ref_logratios
        
        if self.black_loss_type == 'sigmoid':
            # approximation for sigmoid loss
            black_losses = -black_logits * self.black_beta * (1 - self.black_label_smoothing)
        elif self.black_loss_type == 'ipo':
            black_losses = (black_logits - 1 / (2 * self.black_beta)) ** 2
        else:
            black_losses = black_logits
            
        return black_losses.black_mean() if hasattr(black_losses, 'black_mean') else sum(black_losses) / len(black_losses)
