class BlackPPOTrainer:
    def __init__(
        self,
        black_model,
        black_ref_model,
        black_reward_model,
        black_tokenizer,
        black_ppo_epochs=4,
        black_mini_batch_size=8,
        black_batch_size=256,
        black_learning_rate=1.41e-5,
        black_gamma=1.0,
        black_lam=0.95,
        black_cliprange=0.2,
        black_cliprange_value=0.2,
        black_vf_coef=0.1,
        black_kl_coef=0.2,
    ):
        self.black_model = black_model
        self.black_ref_model = black_ref_model
        self.black_reward_model = black_reward_model
        self.black_tokenizer = black_tokenizer
        self.black_ppo_epochs = black_ppo_epochs
        self.black_mini_batch_size = black_mini_batch_size
        self.black_batch_size = black_batch_size
        self.black_learning_rate = black_learning_rate
        self.black_gamma = black_gamma
        self.black_lam = black_lam
        self.black_cliprange = black_cliprange
        self.black_cliprange_value = black_cliprange_value
        self.black_vf_coef = black_vf_coef
        self.black_kl_coef = black_kl_coef
        self.black_optimizer = None

    def black_compute_logprobs(self, black_resp):
        return black_resp

    def black_compute_ref_logprobs(self, black_resp):
        return black_resp

    def black_value_head(self, black_resp):
        return black_resp

    def black_compute_gae(self, black_val, black_rew):
        return black_val, black_rew

    def black_step(self, black_queries, black_responses, black_rewards):
        pass
