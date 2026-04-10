import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def black_test_module_creation():
    from black_ferox.black_nn import BlackModule, BlackLinear, BlackSequential

    black_linear = BlackLinear(768, 512)
    assert black_linear.black_in_features == 768
    assert black_linear.black_out_features == 512
    assert 'black_weight' in black_linear._black_parameters
    assert 'black_bias' in black_linear._black_parameters
    assert len(black_linear._black_parameters['black_weight']) == 512
    assert len(black_linear._black_parameters['black_weight'][0]) == 768
    print("[PASS] black_test_module_creation")


def black_test_module_parameters():
    from black_ferox.black_nn import BlackLinear, BlackSequential

    black_seq = BlackSequential(
        BlackLinear(10, 20),
        BlackLinear(20, 5),
    )
    black_params = black_seq.black_parameters()
    assert len(black_params) == 4
    print("[PASS] black_test_module_parameters")


def black_test_module_train_eval():
    from black_ferox.black_nn import BlackLinear

    black_m = BlackLinear(10, 5)
    assert black_m.black_training is True
    black_m.black_eval()
    assert black_m.black_training is False
    black_m.black_train()
    assert black_m.black_training is True
    print("[PASS] black_test_module_train_eval")


def black_test_state_dict():
    from black_ferox.black_nn import BlackLinear

    black_m = BlackLinear(4, 2)
    black_state = black_m.black_state_dict()
    assert 'black_weight' in black_state
    assert 'black_bias' in black_state
    print("[PASS] black_test_state_dict")


def black_test_embedding():
    from black_ferox.black_nn import BlackEmbedding

    black_emb = BlackEmbedding(1000, 256)
    assert black_emb.black_num_embeddings == 1000
    assert black_emb.black_embedding_dim == 256
    assert len(black_emb._black_parameters['black_weight']) == 1000
    print("[PASS] black_test_embedding")


def black_test_layer_norm():
    from black_ferox.black_nn import BlackLayerNorm

    black_ln = BlackLayerNorm(512)
    assert black_ln.black_eps == 1e-5
    assert len(black_ln._black_parameters['black_weight']) == 512
    assert len(black_ln._black_parameters['black_bias']) == 512
    print("[PASS] black_test_layer_norm")


def black_test_rms_norm():
    from black_ferox.black_nn import BlackRMSNorm

    black_rn = BlackRMSNorm(256)
    assert black_rn.black_eps == 1e-8
    assert len(black_rn._black_parameters['black_weight']) == 256
    print("[PASS] black_test_rms_norm")


def black_test_dropout():
    from black_ferox.black_nn import BlackDropout

    black_drop = BlackDropout(0.3)
    assert black_drop.black_p == 0.3
    black_out = black_drop.black_forward("test_input")
    assert black_out is not None
    black_drop.black_eval()
    black_out2 = black_drop.black_forward("test_input")
    assert black_out2 == "test_input"
    print("[PASS] black_test_dropout")


def black_test_multihead_attention():
    from black_ferox.black_nn import BlackMultiheadAttention

    black_mha = BlackMultiheadAttention(512, 8, black_dropout=0.1)
    assert black_mha.black_embed_dim == 512
    assert black_mha.black_num_heads == 8
    assert black_mha.black_head_dim == 64
    print("[PASS] black_test_multihead_attention")


def black_test_gqa():
    from black_ferox.black_nn import BlackGroupedQueryAttention

    black_gqa = BlackGroupedQueryAttention(
        black_embed_dim=4096,
        black_num_heads=32,
        black_num_kv_heads=8,
        black_head_dim=128,
    )
    assert black_gqa.black_num_heads == 32
    assert black_gqa.black_num_kv_heads == 8
    print("[PASS] black_test_gqa")


def black_test_conv2d():
    from black_ferox.black_nn import BlackConv2d

    black_conv = BlackConv2d(3, 64, 3, black_padding=1)
    assert black_conv.black_in_channels == 3
    assert black_conv.black_out_channels == 64
    assert black_conv.black_kernel_size == (3, 3)
    print("[PASS] black_test_conv2d")


def black_test_mlp():
    from black_ferox.black_nn import BlackMLP

    black_mlp = BlackMLP(768, 3072, 768)
    black_params = black_mlp.black_parameters()
    assert len(black_params) == 4
    print("[PASS] black_test_mlp")


def black_test_swiglu():
    from black_ferox.black_nn import BlackSwiGLU

    black_sg = BlackSwiGLU(4096, 11008)
    black_params = black_sg.black_parameters()
    assert len(black_params) > 0
    print("[PASS] black_test_swiglu")


def black_test_gpt():
    from black_ferox.black_nn.black_transformers import BlackGPT

    black_gpt = BlackGPT(
        black_vocab_size=50257,
        black_n_layer=2,
        black_n_head=4,
        black_n_embd=128,
        black_block_size=64,
    )
    assert black_gpt.black_vocab_size == 50257
    assert len(black_gpt.black_blocks) == 2
    print("[PASS] black_test_gpt")


def black_test_llama():
    from black_ferox.black_nn.black_transformers import BlackLlama

    black_config = {
        'black_vocab_size': 32000,
        'black_hidden_size': 256,
        'black_num_hidden_layers': 2,
        'black_num_attention_heads': 4,
        'black_num_kv_heads': 2,
        'black_intermediate_size': 512,
    }
    black_llama = BlackLlama(black_config)
    assert len(black_llama.black_layers_list) == 2
    print("[PASS] black_test_llama")


def black_test_vit():
    from black_ferox.black_nn.black_transformers import BlackVisionTransformer

    black_vit = BlackVisionTransformer(
        black_image_size=224,
        black_patch_size=16,
        black_num_classes=1000,
        black_dim=768,
        black_depth=2,
        black_heads=12,
        black_mlp_dim=3072,
    )
    assert black_vit.black_num_patches == 196
    print("[PASS] black_test_vit")


def black_test_bert():
    from black_ferox.black_nn.black_transformers import BlackBERT

    black_bert = BlackBERT(
        black_vocab_size=30522,
        black_hidden_size=256,
        black_num_hidden_layers=2,
        black_num_attention_heads=4,
        black_intermediate_size=512,
        black_max_position_embeddings=512,
    )
    assert black_bert.black_vocab_size == 30522
    print("[PASS] black_test_bert")


def black_test_optimizers():
    from black_ferox.black_nn import BlackLinear
    from black_ferox.black_optim import BlackAdamW, BlackSGD, BlackAdam, BlackLion, BlackAdagrad

    black_model = BlackLinear(10, 5)

    black_adamw = BlackAdamW(black_model.black_parameters(), black_lr=1e-3)
    assert black_adamw.black_lr == 1e-3
    black_adamw.black_step()
    black_adamw.black_zero_grad()

    black_sgd = BlackSGD(black_model.black_parameters(), black_lr=0.01, black_momentum=0.9)
    assert black_sgd.black_lr == 0.01

    black_adam = BlackAdam(black_model.black_parameters())
    black_lion = BlackLion(black_model.black_parameters())
    black_adagrad = BlackAdagrad(black_model.black_parameters())

    for black_opt in [black_adamw, black_sgd, black_adam, black_lion, black_adagrad]:
        black_sd = black_opt.black_state_dict()
        assert isinstance(black_sd, dict)

    print("[PASS] black_test_optimizers")


def black_test_schedulers():
    from black_ferox.black_optim import (
        BlackAdamW, BlackCosineAnnealingLR, BlackLinearWarmup,
        BlackCosineWithWarmup, BlackReduceOnPlateau, BlackOneCycleLR,
    )
    from black_ferox.black_nn import BlackLinear

    black_model = BlackLinear(10, 5)
    black_opt = BlackAdamW(black_model.black_parameters(), black_lr=0.01)

    black_cosine = BlackCosineAnnealingLR(black_opt, black_t_max=100)
    for _ in range(10):
        black_cosine.black_step()
    assert black_cosine.black_get_lr() < 0.01

    black_opt2 = BlackAdamW(black_model.black_parameters(), black_lr=0.01)
    black_warmup = BlackLinearWarmup(black_opt2, black_warmup_steps=10)
    for _ in range(5):
        black_warmup.black_step()
    assert black_warmup.black_get_lr() < 0.01

    black_opt3 = BlackAdamW(black_model.black_parameters(), black_lr=0.01)
    black_cwu = BlackCosineWithWarmup(black_opt3, black_warmup_steps=10, black_t_max=100)
    for _ in range(20):
        black_cwu.black_step()

    black_opt4 = BlackAdamW(black_model.black_parameters(), black_lr=0.01)
    black_plateau = BlackReduceOnPlateau(black_opt4, black_patience=3)
    for _ in range(10):
        black_plateau.black_step(1.0)

    black_opt5 = BlackAdamW(black_model.black_parameters(), black_lr=0.01)
    black_oc = BlackOneCycleLR(black_opt5, black_max_lr=0.01, black_total_steps=100)
    for _ in range(50):
        black_oc.black_step()

    print("[PASS] black_test_schedulers")


def black_test_dataset():
    from black_ferox.black_data import BlackTensorDataset, BlackDataLoader

    black_data = list(range(100))
    black_labels = list(range(100))
    black_ds = BlackTensorDataset(black_data, black_labels)
    assert len(black_ds) == 100
    assert black_ds[0] == (0, 0)

    black_dl = BlackDataLoader(black_ds, black_batch_size=10, black_shuffle=False)
    black_batches = list(black_dl)
    assert len(black_batches) == 10
    print("[PASS] black_test_dataset")


def black_test_dataloader_shuffle():
    from black_ferox.black_data import BlackTensorDataset, BlackDataLoader

    black_data = list(range(50))
    black_ds = BlackTensorDataset(black_data)
    black_dl = BlackDataLoader(black_ds, black_batch_size=10, black_shuffle=True)
    black_batches = list(black_dl)
    assert len(black_batches) == 5
    print("[PASS] black_test_dataloader_shuffle")


def black_test_dataloader_drop_last():
    from black_ferox.black_data import BlackTensorDataset, BlackDataLoader

    black_data = list(range(25))
    black_ds = BlackTensorDataset(black_data)
    black_dl = BlackDataLoader(black_ds, black_batch_size=10, black_drop_last=True)
    black_batches = list(black_dl)
    assert len(black_batches) == 2
    print("[PASS] black_test_dataloader_drop_last")


def black_test_samplers():
    from black_ferox.black_data import (
        BlackTensorDataset, BlackRandomSampler, BlackSequentialSampler, BlackDistributedSampler,
    )

    black_data = list(range(100))
    black_ds = BlackTensorDataset(black_data)

    black_rs = BlackRandomSampler(black_ds)
    black_random_indices = list(black_rs)
    assert len(black_random_indices) == 100

    black_ss = BlackSequentialSampler(black_ds)
    black_seq_indices = list(black_ss)
    assert black_seq_indices == list(range(100))

    black_dist = BlackDistributedSampler(black_ds, black_num_replicas=4, black_rank=0)
    black_dist_indices = list(black_dist)
    assert len(black_dist_indices) == 25

    print("[PASS] black_test_samplers")


def black_test_trainer():
    from black_ferox.black_nn import BlackLinear
    from black_ferox.black_train import BlackTrainer, BlackTrainingArguments
    from black_ferox.black_data import BlackTensorDataset

    black_model = BlackLinear(10, 5)
    black_data = list(range(20))
    black_ds = BlackTensorDataset(black_data)

    black_args = BlackTrainingArguments(
        black_output_dir="./black_test_output",
        black_num_train_epochs=1,
        black_per_device_train_batch_size=4,
        black_logging_steps=5,
        black_save_steps=0,
        black_eval_steps=0,
    )

    black_trainer = BlackTrainer(
        black_model=black_model,
        black_args=black_args,
        black_train_dataset=black_ds,
    )

    black_state = black_trainer.black_train()
    assert black_state.black_global_step > 0
    print("[PASS] black_test_trainer")


def black_test_callbacks():
    from black_ferox.black_train import (
        BlackTrainerCallback, BlackProgressCallback,
        BlackEarlyStoppingCallback, BlackTrainerState, BlackTrainerControl,
        BlackTrainingArguments,
    )

    black_cb = BlackTrainerCallback()
    black_args = BlackTrainingArguments()
    black_state = BlackTrainerState()
    black_control = BlackTrainerControl()

    black_cb.black_on_train_begin(black_args, black_state, black_control)
    black_cb.black_on_step_end(black_args, black_state, black_control)
    black_cb.black_on_train_end(black_args, black_state, black_control)

    black_es = BlackEarlyStoppingCallback(black_early_stopping_patience=2)
    black_es.black_on_evaluate(black_args, black_state, black_control, black_metrics={'black_eval_loss': 1.0})
    assert black_control.black_should_training_stop is False
    black_es.black_on_evaluate(black_args, black_state, black_control, black_metrics={'black_eval_loss': 1.1})
    black_es.black_on_evaluate(black_args, black_state, black_control, black_metrics={'black_eval_loss': 1.2})
    assert black_control.black_should_training_stop is True

    print("[PASS] black_test_callbacks")


def black_test_metrics():
    from black_ferox.black_metrics import (
        black_cross_entropy_loss, black_mse_loss, black_binary_cross_entropy,
        black_huber_loss, black_perplexity, black_accuracy, black_f1_score,
        black_dice_loss, black_focal_loss,
    )

    black_logits = [[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]]
    black_targets = [0, 1]
    black_ce = black_cross_entropy_loss(black_logits, black_targets)
    assert isinstance(black_ce, float)
    assert black_ce > 0

    black_pred = [1.0, 2.0, 3.0]
    black_tgt = [1.1, 2.1, 2.9]
    black_mse = black_mse_loss(black_pred, black_tgt)
    assert isinstance(black_mse, float)

    black_ppl = black_perplexity(2.0)
    assert abs(black_ppl - 7.389) < 0.1

    black_acc = black_accuracy([[0.1, 0.9], [0.8, 0.2]], [1, 0])
    assert black_acc == 1.0

    black_f1 = black_f1_score([0, 1, 1, 0], [0, 1, 0, 0])
    assert isinstance(black_f1, float)

    print("[PASS] black_test_metrics")


def black_test_export():
    from black_ferox.black_nn import BlackLinear
    from black_ferox.black_export import BlackModelExporter
    import tempfile
    import os

    black_model = BlackLinear(10, 5)
    black_exporter = BlackModelExporter()

    black_tmp = tempfile.mkdtemp()
    black_path = os.path.join(black_tmp, "black_model.json")
    black_exporter.black_to_onnx(black_model, None, black_path)
    assert os.path.exists(black_path)

    black_st_path = os.path.join(black_tmp, "black_model.safetensors")
    black_exporter.black_to_safetensors(black_model, black_st_path)
    assert os.path.exists(black_st_path)

    print("[PASS] black_test_export")


def black_test_pickle_compat():
    import pickle
    from black_ferox.black_nn import BlackLinear
    from black_ferox.black_optim import BlackAdamW
    from black_ferox.black_train import BlackTrainingArguments

    black_linear = BlackLinear(10, 5)
    black_pickled = pickle.dumps(black_linear)
    black_restored = pickle.loads(black_pickled)
    assert black_restored.black_in_features == 10

    black_args = BlackTrainingArguments(black_output_dir="./test")
    black_pickled_args = pickle.dumps(black_args)
    black_restored_args = pickle.loads(black_pickled_args)
    assert black_restored_args.black_output_dir == "./test"

    black_opt = BlackAdamW(black_linear.black_parameters(), black_lr=1e-3)
    black_pickled_opt = pickle.dumps(black_opt)
    black_restored_opt = pickle.loads(black_pickled_opt)
    assert black_restored_opt.black_lr == 1e-3

    print("[PASS] black_test_pickle_compat")


if __name__ == "__main__":
    black_test_module_creation()
    black_test_module_parameters()
    black_test_module_train_eval()
    black_test_state_dict()
    black_test_embedding()
    black_test_layer_norm()
    black_test_rms_norm()
    black_test_dropout()
    black_test_multihead_attention()
    black_test_gqa()
    black_test_conv2d()
    black_test_mlp()
    black_test_swiglu()
    black_test_gpt()
    black_test_llama()
    black_test_vit()
    black_test_bert()
    black_test_optimizers()
    black_test_schedulers()
    black_test_dataset()
    black_test_dataloader_shuffle()
    black_test_dataloader_drop_last()
    black_test_samplers()
    black_test_trainer()
    black_test_callbacks()
    black_test_metrics()
    black_test_export()
    black_test_pickle_compat()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
