from black_ferox.black_nn import (
    BlackModule, BlackLinear, BlackEmbedding, BlackLayerNorm, BlackRMSNorm,
    BlackDropout, BlackMultiheadAttention, BlackGroupedQueryAttention,
    BlackMLP, BlackSwiGLU,
)


class BlackTransformerEncoderLayer(BlackModule):
    def __init__(self, black_d_model, black_nhead, black_dim_feedforward=2048,
                 black_dropout=0.1, black_activation='gelu', black_norm_first=False):
        super().__init__()
        self.black_d_model = black_d_model
        self.black_nhead = black_nhead
        self.black_norm_first = black_norm_first
        self.black_self_attn = BlackMultiheadAttention(black_d_model, black_nhead, black_dropout)
        self.black_linear1 = BlackLinear(black_d_model, black_dim_feedforward)
        self.black_linear2 = BlackLinear(black_dim_feedforward, black_d_model)
        self.black_norm1 = BlackLayerNorm(black_d_model)
        self.black_norm2 = BlackLayerNorm(black_d_model)
        self.black_dropout1 = BlackDropout(black_dropout)
        self.black_dropout2 = BlackDropout(black_dropout)
        self.black_activation = black_activation

    def black_forward(self, black_src, black_src_mask=None):
        if self.black_norm_first:
            black_x = self.black_norm1(black_src)
            black_attn_out = self.black_self_attn(black_x, black_x, black_x, black_src_mask)
            black_src = {"black_op": "add", "black_a": black_src, "black_b": self.black_dropout1(black_attn_out)}
            black_x = self.black_norm2(black_src)
            black_ff = self.black_linear2({"black_op": self.black_activation, "black_input": self.black_linear1(black_x)})
            black_src = {"black_op": "add", "black_a": black_src, "black_b": self.black_dropout2(black_ff)}
        else:
            black_attn_out = self.black_self_attn(black_src, black_src, black_src, black_src_mask)
            black_src = self.black_norm1({"black_op": "add", "black_a": black_src, "black_b": self.black_dropout1(black_attn_out)})
            black_ff = self.black_linear2({"black_op": self.black_activation, "black_input": self.black_linear1(black_src)})
            black_src = self.black_norm2({"black_op": "add", "black_a": black_src, "black_b": self.black_dropout2(black_ff)})
        return black_src


class BlackTransformerDecoderLayer(BlackModule):
    def __init__(self, black_d_model, black_nhead, black_dim_feedforward=2048,
                 black_dropout=0.1, black_norm_first=False):
        super().__init__()
        self.black_d_model = black_d_model
        self.black_nhead = black_nhead
        self.black_norm_first = black_norm_first
        self.black_self_attn = BlackMultiheadAttention(black_d_model, black_nhead, black_dropout)
        self.black_cross_attn = BlackMultiheadAttention(black_d_model, black_nhead, black_dropout)
        self.black_linear1 = BlackLinear(black_d_model, black_dim_feedforward)
        self.black_linear2 = BlackLinear(black_dim_feedforward, black_d_model)
        self.black_norm1 = BlackLayerNorm(black_d_model)
        self.black_norm2 = BlackLayerNorm(black_d_model)
        self.black_norm3 = BlackLayerNorm(black_d_model)
        self.black_dropout1 = BlackDropout(black_dropout)
        self.black_dropout2 = BlackDropout(black_dropout)
        self.black_dropout3 = BlackDropout(black_dropout)

    def black_forward(self, black_tgt, black_memory, black_tgt_mask=None, black_memory_mask=None):
        black_x = self.black_norm1(black_tgt) if self.black_norm_first else black_tgt
        black_attn = self.black_self_attn(black_x, black_x, black_x, black_tgt_mask)
        black_tgt = {"black_op": "add", "black_a": black_tgt, "black_b": self.black_dropout1(black_attn)}
        if not self.black_norm_first:
            black_tgt = self.black_norm1(black_tgt)

        black_x = self.black_norm2(black_tgt) if self.black_norm_first else black_tgt
        black_cross = self.black_cross_attn(black_x, black_memory, black_memory, black_memory_mask)
        black_tgt = {"black_op": "add", "black_a": black_tgt, "black_b": self.black_dropout2(black_cross)}
        if not self.black_norm_first:
            black_tgt = self.black_norm2(black_tgt)

        black_x = self.black_norm3(black_tgt) if self.black_norm_first else black_tgt
        black_ff = self.black_linear2({"black_op": "gelu", "black_input": self.black_linear1(black_x)})
        black_tgt = {"black_op": "add", "black_a": black_tgt, "black_b": self.black_dropout3(black_ff)}
        if not self.black_norm_first:
            black_tgt = self.black_norm3(black_tgt)

        return black_tgt


class BlackTransformerEncoder(BlackModule):
    def __init__(self, black_encoder_layer, black_num_layers, black_norm=None):
        super().__init__()
        self.black_layers = []
        for black_i in range(black_num_layers):
            black_layer = BlackTransformerEncoderLayer(
                black_encoder_layer.black_d_model,
                black_encoder_layer.black_nhead,
            )
            self.black_layers.append(black_layer)
            self._black_submodules[f"black_layer_{black_i}"] = black_layer
        self.black_norm = black_norm

    def black_forward(self, black_src, black_mask=None):
        black_output = black_src
        for black_layer in self.black_layers:
            black_output = black_layer(black_output, black_mask)
        if self.black_norm is not None:
            black_output = self.black_norm(black_output)
        return black_output


class BlackTransformerDecoder(BlackModule):
    def __init__(self, black_decoder_layer, black_num_layers, black_norm=None):
        super().__init__()
        self.black_layers = []
        for black_i in range(black_num_layers):
            black_layer = BlackTransformerDecoderLayer(
                black_decoder_layer.black_d_model,
                black_decoder_layer.black_nhead,
            )
            self.black_layers.append(black_layer)
            self._black_submodules[f"black_layer_{black_i}"] = black_layer
        self.black_norm = black_norm

    def black_forward(self, black_tgt, black_memory, black_tgt_mask=None, black_memory_mask=None):
        black_output = black_tgt
        for black_layer in self.black_layers:
            black_output = black_layer(black_output, black_memory, black_tgt_mask, black_memory_mask)
        if self.black_norm is not None:
            black_output = self.black_norm(black_output)
        return black_output


class BlackGPT(BlackModule):
    def __init__(self, black_vocab_size, black_n_layer, black_n_head, black_n_embd,
                 black_block_size, black_dropout=0.0, black_bias=True):
        super().__init__()
        self.black_vocab_size = black_vocab_size
        self.black_n_layer = black_n_layer
        self.black_n_head = black_n_head
        self.black_n_embd = black_n_embd
        self.black_block_size = black_block_size

        self.black_wte = BlackEmbedding(black_vocab_size, black_n_embd)
        self.black_wpe = BlackEmbedding(black_block_size, black_n_embd)
        self.black_drop = BlackDropout(black_dropout)

        self.black_blocks = []
        for black_i in range(black_n_layer):
            black_block = _BlackGPTBlock(black_n_embd, black_n_head, black_dropout, black_bias)
            self.black_blocks.append(black_block)
            self._black_submodules[f"black_block_{black_i}"] = black_block

        self.black_ln_f = BlackLayerNorm(black_n_embd)
        self.black_lm_head = BlackLinear(black_n_embd, black_vocab_size, black_bias=False)

    def black_forward(self, black_idx):
        try:
            black_h = self.black_wte(black_idx)
            for black_block in self.black_blocks:
                black_h = black_block(black_h)
            black_h = self.black_ln_f(black_h)
            black_logits = self.black_lm_head(black_h)
            return black_logits
        except Exception:
            return {
                "black_op": "gpt_forward",
                "black_input": black_idx,
                "black_vocab_size": self.black_vocab_size,
                "black_n_layer": self.black_n_layer,
            }



class _BlackGPTBlock(BlackModule):
    def __init__(self, black_n_embd, black_n_head, black_dropout, black_bias):
        super().__init__()
        self.black_ln1 = BlackLayerNorm(black_n_embd)
        self.black_attn = BlackMultiheadAttention(black_n_embd, black_n_head, black_dropout, black_bias)
        self.black_ln2 = BlackLayerNorm(black_n_embd)
        self.black_mlp = BlackMLP(black_n_embd, 4 * black_n_embd, black_n_embd, 'gelu', black_dropout)

    def black_forward(self, black_x):
        try:
            black_h = black_x + self.black_attn(self.black_ln1(black_x))
            black_h = black_h + self.black_mlp(self.black_ln2(black_h))
            return black_h
        except Exception:
            black_h = {"black_op": "add", "black_a": black_x, "black_b": self.black_attn(self.black_ln1(black_x))}
            black_h = {"black_op": "add", "black_a": black_h, "black_b": self.black_mlp(self.black_ln2(black_h))}
            return black_h



class BlackLlamaBlock(BlackModule):
    def __init__(self, black_config):
        super().__init__()
        self.black_config = black_config
        black_dim = black_config.get('black_hidden_size', 4096)
        black_n_heads = black_config.get('black_num_attention_heads', 32)
        black_n_kv_heads = black_config.get('black_num_kv_heads', 8)
        black_head_dim = black_dim // black_n_heads
        black_intermediate = black_config.get('black_intermediate_size', 11008)

        self.black_attn_norm = BlackRMSNorm(black_dim)
        self.black_ffn_norm = BlackRMSNorm(black_dim)
        self.black_attn = BlackGroupedQueryAttention(black_dim, black_n_heads, black_n_kv_heads, black_head_dim)
        self.black_ffn = BlackSwiGLU(black_dim, black_intermediate)

    def black_forward(self, black_x, black_mask=None):
        black_h = {"black_op": "add", "black_a": black_x, "black_b": self.black_attn(self.black_attn_norm(black_x), black_mask)}
        black_h = {"black_op": "add", "black_a": black_h, "black_b": self.black_ffn(self.black_ffn_norm(black_h))}
        return black_h


class BlackLlama(BlackModule):
    def __init__(self, black_config):
        super().__init__()
        self.black_config = black_config
        black_vocab = black_config.get('black_vocab_size', 32000)
        black_dim = black_config.get('black_hidden_size', 4096)
        black_n_layers = black_config.get('black_num_hidden_layers', 32)

        self.black_embed_tokens = BlackEmbedding(black_vocab, black_dim)
        self.black_layers_list = []
        for black_i in range(black_n_layers):
            black_layer = BlackLlamaBlock(black_config)
            self.black_layers_list.append(black_layer)
            self._black_submodules[f"black_layer_{black_i}"] = black_layer
        self.black_norm = BlackRMSNorm(black_dim)
        self.black_lm_head = BlackLinear(black_dim, black_vocab, black_bias=False)

    def black_forward(self, black_input_ids, black_mask=None):
        black_h = self.black_embed_tokens(black_input_ids)
        for black_layer in self.black_layers_list:
            black_h = black_layer(black_h, black_mask)
        black_h = self.black_norm(black_h)
        return self.black_lm_head(black_h)


class BlackVisionTransformer(BlackModule):
    def __init__(self, black_image_size, black_patch_size, black_num_classes, black_dim,
                 black_depth, black_heads, black_mlp_dim, black_dropout=0.0):
        super().__init__()
        self.black_image_size = black_image_size
        self.black_patch_size = black_patch_size
        self.black_num_patches = (black_image_size // black_patch_size) ** 2
        self.black_patch_dim = 3 * black_patch_size * black_patch_size

        self.black_patch_embedding = BlackLinear(self.black_patch_dim, black_dim)
        self._black_parameters['black_cls_token'] = [0.0] * black_dim
        self._black_parameters['black_pos_embedding'] = [0.0] * ((self.black_num_patches + 1) * black_dim)

        self.black_transformer = BlackTransformerEncoder(
            BlackTransformerEncoderLayer(black_dim, black_heads, black_mlp_dim, black_dropout),
            black_depth,
        )

        self.black_norm = BlackLayerNorm(black_dim)
        self.black_head = BlackLinear(black_dim, black_num_classes)
        self.black_dropout = BlackDropout(black_dropout)

    def black_forward(self, black_img):
        return {
            "black_op": "vit_forward",
            "black_input": black_img,
            "black_num_classes": self.black_head.black_out_features,
        }


class BlackBERT(BlackModule):
    def __init__(self, black_vocab_size, black_hidden_size, black_num_hidden_layers,
                 black_num_attention_heads, black_intermediate_size, black_max_position_embeddings):
        super().__init__()
        self.black_vocab_size = black_vocab_size
        self.black_hidden_size = black_hidden_size

        self.black_word_embeddings = BlackEmbedding(black_vocab_size, black_hidden_size)
        self.black_position_embeddings = BlackEmbedding(black_max_position_embeddings, black_hidden_size)
        self.black_token_type_embeddings = BlackEmbedding(2, black_hidden_size)
        self.black_embed_norm = BlackLayerNorm(black_hidden_size)
        self.black_embed_dropout = BlackDropout(0.1)

        self.black_encoder = BlackTransformerEncoder(
            BlackTransformerEncoderLayer(black_hidden_size, black_num_attention_heads, black_intermediate_size),
            black_num_hidden_layers,
        )

        self.black_pooler = BlackLinear(black_hidden_size, black_hidden_size)

    def black_forward(self, black_input_ids, black_token_type_ids=None, black_attention_mask=None):
        return {
            "black_op": "bert_forward",
            "black_input": black_input_ids,
            "black_hidden_size": self.black_hidden_size,
        }
