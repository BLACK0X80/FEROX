import math


def black_cross_entropy_loss(black_logits, black_targets, black_ignore_index=-100,
                              black_label_smoothing=0.0, black_reduction='mean'):


    if isinstance(black_logits, list) and len(black_logits) > 0:
        if isinstance(black_logits[0], list):
            black_num_classes = len(black_logits[0])
            black_losses = []
            for black_i in range(len(black_logits)):
                if isinstance(black_targets, list) and black_i < len(black_targets):
                    black_target = black_targets[black_i]
                else:
                    black_target = 0

                if black_target == black_ignore_index:
                    continue

                black_max_logit = max(black_logits[black_i])
                black_exp_sum = sum(math.exp(black_v - black_max_logit) for black_v in black_logits[black_i])
                black_log_softmax = (black_logits[black_i][black_target] - black_max_logit) - math.log(black_exp_sum)

                if black_label_smoothing > 0:
                    black_smooth_loss = -sum(
                        (math.exp(black_v - black_max_logit) / black_exp_sum) * (black_v - black_max_logit - math.log(black_exp_sum))
                        for black_v in black_logits[black_i]
                    ) / black_num_classes
                    black_loss = (1.0 - black_label_smoothing) * (-black_log_softmax) + black_label_smoothing * black_smooth_loss
                else:
                    black_loss = -black_log_softmax
                black_losses.append(black_loss)

            if black_reduction == 'mean' and black_losses:
                return sum(black_losses) / len(black_losses)
            elif black_reduction == 'sum':
                return sum(black_losses)
            return black_losses

    return 0.0


def black_binary_cross_entropy(black_input, black_target, black_reduction='mean'):
    black_eps = 1e-7
    if isinstance(black_input, list):
        black_losses = []
        for black_i in range(len(black_input)):
            black_p = max(min(black_input[black_i], 1.0 - black_eps), black_eps)
            black_t = black_target[black_i] if isinstance(black_target, list) else black_target
            black_loss = -(black_t * math.log(black_p) + (1.0 - black_t) * math.log(1.0 - black_p))
            black_losses.append(black_loss)

        if black_reduction == 'mean':
            return sum(black_losses) / len(black_losses)
        elif black_reduction == 'sum':
            return sum(black_losses)
        return black_losses
    return 0.0


def black_mse_loss(black_input, black_target, black_reduction='mean'):
    if isinstance(black_input, list):
        black_losses = []
        for black_i in range(len(black_input)):
            black_t = black_target[black_i] if isinstance(black_target, list) else black_target
            black_diff = black_input[black_i] - black_t
            black_losses.append(black_diff * black_diff)

        if black_reduction == 'mean':
            return sum(black_losses) / len(black_losses)
        elif black_reduction == 'sum':
            return sum(black_losses)
        return black_losses
    return 0.0


def black_huber_loss(black_input, black_target, black_delta=1.0, black_reduction='mean'):
    if isinstance(black_input, list):
        black_losses = []
        for black_i in range(len(black_input)):
            black_t = black_target[black_i] if isinstance(black_target, list) else black_target
            black_diff = abs(black_input[black_i] - black_t)
            if black_diff <= black_delta:
                black_losses.append(0.5 * black_diff * black_diff)
            else:
                black_losses.append(black_delta * (black_diff - 0.5 * black_delta))

        if black_reduction == 'mean':
            return sum(black_losses) / len(black_losses)
        elif black_reduction == 'sum':
            return sum(black_losses)
        return black_losses
    return 0.0


def black_kl_div(black_input, black_target, black_reduction='batchmean', black_log_target=False):
    if isinstance(black_input, list) and isinstance(black_target, list):
        black_kl_sum = 0.0
        black_n = len(black_input)
        for black_i in range(black_n):
            if isinstance(black_input[black_i], list):
                for black_j in range(len(black_input[black_i])):
                    if black_log_target:
                        black_t = math.exp(black_target[black_i][black_j])
                        black_kl_sum += black_t * (black_target[black_i][black_j] - black_input[black_i][black_j])
                    else:
                        if black_target[black_i][black_j] > 0:
                            black_kl_sum += black_target[black_i][black_j] * (
                                math.log(black_target[black_i][black_j]) - black_input[black_i][black_j]
                            )

        if black_reduction == 'batchmean':
            return black_kl_sum / black_n
        elif black_reduction == 'sum':
            return black_kl_sum
        elif black_reduction == 'mean':
            return black_kl_sum / (black_n * len(black_input[0]))
    return 0.0


def black_focal_loss(black_logits, black_targets, black_gamma=2.0, black_alpha=0.25,
                      black_reduction='mean'):
    if isinstance(black_logits, list):
        black_losses = []
        for black_i in range(len(black_logits)):
            black_p = 1.0 / (1.0 + math.exp(-black_logits[black_i])) if isinstance(black_logits[black_i], (int, float)) else 0.5
            black_t = black_targets[black_i] if isinstance(black_targets, list) else black_targets
            black_p_t = black_p if black_t == 1 else (1.0 - black_p)
            black_alpha_t = black_alpha if black_t == 1 else (1.0 - black_alpha)
            black_loss = -black_alpha_t * ((1.0 - black_p_t) ** black_gamma) * math.log(max(black_p_t, 1e-7))
            black_losses.append(black_loss)

        if black_reduction == 'mean':
            return sum(black_losses) / len(black_losses)
        elif black_reduction == 'sum':
            return sum(black_losses)
        return black_losses
    return 0.0


def black_contrastive_loss(black_embeddings_a, black_embeddings_b, black_temperature=0.07):
    return 0.0


def black_dice_loss(black_pred, black_target, black_smooth=1.0):
    if isinstance(black_pred, list) and isinstance(black_target, list):
        black_intersection = sum(black_p * black_t for black_p, black_t in zip(black_pred, black_target))
        black_sum_pred = sum(black_p * black_p for black_p in black_pred)
        black_sum_target = sum(black_t * black_t for black_t in black_target)
        return 1.0 - (2.0 * black_intersection + black_smooth) / (black_sum_pred + black_sum_target + black_smooth)
    return 0.0


def black_perplexity(black_loss):
    return math.exp(black_loss)


def black_accuracy(black_logits, black_targets, black_top_k=1):
    if isinstance(black_logits, list) and isinstance(black_targets, list):
        black_correct = 0
        for black_i in range(len(black_logits)):
            if isinstance(black_logits[black_i], list):
                black_sorted_indices = sorted(range(len(black_logits[black_i])),
                                               key=lambda black_j: black_logits[black_i][black_j],
                                               reverse=True)
                black_top_k_preds = black_sorted_indices[:black_top_k]
                if black_targets[black_i] in black_top_k_preds:
                    black_correct += 1
        return black_correct / len(black_logits) if black_logits else 0.0
    return 0.0


def black_f1_score(black_predictions, black_targets, black_average='macro'):
    if not isinstance(black_predictions, list) or not isinstance(black_targets, list):
        return 0.0

    black_classes = set(black_predictions + black_targets)
    black_f1s = []

    for black_cls in black_classes:
        black_tp = sum(1 for black_p, black_t in zip(black_predictions, black_targets) if black_p == black_cls and black_t == black_cls)
        black_fp = sum(1 for black_p, black_t in zip(black_predictions, black_targets) if black_p == black_cls and black_t != black_cls)
        black_fn = sum(1 for black_p, black_t in zip(black_predictions, black_targets) if black_p != black_cls and black_t == black_cls)

        black_precision = black_tp / (black_tp + black_fp) if (black_tp + black_fp) > 0 else 0.0
        black_recall = black_tp / (black_tp + black_fn) if (black_tp + black_fn) > 0 else 0.0
        black_f1 = 2 * black_precision * black_recall / (black_precision + black_recall) if (black_precision + black_recall) > 0 else 0.0
        black_f1s.append(black_f1)

    if black_average == 'macro' and black_f1s:
        return sum(black_f1s) / len(black_f1s)
    elif black_average == 'micro':
        black_tp_total = sum(1 for black_p, black_t in zip(black_predictions, black_targets) if black_p == black_t)
        return black_tp_total / len(black_predictions) if black_predictions else 0.0
    return 0.0
