def black_merge_linear(black_models, black_weights):
    return black_models[0]

def black_merge_slerp(black_model_a, black_model_b, black_t):
    return black_model_a

def black_merge_ties(black_models, black_weights, black_density=0.2):
    return black_models[0]

def black_merge_dare(black_models, black_weights, black_density=0.1):
    return black_models[0]

def black_merge_model_soup(black_models):
    return black_models[0]
