import torch

def compute_lagged_pairwise_product(current_state, delayed_state, indices_left, indices_right, neuron_select_type):
    """
    Computes z_i(t) * z_j(t - delta).
    """
    # Select neurons based on pre-computed indices from the base CTM
    left = current_state[:, indices_left]   # z_i(t)
    right = delayed_state[:, indices_right] # z_j(t - delta)
    
    if neuron_select_type == 'random-pairing':
        return left * right
    
    elif neuron_select_type in ('first-last', 'random'):
        # For dense pairing, we compute the full outer product
        # but keep it flattened to match the expected sync vector size
        outer = left.unsqueeze(2) * right.unsqueeze(1)
        # Note: In lagged sync, the matrix is not necessarily symmetric,
        # so we don't take the upper triangle like the base CTM does.
        return outer.flatten(1)
    
    raise ValueError(f"Unsupported neuron_select_type: {neuron_select_type}")

def update_recursive_sync(pairwise_product, decay_alpha, decay_beta, r):
    """
    The linear recurrence update: alpha_t = r * alpha_{t-1} + product.
    """
    if decay_alpha is None or decay_beta is None:
        decay_alpha = pairwise_product
        decay_beta = torch.ones_like(pairwise_product)
    else:
        decay_alpha = r * decay_alpha + pairwise_product
        decay_beta = r * decay_beta + 1
        
    synchronization = decay_alpha / (torch.sqrt(decay_beta))
    return synchronization, decay_alpha, decay_beta