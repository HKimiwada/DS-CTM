import torch

def compute_lagged_pairwise_product(current_state, delayed_state, indices_left, indices_right, neuron_select_type):
    """
    Computes z_i(t) * z_j(t - delta).
    """
    # Select neurons based on pre-computed indices from the base CTM
    left = current_state[:, indices_left]   # z_i(t)
    right = delayed_state[:, indices_right] # z_j(t - delta)
    
    if neuron_select_type == 'random-pairing':
        return left * right #
    
    elif neuron_select_type in ('first-last', 'random'):
        # For dense pairing, we compute the full outer product
        outer = left.unsqueeze(2) * right.unsqueeze(1)
        return outer.flatten(1) #
    
    raise ValueError(f"Unsupported neuron_select_type: {neuron_select_type}") #

def update_recursive_sync(pairwise_product, decay_alpha, decay_beta, r):
    """
    The linear recurrence update as described in CTM technical report.
    """
    if decay_alpha is None or decay_beta is None:
        decay_alpha = pairwise_product #
        decay_beta = torch.ones_like(pairwise_product) #
    else:
        decay_alpha = r * decay_alpha + pairwise_product #
        decay_beta = r * decay_beta + 1 #
        
    synchronization = decay_alpha / (torch.sqrt(decay_beta)) #
    return synchronization, decay_alpha, decay_beta #

if __name__ == "__main__":
    # Example usage
    current_state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    delayed_state = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
    indices_left = [0, 1]
    indices_right = [1, 0]
    neuron_select_type = 'random-pairing'
    
    pairwise_product = compute_lagged_pairwise_product(current_state, delayed_state, indices_left, indices_right, neuron_select_type)
    print("Pairwise Product:\n", pairwise_product)
    
    decay_alpha = None
    decay_beta = None
    r = 0.9
    
    synchronization, decay_alpha, decay_beta = update_recursive_sync(pairwise_product, decay_alpha, decay_beta, r)
    print("Synchronization:\n", synchronization)