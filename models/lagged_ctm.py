import torch
import torch.nn as nn
import numpy as np
from models.ctm import ContinuousThoughtMachine
from utils.sync_metrics import compute_lagged_pairwise_product, update_recursive_sync

class LaggedContinuousThoughtMachine(ContinuousThoughtMachine):
    def __init__(self, *args, lags=[0, 1, 2, 4], **kwargs):
        self.lags = sorted(lags)
        # super().__init__ will now use our overridden calculate_synch_representation_size
        super().__init__(*args, **kwargs)
        
        # 1. Fix Parameter Collision
        # We replace the base class's single Parameter with our Lagged ParameterDict
        if hasattr(self, 'decay_params_action'): 
            delattr(self, 'decay_params_action')
        if hasattr(self, 'decay_params_out'): 
            delattr(self, 'decay_params_out')
        
        self._setup_lagged_parameters()

    def calculate_synch_representation_size(self, n_synch):
        """
        Overridden to support asymmetric (directional) synchrony.
        Returns n_synch * n_synch (Full matrix) instead of base CTM's n_synch*(n_synch+1)//2.
        """
        if self.neuron_select_type == 'random-pairing':
            return n_synch
        elif self.neuron_select_type in ('first-last', 'random'):
            # Lagged synchrony is asymmetric; we need the full directional matrix
            return n_synch * n_synch
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")

    def _setup_lagged_parameters(self):
        """Register separate decay parameters for each temporal offset."""
        # Now uses the correctly computed (larger) representation size
        if self.synch_representation_size_action > 0:
            self.decay_params_action = nn.ParameterDict({
                f"lag_{l}": nn.Parameter(torch.zeros(self.synch_representation_size_action))
                for l in self.lags
            })
        
        self.decay_params_out = nn.ParameterDict({
            f"lag_{l}": nn.Parameter(torch.zeros(self.synch_representation_size_out))
            for l in self.lags
        })

    def compute_lagged_sync(self, current_z, history, decay_dict, alpha_dict, beta_dict, synch_type):
        indices_left = getattr(self, f'{synch_type}_neuron_indices_left')
        indices_right = getattr(self, f'{synch_type}_neuron_indices_right')
        
        all_syncs = []
        for lag in self.lags:
            hist_idx = max(0, len(history) - 1 - lag)
            delayed_z = history[hist_idx]
            
            # prod is now correctly N*N (1024)
            prod = compute_lagged_pairwise_product(
                current_z, delayed_z, indices_left, indices_right, self.neuron_select_type
            )
            
            # r is now correctly N*N (1024)
            r = torch.exp(-torch.clamp(decay_dict[f"lag_{lag}"], 0, 15)).unsqueeze(0)
            sync, a, b = update_recursive_sync(
                prod, alpha_dict.get(lag), beta_dict.get(lag), r
            )
            
            alpha_dict[lag], beta_dict[lag] = a, b
            all_syncs.append(sync)
            
        return torch.cat(all_syncs, dim=-1)

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        pre_activations_tracking, post_activations_tracking = [], [] #
        synch_out_tracking, synch_action_tracking, attention_tracking = [], [], []

        kv = self.compute_features(x) #
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) #
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) #
        
        activation_history = [activated_state] #
        alphas_action, betas_action, alphas_out, betas_out = {}, {}, {}, {}

        predictions = torch.empty(B, self.out_dims, self.iterations, device=device) #
        certainties = torch.empty(B, 2, self.iterations, device=device) #

        for stepi in range(self.iterations):
            sync_action = self.compute_lagged_sync(activated_state, activation_history, 
                                                   self.decay_params_action, alphas_action, betas_action, 'action')

            q = self.q_proj(sync_action).unsqueeze(1) #
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False) #
            attn_out = attn_out.squeeze(1)
            
            pre_synapse_input = torch.cat((attn_out, activated_state), dim=-1) #
            state = self.synapses(pre_synapse_input) #
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1) #
            activated_state = self.trace_processor(state_trace) #
            
            activation_history.append(activated_state) #
            if len(activation_history) > max(self.lags) + 1:
                activation_history.pop(0)

            sync_out = self.compute_lagged_sync(activated_state, activation_history,
                                                self.decay_params_out, alphas_out, betas_out, 'out')

            current_pred = self.output_projector(sync_out) #
            predictions[..., stepi] = current_pred
            certainties[..., stepi] = self.compute_certainty(current_pred)

            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(sync_out.detach().cpu().numpy())
                synch_action_tracking.append(sync_action.detach().cpu().numpy())

        if track:
            return (predictions, certainties, 
                    (np.array(synch_out_tracking), np.array(synch_action_tracking)), 
                    np.array(pre_activations_tracking), 
                    np.array(post_activations_tracking), 
                    np.array(attention_tracking)) #
        
        return predictions, certainties, sync_out #