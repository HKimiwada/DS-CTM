import argparse
import math
import os
import random
import torch
import numpy as np
import wandb
from tqdm.auto import tqdm

# Original repo imports
from autoclip.torch import QuantileClip
from data.custom_datasets import ParityDataset
from models.lagged_ctm import LaggedContinuousThoughtMachine
from utils.losses import parity_loss
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup
from utils.housekeeping import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train Lagged-CTM on Parity Task")

    # --- Lagged CTM Specific ---
    parser.add_argument('--lags', type=int, nargs='+', default=[0, 1, 2, 4], 
                        help='Temporal offsets (Delta) for directional synchronization.')
    parser.add_argument('--wandb_project', type=str, default='lagged-ctm-research')
    
    # --- Standard CTM Architecture (Aligned with baseline) ---
    parser.add_argument('--parity_sequence_length', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--d_input', type=int, default=512)
    parser.add_argument('--synapse_depth', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--n_synch_out', type=int, default=32)
    parser.add_argument('--n_synch_action', type=int, default=32)
    parser.add_argument('--neuron_select_type', type=str, default='random', choices=['first-last', 'random', 'random-pairing'])
    parser.add_argument('--iterations', type=int, default=75, help='Thought ticks (T)')
    parser.add_argument('--memory_length', type=int, default=25, help='NLM history (M)')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--memory_hidden_dims', type=int, default=16)

    # --- Training Configuration ---
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--training_iterations', type=int, default=50001)
    parser.add_argument('--gradient_clipping', type=float, default=0.1, help='Quantile clipping value')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--track_every', type=int, default=1000)
    
    return parser.parse_args()

def train():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Initialize WandB
    wandb.init(project=args.wandb_project, config=args)
    config = wandb.config

    # 2. Setup Data
    train_data = ParityDataset(sequence_length=config.parity_sequence_length, length=100000)
    test_data = ParityDataset(sequence_length=config.parity_sequence_length, length=10000)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    # 3. Build Lagged Model
    # We instantiate the Lagged version directly to handle the 'lags' argument
    model = LaggedContinuousThoughtMachine(
        iterations=config.iterations,
        d_model=config.d_model,
        d_input=config.d_input,
        heads=config.heads,
        n_synch_out=config.n_synch_out,
        n_synch_action=config.n_synch_action,
        synapse_depth=config.synapse_depth,
        memory_length=config.memory_length,
        deep_nlms=config.deep_memory,
        memory_hidden_dims=config.memory_hidden_dims,
        do_layernorm_nlm=False,
        backbone_type='parity_backbone',
        positional_embedding_type='custom-rotational-1d',
        out_dims=config.parity_sequence_length * 2,
        neuron_select_type=config.neuron_select_type,
        lags=config.lags
    ).to(device)

    # 4. Optimizer & Stability (QuantileClip as per original repo)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if args.gradient_clipping != -1:
        optimizer = QuantileClip.as_optimizer(optimizer=optimizer, quantile=args.gradient_clipping, history_length=1000)
    
    scheduler = WarmupMultiStepLR(optimizer, warmup_steps=0, milestones=[8000, 15000, 20000], gamma=0.1)

    # 5. Training Loop
    iterator = iter(trainloader)
    pbar = tqdm(range(config.training_iterations), desc="Training Lagged-CTM")
    
    for bi in pbar:
        model.train()
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(trainloader)
            inputs, targets = next(iterator)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        predictions, certainties, _ = model(inputs)
        
        # Reshape for Parity Loss
        # Expected: [B, seq_len, 2, iterations]
        predictions = predictions.reshape(predictions.size(0), -1, 2, predictions.size(-1))
        loss, where_certain = parity_loss(predictions, certainties, targets)

        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 6. Logging
        if bi % args.log_every == 0:
            # Accuracy at the step where the model is most certain
            batch_idx = torch.arange(predictions.size(0), device=device)
            preds_at_certainty = predictions.argmax(2)[batch_idx, :, where_certain]
            accuracy = (preds_at_certainty == targets).float().mean()

            # Directional Diagnostics: Track decay parameters for each lag
            lag_stats = {}
            for lag in config.lags:
                # Get the mean 'r' value (decay rate) for this lag
                decay_val = model.decay_params_out[f"lag_{lag}"].detach().mean().item()
                lag_stats[f"decays/lag_{lag}"] = math.exp(-decay_val) # Actual decay coefficient

            wandb.log({
                "iteration": bi,
                "train/loss": loss.item(),
                "train/accuracy": accuracy.item(),
                "train/avg_certainty_step": where_certain.float().mean().item(),
                "train/min_certainty_step": where_certain.min().item(),
                "train/max_certainty_step": where_certain.max().item(),
                "lr": optimizer.param_groups[0]['lr'],
                **lag_stats
            })

            pbar.set_description(f"Loss: {loss.item():.4f} Acc: {accuracy.item():.4f}")

        # 7. Periodic Evaluation
        if bi % args.track_every == 0:
            model.eval()
            test_accs = []
            with torch.no_grad():
                for test_inputs, test_targets in testloader:
                    test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                    t_preds, t_certs, _ = model(test_inputs)
                    t_preds = t_preds.reshape(t_preds.size(0), -1, 2, t_preds.size(-1))
                    _, t_where = parity_loss(t_preds, t_certs, test_targets)
                    
                    t_batch_idx = torch.arange(t_preds.size(0), device=device)
                    acc = (t_preds.argmax(2)[t_batch_idx, :, t_where] == test_targets).float().mean()
                    test_accs.append(acc.item())
            
            wandb.log({"test/accuracy": np.mean(test_accs)})

    wandb.finish()

if __name__ == "__main__":
    train()