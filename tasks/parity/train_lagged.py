import argparse
import math
import os
import torch
import numpy as np
import wandb
from tqdm.auto import tqdm

from autoclip.torch import QuantileClip
from data.custom_datasets import ParityDataset
from models.lagged_ctm import LaggedContinuousThoughtMachine
from utils.losses import parity_loss
from utils.schedulers import WarmupMultiStepLR
from utils.housekeeping import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train Lagged-CTM on Parity Task")
    # ... (same arguments as before) ...
    parser.add_argument('--lags', type=int, nargs='+', default=[0, 1, 2, 4])
    parser.add_argument('--wandb_project', type=str, default='lagged-ctm-research')
    parser.add_argument('--parity_sequence_length', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--d_input', type=int, default=512)
    parser.add_argument('--synapse_depth', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--n_synch_out', type=int, default=32)
    parser.add_argument('--n_synch_action', type=int, default=32)
    parser.add_argument('--neuron_select_type', type=str, default='random')
    parser.add_argument('--iterations', type=int, default=75)
    parser.add_argument('--memory_length', type=int, default=25)
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--memory_hidden_dims', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--training_iterations', type=int, default=50001)
    parser.add_argument('--gradient_clipping', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--track_every', type=int, default=1000)
    return parser.parse_args()

def train():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wandb.init(project=args.wandb_project, config=args)
    config = wandb.config

    train_data = ParityDataset(sequence_length=config.parity_sequence_length, length=100000)
    test_data = ParityDataset(sequence_length=config.parity_sequence_length, length=10000)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    model = LaggedContinuousThoughtMachine(
        iterations=config.iterations, d_model=config.d_model, d_input=config.d_input,
        heads=config.heads, n_synch_out=config.n_synch_out, n_synch_action=config.n_synch_action,
        synapse_depth=config.synapse_depth, memory_length=config.memory_length,
        deep_nlms=config.deep_memory, memory_hidden_dims=config.memory_hidden_dims,
        do_layernorm_nlm=False, backbone_type='parity_backbone',
        positional_embedding_type='custom-rotational-1d',
        out_dims=config.parity_sequence_length * 2,
        neuron_select_type=config.neuron_select_type, lags=config.lags
    ).to(device)

    # --- CRITICAL FIX: Dummy Forward Pass to Init Lazy Modules ---
    model.eval()
    with torch.no_grad():
        pseudo_input = train_data[0][0].unsqueeze(0).to(device)
        _ = model(pseudo_input)
    model.train()

    # Now create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if args.gradient_clipping != -1:
        optimizer = QuantileClip.as_optimizer(optimizer=optimizer, quantile=args.gradient_clipping, history_length=1000)
    
    scheduler = WarmupMultiStepLR(optimizer, warmup_steps=0, milestones=[8000, 15000, 20000], gamma=0.1)

    iterator = iter(trainloader)
    pbar = tqdm(range(config.training_iterations), desc="Training Lagged-CTM")
    
    # --- Best Weight Tracking Setup ---
    best_test_acc = 0.0
    save_dir = "checkpoints/lagged_parity"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pt")

    for bi in pbar:
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(trainloader)
            inputs, targets = next(iterator)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        predictions, certainties, _ = model(inputs) 
        predictions = predictions.reshape(predictions.size(0), -1, 2, predictions.size(-1))
        loss, where_certain = parity_loss(predictions, certainties, targets) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Training Logging
        if bi % args.log_every == 0:
            batch_idx = torch.arange(predictions.size(0), device=device)
            accuracy = (predictions.argmax(2)[batch_idx, :, where_certain] == targets).float().mean()
            
            lag_stats = {f"decays/lag_{l}": math.exp(-model.decay_params_out[f"lag_{l}"].detach().mean().item()) 
                         for l in config.lags} 

            wandb.log({"iteration": bi, "train/loss": loss.item(), "train/accuracy": accuracy.item(),
                       "train/avg_certainty_step": where_certain.float().mean().item(), **lag_stats})
            pbar.set_description(f"Loss: {loss.item():.4f} Acc: {accuracy.item():.4f}")

        # --- Periodic Evaluation & Saving Best Weights ---
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
            
            avg_test_acc = np.mean(test_accs)
            wandb.log({"test/accuracy": avg_test_acc})

            # Check if this is the best version found so far
            if avg_test_acc >= best_test_acc:
                best_test_acc = avg_test_acc
                torch.save({
                    'iteration': bi,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'test_acc': best_test_acc,
                    'args': args
                }, best_model_path)
                # Mark best performance in WandB summary
                wandb.run.summary["best_test_accuracy"] = best_test_acc
            
            model.train()

    wandb.finish()

if __name__ == "__main__":
    train()