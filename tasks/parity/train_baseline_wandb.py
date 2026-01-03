import argparse
import math
import os
import torch
import numpy as np
import wandb
from tqdm.auto import tqdm

# Original repo imports
from autoclip.torch import QuantileClip
from data.custom_datasets import ParityDataset
from tasks.parity.utils import prepare_model
from utils.losses import parity_loss
from utils.schedulers import WarmupMultiStepLR
from utils.housekeeping import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline-CTM on Parity Task (with WandB)")

    # --- Added for WandB Support ---
    parser.add_argument('--wandb_project', type=str, default='lagged-ctm-research', help='WandB project name.')
    
    # --- Standard CTM Architecture (Replicated from original tasks/parity/train.py) ---
    parser.add_argument('--model_type', type=str, default="ctm", choices=['ctm', 'lstm'])
    parser.add_argument('--parity_sequence_length', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--d_input', type=int, default=512)
    parser.add_argument('--synapse_depth', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--n_synch_out', type=int, default=32)
    parser.add_argument('--n_synch_action', type=int, default=32)
    parser.add_argument('--neuron_select_type', type=str, default='random', choices=['first-last', 'random', 'random-pairing'])
    parser.add_argument('--n_random_pairing_self', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=75)
    parser.add_argument('--memory_length', type=int, default=25)
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--memory_hidden_dims', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--positional_embedding_type', type=str, default='custom-rotational-1d')
    parser.add_argument('--backbone_type', type=str, default='parity_backbone')

    # --- Training Configuration ---
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--training_iterations', type=int, default=50001)
    parser.add_argument('--gradient_clipping', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--track_every', type=int, default=1000)
    parser.add_argument('--use_most_certain', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_most_certain_with_lstm', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--n_test_batches', type=int, default=20)
    parser.add_argument('--full_eval', action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()

def train_baseline():
    args = parse_args()
    set_seed(args.seed)
    args.out_dims = args.parity_sequence_length * 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Initialize WandB with the project name from CLI
    wandb.init(project=args.wandb_project, config=args, tags=["Baseline"])
    config = wandb.config

    train_data = ParityDataset(sequence_length=config.parity_sequence_length, length=100000)
    test_data = ParityDataset(sequence_length=config.parity_sequence_length, length=10000)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size_test, shuffle=False)

    # 2. Build model
    prediction_reshaper = [config.parity_sequence_length, 2]
    model = prepare_model(prediction_reshaper, config, device)
    
    # Dummy pass for Lazy modules
    model.eval()
    with torch.no_grad():
        pseudo_input = train_data[0][0].unsqueeze(0).to(device)
        _ = model(pseudo_input)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if config.gradient_clipping != -1:
        optimizer = QuantileClip.as_optimizer(optimizer=optimizer, quantile=config.gradient_clipping)
    
    scheduler = WarmupMultiStepLR(optimizer, warmup_steps=0, milestones=[8000, 15000, 20000], gamma=0.1)

    # Setup for best weight saving
    best_test_acc = 0.0
    save_dir = f"checkpoints/baseline_parity_seed_{config.seed}"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_baseline_model.pt")

    iterator = iter(trainloader)
    pbar = tqdm(range(config.training_iterations), desc=f"Baseline Seed {config.seed}")
    
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

        if bi % config.log_every == 0:
            batch_idx = torch.arange(inputs.size(0), device=device)
            acc = (predictions.argmax(2)[batch_idx, :, where_certain] == targets).float().mean()
            wandb.log({
                "iteration": bi,
                "train/loss": loss.item(),
                "train/accuracy": acc.item(),
                "train/avg_certainty_step": where_certain.float().mean().item()
            })

        if bi % config.track_every == 0:
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
                    if not config.full_eval and len(test_accs) >= config.n_test_batches: break
            
            avg_test_acc = np.mean(test_accs)
            wandb.log({"test/accuracy": avg_test_acc})

            if avg_test_acc >= best_test_acc:
                best_test_acc = avg_test_acc
                torch.save({
                    'iteration': bi,
                    'model_state_dict': model.state_dict(),
                    'test_acc': best_test_acc,
                    'args': config
                }, best_model_path)
                wandb.run.summary["best_test_accuracy"] = best_test_acc
            
            model.train()

    wandb.finish()

if __name__ == "__main__":
    train_baseline()