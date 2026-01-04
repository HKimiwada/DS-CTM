"""
Validation Training Script for Lagged-CTM vs Baseline CTM
Runs multiple seeds with comprehensive logging for statistical analysis.

Usage:
    # Single run
    python -m validation.train_validation --model_type lagged --seed 42
    
    # Use the shell scripts for parallel multi-seed runs
"""

import argparse
import os
import json
import torch
import numpy as np
import wandb
from tqdm.auto import tqdm
from datetime import datetime

from autoclip.torch import QuantileClip
from data.custom_datasets import ParityDataset
from models.ctm import ContinuousThoughtMachine
from models.lagged_ctm import LaggedContinuousThoughtMachine
from utils.losses import parity_loss
from utils.schedulers import WarmupCosineAnnealingLR
from utils.housekeeping import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Validation Training for CTM Comparison")
    
    # Experiment identification
    parser.add_argument('--model_type', type=str, required=True, choices=['baseline', 'lagged'],
                        help='Model type to train')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--experiment_name', type=str, default='ctm_validation_v1',
                        help='Experiment group name for wandb')
    parser.add_argument('--wandb_project', type=str, default='lagged-ctm-validation',
                        help='WandB project name')
    
    # Lagged-CTM specific
    parser.add_argument('--lags', type=int, nargs='+', default=[0, 1, 2, 4],
                        help='Lag values for LaggedCTM')
    
    # Architecture (matched between models)
    parser.add_argument('--parity_sequence_length', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--d_input', type=int, default=512)
    parser.add_argument('--synapse_depth', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--n_synch_out', type=int, default=32)
    parser.add_argument('--n_synch_action', type=int, default=32)
    parser.add_argument('--neuron_select_type', type=str, default='random')
    parser.add_argument('--n_random_pairing_self', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=75)
    parser.add_argument('--memory_length', type=int, default=25)
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--memory_hidden_dims', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--training_iterations', type=int, default=200001)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--gradient_clipping', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    
    # Logging
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--n_eval_batches', type=int, default=40)
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/validation')
    
    return parser.parse_args()


def build_model(args, device):
    """Build either baseline CTM or LaggedCTM based on args."""
    out_dims = args.parity_sequence_length * 2
    prediction_reshaper = [args.parity_sequence_length, 2]
    
    if args.model_type == 'baseline':
        model = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=False,
            backbone_type='parity_backbone',
            positional_embedding_type='custom-rotational-1d',
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
        )
    else:  # lagged
        model = LaggedContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=False,
            backbone_type='parity_backbone',
            positional_embedding_type='custom-rotational-1d',
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
            lags=args.lags,
        )
    
    return model.to(device)


def evaluate(model, dataloader, device, args, n_batches=None):
    """Comprehensive evaluation returning multiple metrics."""
    model.eval()
    
    all_losses = []
    all_accuracies = []
    all_certainty_steps = []
    all_per_position_acc = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if n_batches and i >= n_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            predictions, certainties, _ = model(inputs)
            predictions = predictions.reshape(predictions.size(0), -1, 2, predictions.size(-1))
            
            loss, where_certain = parity_loss(predictions, certainties, targets)
            
            batch_idx = torch.arange(predictions.size(0), device=device)
            preds_at_certain = predictions.argmax(2)[batch_idx, :, where_certain]
            
            # Per-sample accuracy (all positions correct)
            per_sample_correct = (preds_at_certain == targets).all(dim=1).float()
            
            # Per-position accuracy
            per_position_correct = (preds_at_certain == targets).float().mean(dim=0)
            
            all_losses.append(loss.item())
            all_accuracies.append(per_sample_correct.mean().item())
            all_certainty_steps.append(where_certain.float().mean().item())
            all_per_position_acc.append(per_position_correct.cpu().numpy())
    
    model.train()
    
    return {
        'loss': np.mean(all_losses),
        'loss_std': np.std(all_losses),
        'accuracy': np.mean(all_accuracies),
        'accuracy_std': np.std(all_accuracies),
        'certainty_step': np.mean(all_certainty_steps),
        'certainty_step_std': np.std(all_certainty_steps),
        'per_position_accuracy': np.mean(all_per_position_acc, axis=0),
    }


def get_lag_statistics(model, args):
    """Extract learned lag decay parameters for LaggedCTM."""
    if args.model_type != 'lagged':
        return {}
    
    stats = {}
    for sync_type in ['action', 'out']:
        decay_dict = getattr(model, f'decay_params_{sync_type}', None)
        if decay_dict is None:
            continue
            
        for lag in args.lags:
            key = f'lag_{lag}'
            if key in decay_dict:
                params = decay_dict[key].detach()
                effective_weight = torch.exp(-params.clamp(0, 15)).mean().item()
                stats[f'{sync_type}_lag_{lag}_weight'] = effective_weight
                stats[f'{sync_type}_lag_{lag}_decay_mean'] = params.mean().item()
                stats[f'{sync_type}_lag_{lag}_decay_std'] = params.std().item()
    
    return stats


def train():
    args = parse_args()
    set_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create checkpoint directory
    run_name = f"{args.model_type}_seed{args.seed}"
    save_dir = os.path.join(args.checkpoint_dir, args.experiment_name, run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        group=args.experiment_name,
        name=run_name,
        config=vars(args),
        tags=[args.model_type, f"seed_{args.seed}"]
    )
    
    # Save args
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Data
    train_data = ParityDataset(sequence_length=args.parity_sequence_length, length=100000)
    test_data = ParityDataset(sequence_length=args.parity_sequence_length, length=10000)
    
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size_test, shuffle=False, num_workers=0
    )
    
    # Model
    model = build_model(args, device)
    
    # Initialize lazy modules
    model.eval()
    with torch.no_grad():
        dummy = train_data[0][0].unsqueeze(0).to(device)
        _ = model(dummy)
    model.train()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_type}, Parameters: {n_params:,}")
    wandb.log({"model/parameters": n_params})
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.gradient_clipping > 0:
        optimizer = QuantileClip.as_optimizer(
            optimizer=optimizer, quantile=args.gradient_clipping, history_length=1000
        )
    
    scheduler = WarmupCosineAnnealingLR(
        optimizer, args.warmup_steps, args.training_iterations,
        warmup_start_lr=1e-20, eta_min=1e-7
    )
    
    # Training state
    best_test_acc = 0.0
    training_history = []
    
    iterator = iter(trainloader)
    pbar = tqdm(range(args.training_iterations), desc=f"{args.model_type} seed={args.seed}")
    
    for step in pbar:
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(trainloader)
            inputs, targets = next(iterator)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward
        predictions, certainties, _ = model(inputs)
        predictions = predictions.reshape(predictions.size(0), -1, 2, predictions.size(-1))
        loss, where_certain = parity_loss(predictions, certainties, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Quick training metrics
        if step % args.log_every == 0:
            batch_idx = torch.arange(inputs.size(0), device=device)
            train_acc = (predictions.argmax(2)[batch_idx, :, where_certain] == targets).float().mean()
            
            log_dict = {
                "train/loss": loss.item(),
                "train/accuracy": train_acc.item(),
                "train/certainty_step_mean": where_certain.float().mean().item(),
                "train/certainty_step_std": where_certain.float().std().item(),
                "train/lr": scheduler.get_last_lr()[0],
                "step": step,
            }
            
            # Add lag statistics for lagged model
            lag_stats = get_lag_statistics(model, args)
            for k, v in lag_stats.items():
                log_dict[f"lags/{k}"] = v
            
            wandb.log(log_dict)
            pbar.set_description(
                f"{args.model_type} s={args.seed} | loss={loss.item():.3f} acc={train_acc.item():.3f}"
            )
        
        # Full evaluation
        if step % args.eval_every == 0:
            train_metrics = evaluate(model, trainloader, device, args, n_batches=args.n_eval_batches)
            test_metrics = evaluate(model, testloader, device, args, n_batches=args.n_eval_batches)
            
            eval_log = {
                "eval/train_loss": train_metrics['loss'],
                "eval/train_accuracy": train_metrics['accuracy'],
                "eval/train_certainty_step": train_metrics['certainty_step'],
                "eval/test_loss": test_metrics['loss'],
                "eval/test_accuracy": test_metrics['accuracy'],
                "eval/test_certainty_step": test_metrics['certainty_step'],
                "step": step,
            }
            
            # Log per-position accuracy as histogram
            wandb.log({
                "eval/test_per_position_accuracy": wandb.Histogram(test_metrics['per_position_accuracy']),
                **eval_log
            })
            
            # Track history for final analysis
            training_history.append({
                'step': step,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'train_certainty_step': train_metrics['certainty_step'],
                'test_loss': test_metrics['loss'],
                'test_accuracy': test_metrics['accuracy'],
                'test_certainty_step': test_metrics['certainty_step'],
                'test_per_position_accuracy': test_metrics['per_position_accuracy'].tolist(),
            })
            
            # Save best model
            if test_metrics['accuracy'] > best_test_acc:
                best_test_acc = test_metrics['accuracy']
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy': best_test_acc,
                    'test_metrics': test_metrics,
                    'args': vars(args),
                }, os.path.join(save_dir, 'best_model.pt'))
                
                wandb.run.summary['best_test_accuracy'] = best_test_acc
                wandb.run.summary['best_step'] = step
        
        # Periodic checkpoint
        if step % args.save_every == 0 and step > 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_history': training_history,
                'args': vars(args),
            }, os.path.join(save_dir, f'checkpoint_{step}.pt'))
    
    # Final evaluation (full test set)
    print("\nRunning final evaluation on full test set...")
    final_test_metrics = evaluate(model, testloader, device, args, n_batches=None)
    
    # Save final results
    final_results = {
        'model_type': args.model_type,
        'seed': args.seed,
        'best_test_accuracy': best_test_acc,
        'final_test_accuracy': final_test_metrics['accuracy'],
        'final_test_loss': final_test_metrics['loss'],
        'final_certainty_step': final_test_metrics['certainty_step'],
        'final_per_position_accuracy': final_test_metrics['per_position_accuracy'].tolist(),
        'training_history': training_history,
        'n_parameters': n_params,
    }
    
    if args.model_type == 'lagged':
        final_results['lag_statistics'] = get_lag_statistics(model, args)
    
    with open(os.path.join(save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Log final metrics
    wandb.run.summary['final_test_accuracy'] = final_test_metrics['accuracy']
    wandb.run.summary['final_test_loss'] = final_test_metrics['loss']
    wandb.run.summary['final_certainty_step'] = final_test_metrics['certainty_step']
    
    # Save final model
    torch.save({
        'step': args.training_iterations - 1,
        'model_state_dict': model.state_dict(),
        'final_metrics': final_test_metrics,
        'args': vars(args),
    }, os.path.join(save_dir, 'final_model.pt'))
    
    wandb.finish()
    print(f"\nTraining complete! Best accuracy: {best_test_acc:.4f}")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    train()