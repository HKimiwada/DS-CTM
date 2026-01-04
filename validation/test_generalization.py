"""
Generalization Test Script

Tests pre-trained models (trained on 64-bit parity) on longer sequences:
- 128-bit
- 256-bit
- 512-bit

This tests how well the learned temporal representations generalize
to longer reasoning chains.

Usage:
    python -m validation.test_generalization \
        --lagged_checkpoint checkpoints/lagged_parity/best_model.pt \
        --baseline_checkpoint checkpoints/baseline_parity_seed_42/best_baseline_model.pt
"""

import argparse
import os
import json
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict

from data.custom_datasets import ParityDataset
from models.ctm import ContinuousThoughtMachine
from models.lagged_ctm import LaggedContinuousThoughtMachine
from utils.losses import parity_loss
from utils.housekeeping import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Test CTM Generalization to Longer Sequences")
    
    # Checkpoints
    parser.add_argument('--lagged_checkpoint', type=str, required=True,
                        help='Path to lagged CTM checkpoint')
    parser.add_argument('--baseline_checkpoint', type=str, required=True,
                        help='Path to baseline CTM checkpoint')
    
    # Test configurations
    parser.add_argument('--sequence_lengths', type=int, nargs='+', 
                        default=[64, 128, 256, 512],
                        help='Sequence lengths to test')
    parser.add_argument('--n_test_samples', type=int, default=5000,
                        help='Number of test samples per sequence length')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of evaluation runs for variance estimation')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='validation/generalization_outputs',
                        help='Directory to save outputs')
    parser.add_argument('--wandb_project', type=str, default='lagged-ctm-validation',
                        help='WandB project name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """
    Load a CTM / Lagged CTM checkpoint robustly.

    Handles:
    - argparse.Namespace
    - dict
    - wandb.Config (BROKEN legacy checkpoints)
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} missing model_state_dict")

    if "args" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} missing args")

    raw_args = checkpoint["args"]

    # -----------------------------
    # SAFELY NORMALIZE args
    # -----------------------------
    if isinstance(raw_args, argparse.Namespace):
        model_args = raw_args

    elif isinstance(raw_args, dict):
        model_args = argparse.Namespace(**raw_args)

    # wandb.Config (legacy footgun)
    elif hasattr(raw_args, "_items"):  # wandb.sdk.wandb_config.Config
        model_args = argparse.Namespace(**dict(raw_args._items))

    else:
        raise TypeError(
            f"Unsupported args type in checkpoint: {type(raw_args)}"
        )

    # -----------------------------
    # Detect model type (NO hasattr)
    # -----------------------------
    model_args_dict = vars(model_args)
    is_lagged = (
        "lags" in model_args_dict or
        any(k.startswith("decay_params_out.lag_") 
            for k in checkpoint["model_state_dict"].keys())
    )

    # -----------------------------
    # Build model
    # -----------------------------
    out_dims = model_args.parity_sequence_length * 2
    prediction_reshaper = [model_args.parity_sequence_length, 2]

    common_kwargs = dict(
        iterations=model_args.iterations,
        d_model=model_args.d_model,
        d_input=model_args.d_input,
        heads=model_args.heads,
        n_synch_out=model_args.n_synch_out,
        n_synch_action=model_args.n_synch_action,
        synapse_depth=model_args.synapse_depth,
        memory_length=model_args.memory_length,
        deep_nlms=model_args_dict.get("deep_memory", True),
        memory_hidden_dims=model_args.memory_hidden_dims,
        do_layernorm_nlm=False,
        backbone_type="parity_backbone",
        positional_embedding_type="custom-rotational-1d",
        out_dims=out_dims,
        prediction_reshaper=prediction_reshaper,
        dropout=model_args_dict.get("dropout", 0.0),
        neuron_select_type=model_args_dict.get("neuron_select_type", "random"),
        n_random_pairing_self=model_args_dict.get(
            "n_random_pairing_self", 256
        ),
    )

    if is_lagged:
        lags = model_args_dict.get("lags", [0, 1, 2, 4])
        model = LaggedContinuousThoughtMachine(
            **common_kwargs, lags=lags
        )
        model_type = "lagged"
    else:
        model = ContinuousThoughtMachine(**common_kwargs)
        model_type = "baseline"

    model = model.to(device)

    # -----------------------------
    # Initialize lazy modules safely
    # -----------------------------
    model.eval()
    with torch.no_grad():
        dummy_ds = ParityDataset(
            sequence_length=model_args.parity_sequence_length,
            length=1
        )
        dummy_input = dummy_ds[0][0].unsqueeze(0).to(device)
        _ = model(dummy_input)

    # -----------------------------
    # Load weights
    # -----------------------------
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, {
        "type": model_type,
        "args": model_args,
        "checkpoint_path": checkpoint_path,
    }

def adapt_model_for_sequence_length(model: torch.nn.Module, new_seq_length: int, 
                                    original_seq_length: int, device: str) -> torch.nn.Module:
    """
    Adapt a model trained on one sequence length to handle a different length.
    
    For parity, we need to:
    1. Update the output projector to handle new output dimensions
    2. Update prediction_reshaper
    """
    # The backbone (parity_backbone) uses embeddings which are sequence-length agnostic
    # The output projector needs to be adapted
    
    new_out_dims = new_seq_length * 2
    model.out_dims = new_out_dims
    model.prediction_reshaper = [new_seq_length, 2]
    
    # Reinitialize output projector with new dimensions
    # Get the input dimension from synch representation
    sync_size = model.synch_representation_size_out
    if hasattr(model, 'lags'):
        sync_size = sync_size * len(model.lags)
    
    # Create new output projector
    old_output_projector = model.output_projector
    model.output_projector = torch.nn.Sequential(
        torch.nn.Linear(sync_size, new_out_dims)
    ).to(device)
    
    # Initialize with small random weights
    torch.nn.init.xavier_uniform_(model.output_projector[0].weight)
    torch.nn.init.zeros_(model.output_projector[0].bias)
    
    return model


def evaluate_on_sequence_length(model: torch.nn.Module, seq_length: int, 
                                n_samples: int, batch_size: int, 
                                device: str, use_most_certain: bool = True) -> Dict:
    """Evaluate model on a specific sequence length."""
    model.eval()
    
    # Generate test data
    test_data = ParityDataset(sequence_length=seq_length, length=n_samples)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    all_losses = []
    all_element_accuracies = []  # Per-element accuracy
    all_sequence_accuracies = []  # Full sequence correct
    all_certainty_steps = []
    per_position_correct = defaultdict(list)
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc=f"Eval seq_len={seq_length}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            predictions, certainties, _ = model(inputs)
            predictions = predictions.reshape(predictions.size(0), -1, 2, predictions.size(-1))
            
            loss, where_certain = parity_loss(predictions, certainties, targets, 
                                              use_most_certain=use_most_certain)
            
            batch_idx = torch.arange(predictions.size(0), device=device)
            preds_at_certain = predictions.argmax(2)[batch_idx, :, where_certain]
            
            # Per-element accuracy
            element_correct = (preds_at_certain == targets).float()
            all_element_accuracies.append(element_correct.mean().item())
            
            # Full sequence accuracy
            sequence_correct = element_correct.all(dim=1).float()
            all_sequence_accuracies.append(sequence_correct.mean().item())
            
            all_losses.append(loss.item())
            all_certainty_steps.append(where_certain.float().mean().item())
            
            # Track per-position accuracy
            for pos in range(seq_length):
                pos_correct = (preds_at_certain[:, pos] == targets[:, pos]).float().mean().item()
                per_position_correct[pos].append(pos_correct)
    
    # Aggregate per-position accuracy
    position_accuracy = np.array([np.mean(per_position_correct[p]) for p in range(seq_length)])
    
    return {
        'sequence_length': seq_length,
        'loss': np.mean(all_losses),
        'loss_std': np.std(all_losses),
        'element_accuracy': np.mean(all_element_accuracies),
        'element_accuracy_std': np.std(all_element_accuracies),
        'sequence_accuracy': np.mean(all_sequence_accuracies),
        'sequence_accuracy_std': np.std(all_sequence_accuracies),
        'certainty_step': np.mean(all_certainty_steps),
        'certainty_step_std': np.std(all_certainty_steps),
        'position_accuracy': position_accuracy.tolist(),
        'position_accuracy_mean': position_accuracy.mean(),
        'position_accuracy_std': position_accuracy.std(),
    }


def run_generalization_test(args):
    """Run the full generalization test."""
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name="generalization_test",
        config=vars(args),
        tags=["generalization"]
    )
    
    # Load models
    print("Loading models...")
    lagged_model, lagged_info = load_model_from_checkpoint(args.lagged_checkpoint, device)
    baseline_model, baseline_info = load_model_from_checkpoint(args.baseline_checkpoint, device)
    
    original_seq_length = lagged_info['args'].parity_sequence_length
    print(f"Models trained on {original_seq_length}-bit sequences")
    
    # Store results
    results = {
        'baseline': {},
        'lagged': {},
        'config': vars(args),
        'original_sequence_length': original_seq_length,
    }
    
    # Test each sequence length
    for seq_length in args.sequence_lengths:
        print(f"\n{'='*60}")
        print(f"Testing on {seq_length}-bit sequences")
        print('='*60)
        
        # For the original training length, use the model as-is
        # For other lengths, we need to adapt
        if seq_length != original_seq_length:
            print("Adapting models for new sequence length...")
            # Create fresh copies and adapt
            lagged_test, _ = load_model_from_checkpoint(args.lagged_checkpoint, device)
            baseline_test, _ = load_model_from_checkpoint(args.baseline_checkpoint, device)
            
            lagged_test = adapt_model_for_sequence_length(
                lagged_test, seq_length, original_seq_length, device
            )
            baseline_test = adapt_model_for_sequence_length(
                baseline_test, seq_length, original_seq_length, device
            )
        else:
            lagged_test = lagged_model
            baseline_test = baseline_model
        
        # Run multiple evaluation runs for variance estimation
        baseline_runs = []
        lagged_runs = []
        
        for run_idx in range(args.n_runs):
            set_seed(args.seed + run_idx)
            
            print(f"\nRun {run_idx + 1}/{args.n_runs}")
            
            # Evaluate baseline
            baseline_result = evaluate_on_sequence_length(
                baseline_test, seq_length, args.n_test_samples,
                args.batch_size, device
            )
            baseline_runs.append(baseline_result)
            
            # Evaluate lagged
            lagged_result = evaluate_on_sequence_length(
                lagged_test, seq_length, args.n_test_samples,
                args.batch_size, device
            )
            lagged_runs.append(lagged_result)
        
        # Aggregate results across runs
        def aggregate_runs(runs):
            aggregated = {'sequence_length': seq_length}
            numeric_keys = ['loss', 'element_accuracy', 'sequence_accuracy', 
                          'certainty_step', 'position_accuracy_mean']
            
            for key in numeric_keys:
                values = [r[key] for r in runs]
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
            
            # Aggregate position accuracy
            all_pos_acc = np.array([r['position_accuracy'] for r in runs])
            aggregated['position_accuracy'] = np.mean(all_pos_acc, axis=0).tolist()
            aggregated['position_accuracy_std'] = np.std(all_pos_acc, axis=0).tolist()
            
            return aggregated
        
        results['baseline'][seq_length] = aggregate_runs(baseline_runs)
        results['lagged'][seq_length] = aggregate_runs(lagged_runs)
        
        # Log to wandb
        wandb.log({
            f'baseline/seq{seq_length}_element_accuracy': results['baseline'][seq_length]['element_accuracy_mean'],
            f'baseline/seq{seq_length}_sequence_accuracy': results['baseline'][seq_length]['sequence_accuracy_mean'],
            f'lagged/seq{seq_length}_element_accuracy': results['lagged'][seq_length]['element_accuracy_mean'],
            f'lagged/seq{seq_length}_sequence_accuracy': results['lagged'][seq_length]['sequence_accuracy_mean'],
        })
        
        # Print summary
        print(f"\nResults for {seq_length}-bit:")
        print(f"  Baseline element accuracy: {results['baseline'][seq_length]['element_accuracy_mean']:.4f} ± {results['baseline'][seq_length]['element_accuracy_std']:.4f}")
        print(f"  Lagged element accuracy:   {results['lagged'][seq_length]['element_accuracy_mean']:.4f} ± {results['lagged'][seq_length]['element_accuracy_std']:.4f}")
        print(f"  Baseline sequence accuracy: {results['baseline'][seq_length]['sequence_accuracy_mean']:.4f}")
        print(f"  Lagged sequence accuracy:   {results['lagged'][seq_length]['sequence_accuracy_mean']:.4f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'generalization_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate visualizations
    generate_generalization_plots(results, args.output_dir)
    
    wandb.finish()
    
    return results


def generate_generalization_plots(results: Dict, output_dir: str):
    """Generate visualizations for generalization results."""
    
    seq_lengths = sorted([int(k) for k in results['baseline'].keys()])
    
    # 1. Accuracy vs Sequence Length
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Element accuracy
    ax1 = axes[0]
    baseline_elem_acc = [results['baseline'][sl]['element_accuracy_mean'] for sl in seq_lengths]
    baseline_elem_std = [results['baseline'][sl]['element_accuracy_std'] for sl in seq_lengths]
    lagged_elem_acc = [results['lagged'][sl]['element_accuracy_mean'] for sl in seq_lengths]
    lagged_elem_std = [results['lagged'][sl]['element_accuracy_std'] for sl in seq_lengths]
    
    x = np.arange(len(seq_lengths))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_elem_acc, width, yerr=baseline_elem_std,
                    label='Baseline CTM', color='#66c2a5', capsize=5)
    bars2 = ax1.bar(x + width/2, lagged_elem_acc, width, yerr=lagged_elem_std,
                    label='Lagged CTM', color='#fc8d62', capsize=5)
    
    ax1.set_xlabel('Sequence Length (bits)')
    ax1.set_ylabel('Element-wise Accuracy')
    ax1.set_title('Per-Element Accuracy vs Sequence Length')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_lengths)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    # Mark training length
    train_idx = seq_lengths.index(results['original_sequence_length'])
    ax1.axvline(x=train_idx, color='red', linestyle=':', alpha=0.7, label='Training Length')
    
    # Sequence accuracy
    ax2 = axes[1]
    baseline_seq_acc = [results['baseline'][sl]['sequence_accuracy_mean'] for sl in seq_lengths]
    baseline_seq_std = [results['baseline'][sl]['sequence_accuracy_std'] for sl in seq_lengths]
    lagged_seq_acc = [results['lagged'][sl]['sequence_accuracy_mean'] for sl in seq_lengths]
    lagged_seq_std = [results['lagged'][sl]['sequence_accuracy_std'] for sl in seq_lengths]
    
    ax2.errorbar(seq_lengths, baseline_seq_acc, yerr=baseline_seq_std,
                 marker='o', markersize=8, capsize=5, label='Baseline CTM',
                 color='#66c2a5', linewidth=2)
    ax2.errorbar(seq_lengths, lagged_seq_acc, yerr=lagged_seq_std,
                 marker='s', markersize=8, capsize=5, label='Lagged CTM',
                 color='#fc8d62', linewidth=2)
    
    ax2.set_xlabel('Sequence Length (bits)')
    ax2.set_ylabel('Full Sequence Accuracy')
    ax2.set_title('Sequence-level Accuracy vs Sequence Length')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(seq_lengths)
    ax2.set_xticklabels(seq_lengths)
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.axvline(x=results['original_sequence_length'], color='red', linestyle=':', 
                alpha=0.7, label='Training Length')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_sequence_length.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_sequence_length.pdf'), bbox_inches='tight')
    plt.close()
    
    # 2. Per-Position Accuracy Heatmaps
    fig, axes = plt.subplots(2, len(seq_lengths), figsize=(4*len(seq_lengths), 8))
    
    for i, sl in enumerate(seq_lengths):
        # Baseline
        pos_acc_baseline = np.array(results['baseline'][sl]['position_accuracy'])
        side = int(np.sqrt(sl))
        if side * side == sl:
            baseline_grid = pos_acc_baseline.reshape(side, side)
        else:
            baseline_grid = pos_acc_baseline.reshape(1, -1)
        
        ax = axes[0, i] if len(seq_lengths) > 1 else axes[0]
        im = ax.imshow(baseline_grid, cmap='viridis', vmin=0.5, vmax=1.0)
        ax.set_title(f'Baseline {sl}-bit')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Lagged
        pos_acc_lagged = np.array(results['lagged'][sl]['position_accuracy'])
        if side * side == sl:
            lagged_grid = pos_acc_lagged.reshape(side, side)
        else:
            lagged_grid = pos_acc_lagged.reshape(1, -1)
        
        ax = axes[1, i] if len(seq_lengths) > 1 else axes[1]
        im = ax.imshow(lagged_grid, cmap='viridis', vmin=0.5, vmax=1.0)
        ax.set_title(f'Lagged {sl}-bit')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Per-Position Accuracy', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_accuracy_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'position_accuracy_heatmaps.pdf'), bbox_inches='tight')
    plt.close()
    
    # 3. Accuracy Decay Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize by training accuracy
    train_sl = results['original_sequence_length']
    baseline_train_acc = results['baseline'][train_sl]['element_accuracy_mean']
    lagged_train_acc = results['lagged'][train_sl]['element_accuracy_mean']
    
    baseline_relative = [results['baseline'][sl]['element_accuracy_mean'] / baseline_train_acc 
                         for sl in seq_lengths]
    lagged_relative = [results['lagged'][sl]['element_accuracy_mean'] / lagged_train_acc 
                       for sl in seq_lengths]
    
    ax.plot(seq_lengths, baseline_relative, 'o-', markersize=10, linewidth=2,
            label='Baseline CTM', color='#66c2a5')
    ax.plot(seq_lengths, lagged_relative, 's-', markersize=10, linewidth=2,
            label='Lagged CTM', color='#fc8d62')
    
    ax.set_xlabel('Sequence Length (bits)')
    ax.set_ylabel('Relative Accuracy (vs Training Length)')
    ax.set_title('Accuracy Retention During Generalization')
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lengths)
    ax.set_xticklabels(seq_lengths)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=train_sl, color='red', linestyle=':', alpha=0.7, label='Training Length')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_retention.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'accuracy_retention.pdf'), bbox_inches='tight')
    plt.close()
    
    # 4. Summary Table
    summary_data = []
    for sl in seq_lengths:
        summary_data.append({
            'Sequence Length': sl,
            'Baseline Elem Acc': f"{results['baseline'][sl]['element_accuracy_mean']:.4f}",
            'Lagged Elem Acc': f"{results['lagged'][sl]['element_accuracy_mean']:.4f}",
            'Baseline Seq Acc': f"{results['baseline'][sl]['sequence_accuracy_mean']:.4f}",
            'Lagged Seq Acc': f"{results['lagged'][sl]['sequence_accuracy_mean']:.4f}",
            'Improvement (Elem)': f"{(results['lagged'][sl]['element_accuracy_mean'] - results['baseline'][sl]['element_accuracy_mean'])*100:.2f}%",
        })
    
    import pandas as pd
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'generalization_summary.csv'), index=False)
    
    print("\n" + "="*80)
    print("GENERALIZATION SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    return summary_df


if __name__ == "__main__":
    args = parse_args()
    results = run_generalization_test(args)