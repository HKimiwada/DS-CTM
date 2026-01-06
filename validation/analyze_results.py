"""
Analysis and Visualization Script for CTM Validation Results

This script:
1. Collects results from all validation runs (local files + wandb)
2. Performs statistical significance tests
3. Generates publication-ready visualizations
4. Outputs summary statistics for the research report

Usage:
    python -m validation.analyze_results --checkpoint_dir checkpoints/validation/ctm_validation_v1
    
    # Or pull from wandb directly
    python -m validation.analyze_results --wandb_project lagged-ctm-validation --experiment_name ctm_validation_v1
"""
import argparse
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive CTM Research Analysis")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/validation/ctm_validation_v1')
    parser.add_argument('--wandb_project', type=str, default='lagged-ctm-validation')
    parser.add_argument('--experiment_name', type=str, default='ctm_validation_v1')
    parser.add_argument('--output_dir', type=str, default='validation/analysis_outputs')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--significance_level', type=float, default=0.05)
    return parser.parse_args()

# ==========================================
# Data Loading & Metric Extraction
# ==========================================

def load_local_results(checkpoint_dir: str) -> Dict[str, List[dict]]:
    results = {'baseline': [], 'lagged': []}
    for model_type in ['baseline', 'lagged']:
        pattern = os.path.join(checkpoint_dir, f'{model_type}_seed*', 'final_results.json')
        files = glob.glob(pattern)
        for filepath in files:
            with open(filepath, 'r') as f:
                results[model_type].append(json.load(f))
    print(f"Loaded {len(results['baseline'])} baseline, {len(results['lagged'])} lagged runs.")
    return results

def extract_metrics(results: Dict[str, List[dict]]) -> pd.DataFrame:
    records = []
    for model_type, runs in results.items():
        for run in runs:
            history = run.get('training_history', [])
            
            # --- Fallback Bit-wise Mean Calculation ---
            # If sequence accuracy is 0, we use the average bits correct
            final_bit_acc = 0.0
            best_bit_acc = 0.0
            
            if history:
                final_entry = history[-1]
                # Check top level or history for positional data
                pos_acc = run.get('final_per_position_accuracy') or final_entry.get('test_per_position_accuracy')
                if pos_acc:
                    final_bit_acc = np.mean(pos_acc)
                
                all_bit_means = []
                for h in history:
                    pa = h.get('test_per_position_accuracy') or h.get('eval/test_per_position_accuracy')
                    if pa: all_bit_means.append(np.mean(pa))
                best_bit_acc = max(all_bit_means) if all_bit_means else final_bit_acc

            record = {
                'model_type': model_type,
                'seed': run.get('seed'),
                'final_accuracy': run.get('final_test_accuracy') or final_bit_acc,
                'best_accuracy': run.get('best_test_accuracy') or best_bit_acc,
                'final_loss': run.get('final_test_loss'),
                'certainty_step': run.get('final_certainty_step'),
                'n_parameters': run.get('n_parameters'),
            }
            
            if model_type == 'lagged' and 'lag_statistics' in run:
                for key, value in run['lag_statistics'].items():
                    record[f'lag_{key}'] = value
            records.append(record)
    return pd.DataFrame(records)

# ==========================================
# Statistical Engine
# ==========================================

def perform_full_stats(df: pd.DataFrame, metric: str, alpha: float) -> Dict:
    b = df[df['model_type'] == 'baseline'][metric].dropna().values
    l = df[df['model_type'] == 'lagged'][metric].dropna().values
    if len(b) < 2 or len(l) < 2: return {}

    t_stat, t_p = stats.ttest_ind(l, b)
    u_stat, u_p = stats.mannwhitneyu(l, b)
    welch_stat, welch_p = stats.ttest_ind(l, b, equal_var=False)
    
    pooled_std = np.sqrt(((len(b)-1)*b.std()**2 + (len(l)-1)*l.std()**2) / (len(b)+len(l)-2))
    cohens_d = (l.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0
    
    return {
        'mean_b': b.mean(), 'std_b': b.std(), 'mean_l': l.mean(), 'std_l': l.std(),
        't_p': t_p, 'u_p': u_p, 'welch_p': welch_p, 'cohens_d': cohens_d,
        'sig': t_p < alpha
    }

# ==========================================
# Visualization Suite
# ==========================================

def plot_knowledge_frontier(results: Dict, output_dir: str):
    """Visualizes which bits the model actually solves with clipped variance."""
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    colors = {'baseline': '#66c2a5', 'lagged': '#fc8d62'}
    
    for mtype, color in colors.items():
        pos_data = []
        for run in results[mtype]:
            pa = run.get('final_per_position_accuracy') or \
                 (run['training_history'][-1].get('test_per_position_accuracy') if run.get('training_history') else None)
            if pa: pos_data.append(pa)
            
        if pos_data:
            pos_data = np.array(pos_data)
            mu = np.mean(pos_data, axis=0)
            std = np.std(pos_data, axis=0)
            
            # Define boundaries and CLIP them to [0.0, 1.0]
            upper_bound = np.clip(mu + std, 0, 1.0)
            lower_bound = np.clip(mu - std, 0, 1.0)
            
            x = range(len(mu))
            plt.plot(x, mu, label=f'{mtype.capitalize()}', color=color, lw=2)
            plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.2, edgecolor='none')
    
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random Chance (0.5)')
    plt.ylim(0.45, 1.02) # Set slightly above 1.0 for visual breathing room
    plt.title("Knowledge Frontier: Per-Bit Accuracy")
    plt.xlabel("Bit Position (Sequence Index)")
    plt.ylabel("Accuracy")
    plt.legend(frameon=True, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'knowledge_frontier.png'), dpi=300)
    plt.close()

def plot_comprehensive_comparisons(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [('final_accuracy', 'Bit-wise Accuracy'), 
               ('certainty_step', 'Thinking Steps'), 
               ('final_loss', 'Final Loss')]
    
    for i, (col, name) in enumerate(metrics):
        sns.boxplot(data=df, x='model_type', y=col, ax=axes[i], palette='Set2')
        sns.stripplot(data=df, x='model_type', y=col, ax=axes[i], color='black', alpha=0.3)
        axes[i].set_title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_distributions.png'))

def plot_learning_curves(results: Dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (m_key, title) in enumerate([('test_accuracy', 'Bit Accuracy'), ('test_loss', 'Loss')]):
        for mtype, color in [('baseline', '#66c2a5'), ('lagged', '#fc8d62')]:
            curves = []
            for run in results[mtype]:
                hist = run.get('training_history', [])
                if not hist: continue
                # Extract bit-wise mean if the logged accuracy is sequence-level (0)
                vals = [ (h.get(m_key) if h.get(m_key, 0) > 0 else np.mean(h.get('test_per_position_accuracy', [0]))) for h in hist]
                steps = [h['step'] for h in hist]
                curves.append((steps, vals))
            if curves:
                max_s = max(c[0][-1] for c in curves)
                xs = np.linspace(0, max_s, 100)
                interp_ys = [np.interp(xs, c[0], c[1]) for c in curves]
                axes[i].plot(xs, np.mean(interp_ys, axis=0), color=color, label=mtype, lw=2)
                axes[i].fill_between(xs, np.mean(interp_ys, axis=0)-np.std(interp_ys, axis=0), 
                                     np.mean(interp_ys, axis=0)+np.std(interp_ys, axis=0), color=color, alpha=0.1)
        axes[i].set_title(f"Learning Curve: {title}"), axes[i].legend()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))

def plot_lag_weights(df: pd.DataFrame, output_dir: str):
    l_df = df[df['model_type'] == 'lagged']
    weight_cols = [c for c in l_df.columns if 'weight' in c and 'lag' in c]
    if not weight_cols: return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sync in zip(axes, ['out', 'action']):
        cols = sorted([c for c in weight_cols if sync in c])
        means = [l_df[c].mean() for c in cols]
        errs = [l_df[c].std() for c in cols]
        ax.bar([c.split('_')[-2] for c in cols], means, yerr=errs, capsize=5, color='#8da0cb')
        ax.set_title(f"Learned {sync.capitalize()} Lag Weights")
        ax.set_ylim(0, 1.1)
    plt.savefig(os.path.join(output_dir, 'lag_analysis.png'))

# ==========================================
# Main Execution
# ==========================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = load_local_results(args.checkpoint_dir)
    df = extract_metrics(results)
    df.to_csv(os.path.join(args.output_dir, 'metrics_raw.csv'), index=False)

    # Generate Statistical Report
    with open(os.path.join(args.output_dir, 'statistical_report.txt'), 'w') as f:
        f.write("CTM COMPREHENSIVE ANALYSIS REPORT\n" + "="*40 + "\n")
        for metric in ['final_accuracy', 'certainty_step', 'final_loss']:
            res = perform_full_stats(df, metric, args.significance_level)
            if not res: continue
            f.write(f"\nMETRIC: {metric.upper()}\n")
            f.write(f"Baseline: {res['mean_b']:.4f} ± {res['std_b']:.4f}\n")
            f.write(f"Lagged:   {res['mean_l']:.4f} ± {res['std_l']:.4f}\n")
            f.write(f"P-Value (Welch): {res['welch_p']:.6f} | Cohen's d: {res['cohens_d']:.3f}\n")
            f.write(f"Significant: {res['sig']}\n")

    print("\nGenerating Paper-Ready Visualizations...")
    plot_knowledge_frontier(results, args.output_dir)
    plot_comprehensive_comparisons(df, args.output_dir)
    plot_learning_curves(results, args.output_dir)
    plot_lag_weights(df, args.output_dir)
    
    print(f"Analysis complete. All files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()