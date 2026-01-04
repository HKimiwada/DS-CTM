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

# Try to import wandb for API access
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Will only use local files.")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze CTM Validation Results")
    
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='checkpoints/validation/ctm_validation_v1',
                        help='Directory containing validation checkpoints')
    parser.add_argument('--wandb_project', type=str, default='lagged-ctm-validation',
                        help='WandB project name')
    parser.add_argument('--experiment_name', type=str, default='ctm_validation_v1',
                        help='Experiment group name')
    parser.add_argument('--output_dir', type=str, default='validation/analysis_outputs',
                        help='Directory to save analysis outputs')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Pull data from wandb API')
    parser.add_argument('--significance_level', type=float, default=0.05,
                        help='Significance level for statistical tests')
    
    return parser.parse_args()


def load_local_results(checkpoint_dir: str) -> Dict[str, List[dict]]:
    """Load results from local JSON files."""
    results = {'baseline': [], 'lagged': []}
    
    for model_type in ['baseline', 'lagged']:
        pattern = os.path.join(checkpoint_dir, f'{model_type}_seed*', 'final_results.json')
        files = glob.glob(pattern)
        
        for filepath in files:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results[model_type].append(data)
    
    print(f"Loaded {len(results['baseline'])} baseline runs, {len(results['lagged'])} lagged runs")
    return results


def load_wandb_results(project: str, experiment_name: str) -> Dict[str, List[dict]]:
    """Load results from wandb API."""
    if not WANDB_AVAILABLE:
        raise ImportError("wandb not installed")
    
    api = wandb.Api()
    runs = api.runs(project, filters={"group": experiment_name})
    
    results = {'baseline': [], 'lagged': []}
    
    for run in runs:
        model_type = run.config.get('model_type', 'unknown')
        if model_type not in results:
            continue
            
        run_data = {
            'model_type': model_type,
            'seed': run.config.get('seed'),
            'final_test_accuracy': run.summary.get('final_test_accuracy'),
            'final_test_loss': run.summary.get('final_test_loss'),
            'final_certainty_step': run.summary.get('final_certainty_step'),
            'best_test_accuracy': run.summary.get('best_test_accuracy'),
            'n_parameters': run.summary.get('model/parameters'),
            'config': dict(run.config),
            'history': run.history(pandas=True),
        }
        results[model_type].append(run_data)
    
    print(f"Loaded {len(results['baseline'])} baseline runs, {len(results['lagged'])} lagged runs from wandb")
    return results


def extract_metrics(results: Dict[str, List[dict]]) -> pd.DataFrame:
    """Extract key metrics into a DataFrame for analysis."""
    records = []
    
    for model_type, runs in results.items():
        for run in runs:
            record = {
                'model_type': model_type,
                'seed': run.get('seed'),
                'final_accuracy': run.get('final_test_accuracy'),
                'best_accuracy': run.get('best_test_accuracy'),
                'final_loss': run.get('final_test_loss'),
                'certainty_step': run.get('final_certainty_step'),
                'n_parameters': run.get('n_parameters'),
            }
            
            # Extract lag statistics if available
            if model_type == 'lagged' and 'lag_statistics' in run:
                for key, value in run['lag_statistics'].items():
                    record[f'lag_{key}'] = value
            
            records.append(record)
    
    return pd.DataFrame(records)


def compute_statistics(df: pd.DataFrame, metric: str) -> Dict:
    """Compute summary statistics for a metric."""
    baseline = df[df['model_type'] == 'baseline'][metric].dropna()
    lagged = df[df['model_type'] == 'lagged'][metric].dropna()
    
    return {
        'baseline_mean': baseline.mean(),
        'baseline_std': baseline.std(),
        'baseline_n': len(baseline),
        'lagged_mean': lagged.mean(),
        'lagged_std': lagged.std(),
        'lagged_n': len(lagged),
        'difference': lagged.mean() - baseline.mean(),
        'percent_change': ((lagged.mean() - baseline.mean()) / baseline.mean() * 100) if baseline.mean() != 0 else np.nan,
    }


def perform_significance_tests(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> Dict:
    """Perform statistical significance tests."""
    baseline = df[df['model_type'] == 'baseline'][metric].dropna().values
    lagged = df[df['model_type'] == 'lagged'][metric].dropna().values
    
    if len(baseline) < 2 or len(lagged) < 2:
        return {'error': 'Insufficient samples for statistical tests'}
    
    # Independent samples t-test
    t_stat, t_pvalue = stats.ttest_ind(lagged, baseline)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(lagged, baseline, alternative='two-sided')
    
    # Welch's t-test (unequal variances)
    welch_stat, welch_pvalue = stats.ttest_ind(lagged, baseline, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(baseline) - 1) * baseline.std()**2 + 
                          (len(lagged) - 1) * lagged.std()**2) / 
                         (len(baseline) + len(lagged) - 2))
    cohens_d = (lagged.mean() - baseline.mean()) / pooled_std if pooled_std > 0 else np.nan
    
    # 95% confidence interval for the difference
    se_diff = np.sqrt(baseline.var()/len(baseline) + lagged.var()/len(lagged))
    ci_low = (lagged.mean() - baseline.mean()) - 1.96 * se_diff
    ci_high = (lagged.mean() - baseline.mean()) + 1.96 * se_diff
    
    return {
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        't_significant': t_pvalue < alpha,
        'mann_whitney_u': u_stat,
        'mann_whitney_pvalue': u_pvalue,
        'mann_whitney_significant': u_pvalue < alpha,
        'welch_t_statistic': welch_stat,
        'welch_pvalue': welch_pvalue,
        'welch_significant': welch_pvalue < alpha,
        'cohens_d': cohens_d,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
    }


def plot_accuracy_comparison(df: pd.DataFrame, output_dir: str):
    """Plot accuracy comparison between models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot
    ax1 = axes[0]
    sns.boxplot(data=df, x='model_type', y='final_accuracy', ax=ax1, palette='Set2')
    sns.stripplot(data=df, x='model_type', y='final_accuracy', ax=ax1, 
                  color='black', alpha=0.5, size=8)
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Final Test Accuracy')
    ax1.set_title('Accuracy Distribution')
    ax1.set_xticklabels(['Baseline CTM', 'Lagged CTM'])
    
    # Add significance annotation
    baseline_acc = df[df['model_type'] == 'baseline']['final_accuracy']
    lagged_acc = df[df['model_type'] == 'lagged']['final_accuracy']
    _, pvalue = stats.ttest_ind(lagged_acc, baseline_acc)
    
    y_max = df['final_accuracy'].max()
    ax1.plot([0, 0, 1, 1], [y_max + 0.02, y_max + 0.03, y_max + 0.03, y_max + 0.02], 'k-', lw=1.5)
    sig_text = f'p = {pvalue:.4f}' + (' ***' if pvalue < 0.001 else ' **' if pvalue < 0.01 else ' *' if pvalue < 0.05 else ' ns')
    ax1.text(0.5, y_max + 0.035, sig_text, ha='center', va='bottom', fontsize=11)
    
    # Bar plot with error bars
    ax2 = axes[1]
    means = df.groupby('model_type')['final_accuracy'].mean()
    stds = df.groupby('model_type')['final_accuracy'].std()
    sems = stds / np.sqrt(df.groupby('model_type').size())
    
    x = np.arange(2)
    bars = ax2.bar(x, [means['baseline'], means['lagged']], 
                   yerr=[sems['baseline'], sems['lagged']],
                   capsize=5, color=['#66c2a5', '#fc8d62'], edgecolor='black', linewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Baseline CTM', 'Lagged CTM'])
    ax2.set_ylabel('Final Test Accuracy')
    ax2.set_title('Mean Accuracy (± SEM)')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, [means['baseline'], means['lagged']], 
                               [stds['baseline'], stds['lagged']]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.pdf'), bbox_inches='tight')
    plt.close()


def plot_certainty_comparison(df: pd.DataFrame, output_dir: str):
    """Plot certainty step comparison between models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot
    ax1 = axes[0]
    sns.boxplot(data=df, x='model_type', y='certainty_step', ax=ax1, palette='Set2')
    sns.stripplot(data=df, x='model_type', y='certainty_step', ax=ax1,
                  color='black', alpha=0.5, size=8)
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Mean Certainty Step')
    ax1.set_title('When Models Reach Certainty')
    ax1.set_xticklabels(['Baseline CTM', 'Lagged CTM'])
    
    # Interpretation
    ax2 = axes[1]
    baseline_cert = df[df['model_type'] == 'baseline']['certainty_step']
    lagged_cert = df[df['model_type'] == 'lagged']['certainty_step']
    
    ax2.hist(baseline_cert, bins=15, alpha=0.6, label='Baseline', color='#66c2a5')
    ax2.hist(lagged_cert, bins=15, alpha=0.6, label='Lagged', color='#fc8d62')
    ax2.axvline(baseline_cert.mean(), color='#66c2a5', linestyle='--', linewidth=2,
                label=f'Baseline μ={baseline_cert.mean():.1f}')
    ax2.axvline(lagged_cert.mean(), color='#fc8d62', linestyle='--', linewidth=2,
                label=f'Lagged μ={lagged_cert.mean():.1f}')
    ax2.set_xlabel('Certainty Step')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Certainty Steps')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'certainty_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'certainty_comparison.pdf'), bbox_inches='tight')
    plt.close()


def plot_loss_comparison(df: pd.DataFrame, output_dir: str):
    """Plot loss comparison between models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.boxplot(data=df, x='model_type', y='final_loss', ax=ax, palette='Set2')
    sns.stripplot(data=df, x='model_type', y='final_loss', ax=ax,
                  color='black', alpha=0.5, size=8)
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Final Test Loss')
    ax.set_title('Loss Distribution')
    ax.set_xticklabels(['Baseline CTM', 'Lagged CTM'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'loss_comparison.pdf'), bbox_inches='tight')
    plt.close()


def plot_lag_weights(df: pd.DataFrame, output_dir: str):
    """Plot learned lag weights for lagged models."""
    lagged_df = df[df['model_type'] == 'lagged']
    
    # Extract lag weight columns
    weight_cols = [col for col in lagged_df.columns if 'weight' in col and 'lag' in col]
    
    if not weight_cols:
        print("No lag weight data found")
        return
    
    # Organize by sync type and lag
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, sync_type in zip(axes, ['out', 'action']):
        cols = [col for col in weight_cols if sync_type in col]
        if not cols:
            continue
            
        # Extract lag numbers and sort
        lag_data = {}
        for col in cols:
            # Parse lag number from column name like 'lag_out_lag_0_weight'
            parts = col.split('_')
            for i, part in enumerate(parts):
                if part == 'lag' and i + 1 < len(parts):
                    try:
                        lag_num = int(parts[i + 1])
                        lag_data[lag_num] = lagged_df[col].dropna().values
                        break
                    except ValueError:
                        continue
        
        if not lag_data:
            continue
            
        lags = sorted(lag_data.keys())
        means = [np.mean(lag_data[l]) for l in lags]
        stds = [np.std(lag_data[l]) for l in lags]
        
        bars = ax.bar(range(len(lags)), means, yerr=stds, capsize=5,
                      color=plt.cm.viridis(np.linspace(0.2, 0.8, len(lags))),
                      edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels([f'Lag {l}' for l in lags])
        ax.set_ylabel('Effective Weight (exp(-decay))')
        ax.set_title(f'{sync_type.capitalize()} Synchronization Lag Weights')
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lag_weights.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'lag_weights.pdf'), bbox_inches='tight')
    plt.close()


def plot_learning_curves(results: Dict[str, List[dict]], output_dir: str):
    """Plot aggregated learning curves from training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('test_accuracy', 'Test Accuracy', axes[0, 0]),
        ('test_loss', 'Test Loss', axes[0, 1]),
        ('test_certainty_step', 'Certainty Step', axes[1, 0]),
        ('train_accuracy', 'Train Accuracy', axes[1, 1]),
    ]
    
    colors = {'baseline': '#66c2a5', 'lagged': '#fc8d62'}
    
    for metric_key, metric_name, ax in metrics:
        for model_type in ['baseline', 'lagged']:
            all_curves = []
            
            for run in results[model_type]:
                if 'training_history' not in run:
                    continue
                    
                history = run['training_history']
                if isinstance(history, list):
                    steps = [h['step'] for h in history]
                    values = [h.get(metric_key, np.nan) for h in history]
                    all_curves.append((steps, values))
            
            if not all_curves:
                continue
            
            # Interpolate to common x-axis
            max_step = max(max(s) for s, _ in all_curves)
            common_steps = np.linspace(0, max_step, 200)
            
            interpolated = []
            for steps, values in all_curves:
                interp_values = np.interp(common_steps, steps, values)
                interpolated.append(interp_values)
            
            interpolated = np.array(interpolated)
            mean_curve = np.mean(interpolated, axis=0)
            std_curve = np.std(interpolated, axis=0)
            
            ax.plot(common_steps, mean_curve, color=colors[model_type], 
                   label=f'{model_type.capitalize()}', linewidth=2)
            ax.fill_between(common_steps, mean_curve - std_curve, mean_curve + std_curve,
                           color=colors[model_type], alpha=0.2)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} During Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'learning_curves.pdf'), bbox_inches='tight')
    plt.close()


def generate_summary_table(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Generate a summary table of all metrics with statistics."""
    metrics = ['final_accuracy', 'best_accuracy', 'final_loss', 'certainty_step']
    
    rows = []
    for metric in metrics:
        stats_dict = compute_statistics(df, metric)
        sig_tests = perform_significance_tests(df, metric, alpha)
        
        row = {
            'Metric': metric.replace('_', ' ').title(),
            'Baseline Mean': f"{stats_dict['baseline_mean']:.4f}",
            'Baseline Std': f"{stats_dict['baseline_std']:.4f}",
            'Lagged Mean': f"{stats_dict['lagged_mean']:.4f}",
            'Lagged Std': f"{stats_dict['lagged_std']:.4f}",
            'Difference': f"{stats_dict['difference']:.4f}",
            '% Change': f"{stats_dict['percent_change']:.2f}%",
            'p-value (t-test)': f"{sig_tests.get('t_pvalue', np.nan):.4f}",
            "Cohen's d": f"{sig_tests.get('cohens_d', np.nan):.3f}",
            'Significant': '✓' if sig_tests.get('t_significant', False) else '✗',
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_full_report(df: pd.DataFrame, results: Dict, output_dir: str, alpha: float):
    """Save a comprehensive text report."""
    report_path = os.path.join(output_dir, 'statistical_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CTM VALIDATION EXPERIMENT - STATISTICAL REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Sample sizes
        f.write("SAMPLE SIZES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Baseline CTM runs: {len(results['baseline'])}\n")
        f.write(f"Lagged CTM runs: {len(results['lagged'])}\n\n")
        
        # Main metrics
        for metric in ['final_accuracy', 'best_accuracy', 'final_loss', 'certainty_step']:
            f.write(f"\n{metric.upper().replace('_', ' ')}\n")
            f.write("-" * 40 + "\n")
            
            stats_dict = compute_statistics(df, metric)
            sig_tests = perform_significance_tests(df, metric, alpha)
            
            f.write(f"Baseline: {stats_dict['baseline_mean']:.4f} ± {stats_dict['baseline_std']:.4f} (n={stats_dict['baseline_n']})\n")
            f.write(f"Lagged:   {stats_dict['lagged_mean']:.4f} ± {stats_dict['lagged_std']:.4f} (n={stats_dict['lagged_n']})\n")
            f.write(f"Difference: {stats_dict['difference']:.4f} ({stats_dict['percent_change']:.2f}%)\n\n")
            
            f.write("Statistical Tests:\n")
            f.write(f"  Independent t-test: t={sig_tests.get('t_statistic', np.nan):.3f}, p={sig_tests.get('t_pvalue', np.nan):.6f}\n")
            f.write(f"  Welch's t-test: t={sig_tests.get('welch_t_statistic', np.nan):.3f}, p={sig_tests.get('welch_pvalue', np.nan):.6f}\n")
            f.write(f"  Mann-Whitney U: U={sig_tests.get('mann_whitney_u', np.nan):.1f}, p={sig_tests.get('mann_whitney_pvalue', np.nan):.6f}\n")
            f.write(f"  Effect size (Cohen's d): {sig_tests.get('cohens_d', np.nan):.3f}\n")
            f.write(f"  95% CI for difference: [{sig_tests.get('ci_95_low', np.nan):.4f}, {sig_tests.get('ci_95_high', np.nan):.4f}]\n")
            
            sig_status = "SIGNIFICANT" if sig_tests.get('t_significant', False) else "NOT SIGNIFICANT"
            f.write(f"  Result: {sig_status} at α={alpha}\n")
        
        # Lag statistics summary
        lagged_runs = [r for r in results['lagged'] if 'lag_statistics' in r]
        if lagged_runs:
            f.write("\n\nLAG WEIGHT STATISTICS (Lagged CTM only)\n")
            f.write("-" * 40 + "\n")
            
            all_lag_stats = defaultdict(list)
            for run in lagged_runs:
                for key, value in run['lag_statistics'].items():
                    all_lag_stats[key].append(value)
            
            for key, values in sorted(all_lag_stats.items()):
                f.write(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Full report saved to: {report_path}")


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    if args.use_wandb and WANDB_AVAILABLE:
        results = load_wandb_results(args.wandb_project, args.experiment_name)
    else:
        results = load_local_results(args.checkpoint_dir)
    
    if not results['baseline'] and not results['lagged']:
        print("No results found! Check your checkpoint directory or wandb settings.")
        return
    
    # Extract metrics to DataFrame
    df = extract_metrics(results)
    df.to_csv(os.path.join(args.output_dir, 'metrics_summary.csv'), index=False)
    print(f"\nMetrics saved to: {os.path.join(args.output_dir, 'metrics_summary.csv')}")
    
    # Generate summary table
    summary_table = generate_summary_table(df, args.significance_level)
    summary_table.to_csv(os.path.join(args.output_dir, 'statistical_summary.csv'), index=False)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(summary_table.to_string(index=False))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(df, args.output_dir)
    plot_certainty_comparison(df, args.output_dir)
    plot_loss_comparison(df, args.output_dir)
    plot_lag_weights(df, args.output_dir)
    plot_learning_curves(results, args.output_dir)
    
    # Save full report
    save_full_report(df, results, args.output_dir, args.significance_level)
    
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("\nKey findings:")
    
    # Print key findings
    acc_stats = compute_statistics(df, 'final_accuracy')
    acc_sig = perform_significance_tests(df, 'final_accuracy', args.significance_level)
    
    print(f"  - Accuracy improvement: {acc_stats['difference']*100:.2f}% ({acc_stats['percent_change']:.1f}% relative)")
    print(f"  - Statistical significance: p={acc_sig.get('t_pvalue', np.nan):.6f}")
    print(f"  - Effect size (Cohen's d): {acc_sig.get('cohens_d', np.nan):.3f}")
    
    cert_stats = compute_statistics(df, 'certainty_step')
    print(f"  - Certainty step difference: {cert_stats['difference']:.2f} steps")


if __name__ == "__main__":
    main()