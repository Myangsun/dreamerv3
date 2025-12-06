#!/usr/bin/env python3
"""Plot episode scores for all DMC tasks across different encoders."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

LOGDIR = Path("/mnt/home/tianyuez/DL-PJ/dreamerv3/logdir")

# Task configurations
TASKS = {
    'Walker Walk': {
        'CNN + AE': 'dmc_walker_walk_cnn_ae_20251206_024217',
        'CNN + MAE': 'dmc_walker_walk_cnn_mae_20251206_024217',
        'ViT + AE': 'dmc_walker_walk_vit_ae_20251206_024217',
        'ViT + MAE': 'dmc_walker_walk_vit_mae_20251206_024217',
    },
    'Cheetah Run': {
        'CNN + AE': 'dmc_cheetah_run_cnn_ae_20251206_024217',
        'CNN + MAE': 'dmc_cheetah_run_cnn_mae_20251206_024217',
        'ViT + AE': 'dmc_cheetah_run_vit_ae_20251206_084243',
        'ViT + MAE': 'dmc_cheetah_run_vit_mae_20251206_084243',
    },
    'Hopper Hop': {
        'CNN + AE': 'dmc_hopper_hop_cnn_ae_20251206_084243',
        'CNN + MAE': 'dmc_hopper_hop_cnn_mae_20251206_084243',
        'ViT + AE': 'dmc_hopper_hop_vit_ae_20251206_084243',
        'ViT + MAE': 'dmc_hopper_hop_vit_mae_20251206_084243',
    },
}

# Colors for each encoder
COLORS = {
    'CNN + AE': '#27ae60',   # Green
    'CNN + MAE': '#3498db',  # Blue
    'ViT + AE': '#e74c3c',   # Red
    'ViT + MAE': '#9b59b6',  # Purple
}

LINESTYLES = {
    'CNN + AE': '-',
    'CNN + MAE': '-',
    'ViT + AE': '--',
    'ViT + MAE': '--',
}

def load_scores(scores_file):
    """Load scores from a JSONL file."""
    steps = []
    scores = []
    with open(scores_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            steps.append(data['step'])
            scores.append(data['episode/score'])
    return np.array(steps), np.array(scores)

def smooth(data, window=30):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Summary statistics storage
    all_stats = {}
    
    for task_idx, (task_name, encoders) in enumerate(TASKS.items()):
        ax_raw = axes[0, task_idx]
        ax_smooth = axes[1, task_idx]
        all_stats[task_name] = {}
        
        for encoder_name, run_dir in encoders.items():
            scores_file = LOGDIR / run_dir / 'scores.jsonl'
            if not scores_file.exists():
                print(f"Warning: {scores_file} not found, skipping...")
                continue
            
            steps, scores = load_scores(scores_file)
            color = COLORS[encoder_name]
            linestyle = LINESTYLES[encoder_name]
            
            # Store stats
            final_scores = scores[-100:] if len(scores) >= 100 else scores
            all_stats[task_name][encoder_name] = {
                'mean': np.mean(final_scores),
                'max': np.max(final_scores),
                'std': np.std(final_scores),
                'final_step': steps[-1] if len(steps) > 0 else 0,
            }
            
            # Raw data plot
            ax_raw.plot(steps, scores, alpha=0.6, color=color, 
                       linewidth=0.8, linestyle=linestyle, label=encoder_name)
            
            # Smoothed data plot
            window = 30
            if len(scores) >= window:
                smoothed_scores = smooth(scores, window)
                smoothed_steps = steps[window-1:][:len(smoothed_scores)]
                ax_smooth.plot(smoothed_steps, smoothed_scores, 
                              color=color, linewidth=2.5, linestyle=linestyle,
                              label=encoder_name)
        
        # Configure raw plot
        ax_raw.set_title(f'{task_name} (Raw)', fontweight='bold')
        ax_raw.set_xlabel('Training Steps')
        ax_raw.set_ylabel('Episode Score')
        ax_raw.legend(loc='upper left', framealpha=0.9)
        ax_raw.set_xlim(left=0)
        ax_raw.set_ylim(bottom=0)
        ax_raw.grid(True, alpha=0.3)
        
        # Configure smoothed plot
        ax_smooth.set_title(f'{task_name} (Smoothed)', fontweight='bold')
        ax_smooth.set_xlabel('Training Steps')
        ax_smooth.set_ylabel('Episode Score')
        ax_smooth.legend(loc='upper left', framealpha=0.9)
        ax_smooth.set_xlim(left=0)
        ax_smooth.set_ylim(bottom=0)
        ax_smooth.grid(True, alpha=0.3)
    
    plt.suptitle('DreamerV3 Encoder Ablation Study - DMC Tasks\n(CNN vs ViT, AE vs MAE)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = LOGDIR.parent / 'all_tasks_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved plot to: {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS - Final 100 Episodes Statistics")
    print("="*80)
    
    for task_name, encoders in all_stats.items():
        print(f"\n📊 {task_name}")
        print("-"*70)
        print(f"{'Encoder':<15} {'Mean Score':>12} {'Max Score':>12} {'Std':>10} {'Steps':>12}")
        print("-"*70)
        
        # Sort by mean score
        sorted_encoders = sorted(encoders.items(), key=lambda x: x[1]['mean'], reverse=True)
        for i, (enc_name, stats) in enumerate(sorted_encoders):
            medal = ['🥇', '🥈', '🥉', '  '][i] if i < 4 else '  '
            print(f"{medal} {enc_name:<12} {stats['mean']:>12.2f} {stats['max']:>12.2f} "
                  f"{stats['std']:>10.2f} {stats['final_step']:>12,}")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL ENCODER RANKING (Average across tasks)")
    print("="*80)
    
    encoder_totals = {}
    for task_name, encoders in all_stats.items():
        for enc_name, stats in encoders.items():
            if enc_name not in encoder_totals:
                encoder_totals[enc_name] = []
            encoder_totals[enc_name].append(stats['mean'])
    
    encoder_avg = {k: np.mean(v) for k, v in encoder_totals.items()}
    sorted_overall = sorted(encoder_avg.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Rank':<6} {'Encoder':<15} {'Average Score':>15}")
    print("-"*40)
    for i, (enc_name, avg_score) in enumerate(sorted_overall):
        medal = ['🥇', '🥈', '🥉', '4th'][i]
        print(f"{medal:<6} {enc_name:<15} {avg_score:>15.2f}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

