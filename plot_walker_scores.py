#!/usr/bin/env python3
"""Plot episode scores for Walker task across different encoders."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# Define log directories (latest runs)
LOGDIR = Path("/mnt/home/tianyuez/DL-PJ/dreamerv3/logdir")
ENCODERS = {
    'CNN + AE': 'dmc_walker_walk_cnn_ae_20251206_024217',
    'CNN + MAE': 'dmc_walker_walk_cnn_mae_20251206_024217',
    'ViT + AE': 'dmc_walker_walk_vit_ae_20251206_024217',
    'ViT + MAE': 'dmc_walker_walk_vit_mae_20251206_024217',
}

# Colors for each encoder
COLORS = {
    'CNN + AE': '#2ecc71',   # Green
    'CNN + MAE': '#3498db',  # Blue
    'ViT + AE': '#e74c3c',   # Red
    'ViT + MAE': '#9b59b6',  # Purple
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

def smooth(data, window=50):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot raw and smoothed data
    for encoder_name, run_dir in ENCODERS.items():
        scores_file = LOGDIR / run_dir / 'scores.jsonl'
        if not scores_file.exists():
            print(f"Warning: {scores_file} not found, skipping...")
            continue
        
        steps, scores = load_scores(scores_file)
        color = COLORS[encoder_name]
        
        # Raw data (with transparency)
        ax1.plot(steps, scores, alpha=0.3, color=color, linewidth=0.5)
        
        # Smoothed data
        window = 50
        if len(scores) >= window:
            smoothed_scores = smooth(scores, window)
            # Adjust steps to match smoothed data length
            smoothed_steps = steps[window-1:][:len(smoothed_scores)]
            ax1.plot(smoothed_steps, smoothed_scores, label=encoder_name, 
                    color=color, linewidth=2)
            ax2.plot(smoothed_steps, smoothed_scores, label=encoder_name,
                    color=color, linewidth=2.5)
    
    # Configure left plot (raw + smoothed)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Episode Score')
    ax1.set_title('Walker Walk - Episode Score (Raw + Smoothed)')
    ax1.legend(loc='upper left')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Configure right plot (smoothed only)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Episode Score')
    ax2.set_title('Walker Walk - Episode Score (Smoothed, window=50)')
    ax2.legend(loc='upper left')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    plt.suptitle('DreamerV3 Encoder Ablation Study - Walker Walk Task', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = LOGDIR.parent / 'walker_scores_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")
    
    # Also create a summary table
    print("\n" + "="*60)
    print("Summary Statistics (Final 100 episodes)")
    print("="*60)
    print(f"{'Encoder':<15} {'Mean Score':>12} {'Max Score':>12} {'Std':>10}")
    print("-"*60)
    
    for encoder_name, run_dir in ENCODERS.items():
        scores_file = LOGDIR / run_dir / 'scores.jsonl'
        if scores_file.exists():
            _, scores = load_scores(scores_file)
            final_scores = scores[-100:] if len(scores) >= 100 else scores
            print(f"{encoder_name:<15} {np.mean(final_scores):>12.2f} "
                  f"{np.max(final_scores):>12.2f} {np.std(final_scores):>10.2f}")
    print("="*60)

if __name__ == '__main__':
    main()


