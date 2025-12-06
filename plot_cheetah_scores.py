#!/usr/bin/env python3
"""Plot episode scores for Cheetah task across different encoders."""

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

# Define log directories (latest runs for each encoder)
LOGDIR = Path("/mnt/home/tianyuez/DL-PJ/dreamerv3/logdir")
ENCODERS = {
    'CNN + AE': 'dmc_cheetah_run_cnn_ae_20251206_024217',
    'CNN + MAE': 'dmc_cheetah_run_cnn_mae_20251206_024217',
    'ViT + AE': 'dmc_cheetah_run_vit_ae_20251206_084243',
    'ViT + MAE': 'dmc_cheetah_run_vit_mae_20251206_084243',
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
        window = max(1, len(data) // 5)  # Use smaller window for limited data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    stats = {}
    
    # Plot raw and smoothed data
    for encoder_name, run_dir in ENCODERS.items():
        scores_file = LOGDIR / run_dir / 'scores.jsonl'
        if not scores_file.exists():
            print(f"Warning: {scores_file} not found, skipping...")
            continue
        
        steps, scores = load_scores(scores_file)
        if len(scores) == 0:
            continue
            
        color = COLORS[encoder_name]
        
        # Raw data (with transparency)
        ax1.plot(steps, scores, alpha=0.3, color=color, linewidth=0.5)
        
        # Smoothed data
        window = min(50, max(5, len(scores) // 10))
        smoothed_scores = smooth(scores, window)
        smoothed_steps = steps[window-1:][:len(smoothed_scores)]
        
        # Check if still training
        status = " (training)" if len(scores) < 400 else ""
        label = f"{encoder_name}{status}"
        
        ax1.plot(smoothed_steps, smoothed_scores, label=label, 
                color=color, linewidth=2)
        ax2.plot(smoothed_steps, smoothed_scores, label=label,
                color=color, linewidth=2.5)
        
        # Store stats
        final_n = min(100, len(scores))
        final_scores = scores[-final_n:]
        stats[encoder_name] = {
            'episodes': len(scores),
            'mean': np.mean(final_scores),
            'max': np.max(final_scores),
            'std': np.std(final_scores),
            'final_step': steps[-1] if len(steps) > 0 else 0
        }
    
    # Configure left plot (raw + smoothed)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Episode Score')
    ax1.set_title('Cheetah Run - Episode Score (Raw + Smoothed)')
    ax1.legend(loc='upper left')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Configure right plot (smoothed only)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Episode Score')
    ax2.set_title('Cheetah Run - Episode Score (Smoothed)')
    ax2.legend(loc='upper left')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    plt.suptitle('DreamerV3 Encoder Ablation Study - Cheetah Run Task', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = LOGDIR.parent / 'cheetah_scores_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("Summary Statistics (Final episodes, up to 100)")
    print("="*70)
    print(f"{'Encoder':<20} {'Episodes':>10} {'Mean Score':>12} {'Max Score':>12} {'Std':>10}")
    print("-"*70)
    
    for encoder_name, s in stats.items():
        status = "*" if s['episodes'] < 400 else " "
        print(f"{encoder_name:<20} {s['episodes']:>10}{status} {s['mean']:>12.2f} "
              f"{s['max']:>12.2f} {s['std']:>10.2f}")
    print("-"*70)
    print("* = Still training")
    print("="*70)

if __name__ == '__main__':
    main()


