#!/usr/bin/env python3
"""Adapt DreamerV3 training logs for plotting and scientific comparison.

This script automates three tasks:
1. Scans existing log directories for `scores.jsonl`.
2. Computes summary statistics per task/encoder, optionally comparing against
   the official reference score dumps (see `scores/view.py`).
3. Prepares a directory layout that matches the naming assumptions of
   `plot.py` and, if requested, calls `plot.py` directly to generate figures.

Example:
  python analyze_runs.py \
      --runs-root logdir \
      --reference scores/dmc_vision-dreamerv3.json.gz \
      --reference-method dreamerv3 \
      --outdir analysis \
      --plot
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ruamel.yaml as ryaml

import elements
import plot as plot_module
import types


def parse_args():
  parser = argparse.ArgumentParser(
      description='Analyze DreamerV3 runs and compare to reference scores.')
  parser.add_argument(
      '--runs-root', nargs='+', default=['logdir'],
      help='Root directories that contain subfolders with scores.jsonl.')
  parser.add_argument(
      '--outdir', default='analysis',
      help='Where to store summaries, plot-ready symlinks, and figures.')
  parser.add_argument(
      '--reference', default='',
      help='Optional path to *.json or *.json.gz score dump to compare against.')
  parser.add_argument(
      '--reference-method', default='dreamerv3',
      help='Method label inside the reference score dump to compare against.')
  parser.add_argument(
      '--tail', type=int, default=100,
      help='Window size for computing the mean of the last N evaluation scores.')
  parser.add_argument(
      '--plot', action='store_true',
      help='If set, prepare symlinks and invoke plot.py to render curves.')
  parser.add_argument(
      '--plot-stats', default='runs,auto',
      help='Comma separated list of stats to forward to plot.py (see plot.py).')
  parser.add_argument(
      '--tasks', default='.*',
      help='Regex of tasks to keep when plotting.')
  parser.add_argument(
      '--methods', default='.*',
      help='Regex of method names to keep when plotting.')
  parser.add_argument(
      '--bins', type=int, default=30,
      help='Number of temporal bins for plot.py aggregation.')
  parser.add_argument(
      '--workers', type=int, default=16,
      help='Number of loader workers to forward to plot.py.')
  parser.add_argument(
      '--plot-xlim', type=float, default=0,
      help='Optional x-axis limit in environment steps for plot.py.')
  parser.add_argument(
      '--plot-ylim', type=float, default=0,
      help='Optional y-axis limit for plot.py.')
  parser.add_argument(
      '--plot-binsize', type=float, default=0,
      help='Fixed bin size for plot.py (overrides --bins when >0).')
  parser.add_argument(
      '--plot-raw', action='store_true',
      help='Disable aggregation in plot.py (shows raw per-seed curves).')
  return parser.parse_args()


def read_config(run_dir):
  config_path = run_dir / 'config.yaml'
  if not config_path.exists():
    return {}
  data = ryaml.YAML(typ='safe').load(config_path.read_text())
  return data or {}


def summarize_scores(scores_path, tail):
  steps, scores = [], []
  with scores_path.open() as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      record = json.loads(line)
      steps.append(record['step'])
      scores.append(record['episode/score'])
  if not scores:
    return None
  tail_values = scores[-tail:] if len(scores) >= tail else scores
  return dict(
      num_points=len(scores),
      step_start=steps[0],
      step_end=steps[-1],
      score_last=scores[-1],
      score_best=max(scores),
      score_mean_last=float(np.mean(tail_values)),
  )


def gather_runs(roots, tail):
  runs = []
  for root in roots:
    root = Path(root).expanduser().resolve()
    if not root.exists():
      continue
    for scores_path in root.rglob('scores.jsonl'):
      scores_path = scores_path.resolve()
      run_dir = scores_path.parent
      config = read_config(run_dir)
      task = config.get('task')
      encoder = config.get('encoder_type', 'cnn_ae')
      seed = config.get('seed', 0)
      summary = summarize_scores(scores_path, tail)
      if not summary or not task:
        continue
      runs.append({
          'task': task,
          'arch': encoder,
          'seed': seed,
          'run_dir': run_dir,
          'scores_path': scores_path,
          **summary,
      })
  return sorted(runs, key=lambda x: (x['task'], x['arch'], x['seed']))


def load_reference(reference_path, method):
  reference = {}
  if not reference_path:
    return reference
  reference_path = Path(reference_path)
  if not reference_path.exists():
    return reference
  df = pd.read_json(reference_path)
  df = df[df['method'] == method]
  if df.empty:
    return reference
  for (task, _), group in df.groupby(['task', 'method']):
    finals = [row[-1] for row in group['ys']]
    if not finals:
      continue
    reference[task] = dict(
        mean=float(np.mean(finals)),
        std=float(np.std(finals)),
        count=len(finals),
    )
  return reference


def write_summary(runs, reference, outdir):
  outdir.mkdir(parents=True, exist_ok=True)
  csv_path = outdir / 'summary.csv'
  fieldnames = [
      'task', 'arch', 'seed', 'num_points', 'step_start', 'step_end',
      'score_last', 'score_mean_last', 'score_best',
      'reference_mean', 'reference_std', 'reference_count', 'delta',
  ]
  with csv_path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for run in runs:
      ref_stats = reference.get(run['task'], {})
      delta = (
          run['score_mean_last'] - ref_stats['mean']
          if ref_stats else None)
      row = {
          **{k: run.get(k) for k in fieldnames if k in run},
          'reference_mean': ref_stats.get('mean'),
          'reference_std': ref_stats.get('std'),
          'reference_count': ref_stats.get('count'),
          'delta': delta,
      }
      writer.writerow(row)
  return csv_path


def print_table(runs, reference):
  if not runs:
    print('No runs found.')
    return
  header = (
      f'{"Task":20} {"Arch":10} {"Seed":4} '
      f'{"Last100":>10} {"Best":>10} {"RefMean":>10} {"Delta":>10}')
  print(header)
  print('-' * len(header))
  for run in runs:
    ref = reference.get(run['task'])
    delta = (run['score_mean_last'] - ref['mean']) if ref else float('nan')
    ref_mean = ref['mean'] if ref else float('nan')
    print(
        f'{run["task"]:20} {run["arch"]:10} {run["seed"]:4d} '
        f'{run["score_mean_last"]:10.2f} {run["score_best"]:10.2f} '
        f'{ref_mean:10.2f} {delta:10.2f}')


def prepare_plot_symlinks(runs, dest):
  dest.mkdir(parents=True, exist_ok=True)
  created = []
  for idx, run in enumerate(runs):
    seed_label = f'seed{run["seed"]}v{idx}'
    folder = dest / f'adapt-{run["task"]}-{run["arch"]}-{seed_label}'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'scores.jsonl'
    if target.exists() or target.is_symlink():
      target.unlink()
    os.symlink(run['scores_path'], target)
    created.append(folder)
  return created


def run_plot(prepared_dirs, outdir, args):
  stats = [s.strip() for s in args.plot_stats.split(',') if s.strip()]
  plot_config = types.SimpleNamespace(
      pattern='**/scores.jsonl',
      indirs=[str(prepared_dirs)],
      outdir=str(outdir),
      methods=args.methods,
      tasks=args.tasks,
      newstyle=True,
      indir_prefix=False,
      workers=args.workers,
      xkeys=['xs', 'step'],
      ykeys=['ys', 'episode/score'],
      ythres=0.0,
      xlim=args.plot_xlim,
      ylim=args.plot_ylim,
      binsize=args.plot_binsize,
      bins=args.bins,
      cols=0,
      legendcols=0,
      size=[3, 3],
      xticks=4,
      yticks=10,
      stats=stats or ['runs', 'auto'],
      agg=not args.plot_raw,
      todf='',
  )
  plot_module.main(plot_config)


def main():
  args = parse_args()
  runs = gather_runs(args.runs_root, args.tail)
  reference = load_reference(args.reference, args.reference_method)
  outdir = Path(args.outdir)
  csv_path = write_summary(runs, reference, outdir)
  print(f'Wrote summary to {csv_path}')
  print_table(runs, reference)
  if args.plot and runs:
    plot_root = outdir / 'plot_ready'
    prepare_plot_symlinks(runs, plot_root)
    figures_dir = outdir / 'figures'
    run_plot(plot_root, figures_dir, args)
    print(f'Plot saved under {figures_dir}')


if __name__ == '__main__':
  sys.exit(main())
