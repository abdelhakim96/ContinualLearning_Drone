"""
BlueROV2 Continual Learning — Main Entry Point

Usage:
    python main.py [--mode MODE]

Modes
-----
  data      Generate training dataset
  train     Pre-train base inverse MLP
  run       Run comparison experiments
  plot      Generate publication figures
  all       Run full pipeline end-to-end (default)
"""

import argparse
import os
import sys

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def run_data():
    print("\n[1/4] Generating dataset …")
    from bluerov.training.generate_dataset import generate_dataset, SAVE_PATH
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df = generate_dataset(n_episodes=40, episode_len=500)
    df.to_csv(SAVE_PATH, index=False)
    print(f"Dataset: {len(df)} samples → {SAVE_PATH}")


def run_train():
    print("\n[2/4] Training base inverse MLP …")
    from bluerov.training.train_inverse import train
    train()


def run_experiments():
    print("\n[3/4] Running comparison experiments …")
    from bluerov.experiments.run_comparison import run_all
    run_all()


def run_plot():
    print("\n[4/4] Generating figures …")
    from bluerov.utils.plot_results import generate_all_figures
    generate_all_figures()


def main():
    parser = argparse.ArgumentParser(
        description='BlueROV2 LoRA Continual Learning Pipeline')
    parser.add_argument('--mode', default='all',
                        choices=['data', 'train', 'run', 'plot', 'all'],
                        help='Pipeline stage to execute (default: all)')
    args = parser.parse_args()

    if args.mode in ('data', 'all'):
        run_data()
    if args.mode in ('train', 'all'):
        run_train()
    if args.mode in ('run', 'all'):
        run_experiments()
    if args.mode in ('plot', 'all'):
        run_plot()


if __name__ == '__main__':
    main()
