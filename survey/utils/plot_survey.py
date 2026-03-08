"""
Publication-quality figures for the data-driven dynamics survey.

Generates:
  1. Spider / radar diagram comparing methods across metrics
  2. Bar charts for one-step RMSE (clean vs disturbed)
  3. Rollout comparison plots
  4. MPC tracking comparison
  5. Computation time comparison
  6. LaTeX tables
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

COLORS = {
    'Classic': '#2196F3',
    'NN':      '#FF9800',
    'PINN':    '#4CAF50',
    'SINDy':   '#9C27B0',
    'DMD':     '#F44336',
    'GP':      '#00BCD4',
    'Hankel':  '#795548',
}

METHOD_ORDER = ['Classic', 'NN', 'PINN', 'SINDy', 'DMD', 'GP', 'Hankel']


def _setup():
    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11,
        'axes.titlesize': 12, 'legend.fontsize': 9,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    })
    os.makedirs(FIG_DIR, exist_ok=True)


def fig_spider(summary_df):
    """Radar/spider diagram comparing methods across normalised metrics."""
    _setup()

    metrics = ['rmse_pos', 'rmse_vel', 'rmse_pos_dist',
               'rollout_pos_rmse', 'pred_time_ms', 'train_time_s']
    labels = ['Pos. RMSE\n(clean)', 'Vel. RMSE\n(clean)',
              'Pos. RMSE\n(disturbed)', 'Rollout\nRMSE',
              'Pred. Time\n(ms)', 'Train Time\n(s)']

    # Only use metrics that exist
    available = [m for m in metrics if m in summary_df.columns]
    labels = [labels[metrics.index(m)] for m in available]

    df = summary_df.set_index('method')
    # Normalise each metric to [0, 1] (lower is better → invert)
    vals = {}
    for meth in METHOD_ORDER:
        if meth not in df.index:
            continue
        raw = [df.loc[meth, m] if m in df.columns else 0 for m in available]
        vals[meth] = raw

    # Normalize: for each metric, 0=best, 1=worst
    arr = np.array(list(vals.values()))
    mins = arr.min(0); maxs = arr.max(0)
    rng = maxs - mins; rng[rng == 0] = 1

    N = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for meth, raw in vals.items():
        norm = (np.array(raw) - mins) / rng
        norm = np.append(norm, norm[0])
        ax.plot(angles, norm, 'o-', label=meth, color=COLORS.get(meth, 'gray'),
                linewidth=2, markersize=5)
        ax.fill(angles, norm, alpha=0.08, color=COLORS.get(meth, 'gray'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_title('Method Comparison (normalised, lower = better)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    fig.savefig(os.path.join(FIG_DIR, 'fig_spider.pdf'))
    plt.close(fig)
    print("  Saved fig_spider.pdf")


def fig_bar_onestep(summary_df):
    """Grouped bar chart: one-step RMSE clean vs disturbed."""
    _setup()
    df = summary_df.set_index('method').loc[
        [m for m in METHOD_ORDER if m in summary_df['method'].values]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Position RMSE
    ax = axes[0]
    x = np.arange(len(df))
    w = 0.35
    bars1 = ax.bar(x - w/2, df['rmse_pos'], w, label='Clean',
                   color=[COLORS.get(m, 'gray') for m in df.index], alpha=0.85)
    if 'rmse_pos_dist' in df.columns:
        bars2 = ax.bar(x + w/2, df['rmse_pos_dist'], w, label='Disturbed',
                       color=[COLORS.get(m, 'gray') for m in df.index], alpha=0.45,
                       hatch='//')
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=30)
    ax.set_ylabel('Position RMSE'); ax.set_title('One-Step Position Prediction')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Velocity RMSE
    ax = axes[1]
    ax.bar(x - w/2, df['rmse_vel'], w, label='Clean',
           color=[COLORS.get(m, 'gray') for m in df.index], alpha=0.85)
    if 'rmse_vel_dist' in df.columns:
        ax.bar(x + w/2, df['rmse_vel_dist'], w, label='Disturbed',
               color=[COLORS.get(m, 'gray') for m in df.index], alpha=0.45,
               hatch='//')
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=30)
    ax.set_ylabel('Velocity RMSE'); ax.set_title('One-Step Velocity Prediction')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_bar_onestep.pdf'))
    plt.close(fig)
    print("  Saved fig_bar_onestep.pdf")


def fig_rollout(summary_df):
    """Rollout RMSE bar chart."""
    _setup()
    if 'rollout_pos_rmse' not in summary_df.columns:
        return
    df = summary_df.set_index('method').loc[
        [m for m in METHOD_ORDER if m in summary_df['method'].values]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    colors = [COLORS.get(m, 'gray') for m in df.index]
    ax.bar(x, df['rollout_pos_rmse'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=30)
    ax.set_ylabel('Position RMSE (m)')
    ax.set_title('200-Step Open-Loop Rollout Error')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_rollout.pdf'))
    plt.close(fig)
    print("  Saved fig_rollout.pdf")


def fig_timing(summary_df):
    """Computation time comparison."""
    _setup()
    df = summary_df.set_index('method').loc[
        [m for m in METHOD_ORDER if m in summary_df['method'].values]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(df))
    colors = [COLORS.get(m, 'gray') for m in df.index]

    ax = axes[0]
    ax.bar(x, df['train_time_s'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=30)
    ax.set_ylabel('Time (s)'); ax.set_title('Training Time')
    ax.set_yscale('log'); ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    ax.bar(x, df['pred_time_ms'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=30)
    ax.set_ylabel('Time (ms)'); ax.set_title('Prediction Time (per sample)')
    ax.set_yscale('log'); ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_timing.pdf'))
    plt.close(fig)
    print("  Saved fig_timing.pdf")


def fig_mpc_tracking():
    """MPC tracking trajectories for each model."""
    _setup()
    mpc_file = os.path.join(RESULTS_DIR, 'mpc_results.csv')
    if not os.path.exists(mpc_file):
        print("  Skipping MPC tracking plot (no mpc_results.csv)")
        return

    mpc_df = pd.read_csv(mpc_file)
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(mpc_df))
    colors = [COLORS.get(m, 'gray') for m in mpc_df['method']]
    ax.bar(x, mpc_df['mpc_pos_rmse'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(mpc_df['method'], rotation=30)
    ax.set_ylabel('Position RMSE (m)')
    ax.set_title('MPC Tracking Performance (30s lemniscate)')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_mpc_tracking.pdf'))
    plt.close(fig)
    print("  Saved fig_mpc_tracking.pdf")


def fig_robustness_heatmap(summary_df):
    """Heatmap showing degradation from clean to disturbed."""
    _setup()
    if 'rmse_pos_dist' not in summary_df.columns:
        return

    df = summary_df.set_index('method').loc[
        [m for m in METHOD_ORDER if m in summary_df['method'].values]]

    # Degradation ratio
    degrad = df['rmse_pos_dist'] / (df['rmse_pos'] + 1e-10)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    colors_bar = ['#4CAF50' if d < 2 else '#FF9800' if d < 5 else '#F44336'
                  for d in degrad]
    ax.barh(x, degrad, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.set_yticks(x); ax.set_yticklabels(df.index)
    ax.set_xlabel('Degradation Ratio (disturbed / clean)')
    ax.set_title('Robustness: Prediction Degradation Under Disturbance')
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_robustness.pdf'))
    plt.close(fig)
    print("  Saved fig_robustness.pdf")


def generate_latex_tables(summary_df):
    """Generate LaTeX tables for the paper."""
    _setup()
    df = summary_df.set_index('method').loc[
        [m for m in METHOD_ORDER if m in summary_df['method'].values]]

    # One-step prediction table
    cols_1step = {
        'rmse_pos': 'Pos.\\ RMSE',
        'rmse_vel': 'Vel.\\ RMSE',
        'rmse_total': 'Total RMSE',
        'pred_time_ms': 'Time (ms)',
    }
    available_1step = {k: v for k, v in cols_1step.items() if k in df.columns}

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{One-step prediction accuracy (clean conditions).}',
        r'\label{tab:onestep}',
        r'\begin{tabular}{l' + 'c' * len(available_1step) + '}',
        r'\toprule',
        'Method & ' + ' & '.join(
            [f'${v}$ $\\downarrow$' for v in available_1step.values()]) + r' \\',
        r'\midrule',
    ]
    for meth in df.index:
        vals = ' & '.join([f'{df.loc[meth, k]:.5f}' if k != 'pred_time_ms'
                          else f'{df.loc[meth, k]:.3f}'
                          for k in available_1step])
        lines.append(f'{meth} & {vals} ' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    with open(os.path.join(FIG_DIR, 'table_onestep.tex'), 'w') as f:
        f.write('\n'.join(lines))
    print("  Saved table_onestep.tex")

    # Robustness table
    if 'rmse_pos_dist' in df.columns:
        lines2 = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{Prediction accuracy under disturbance and model uncertainty.}',
            r'\label{tab:robustness}',
            r'\begin{tabular}{lcccc}',
            r'\toprule',
            r'Method & Pos.\ RMSE (clean) & Pos.\ RMSE (dist.) & Degradation $\uparrow$ & Vel.\ RMSE (dist.) \\',
            r'\midrule',
        ]
        for meth in df.index:
            pc = df.loc[meth, 'rmse_pos']
            pd_ = df.loc[meth, 'rmse_pos_dist']
            deg = pd_ / (pc + 1e-10)
            vd = df.loc[meth, 'rmse_vel_dist']
            lines2.append(
                f'{meth} & {pc:.5f} & {pd_:.5f} & {deg:.1f}$\\times$ & {vd:.5f} ' + r'\\')
        lines2 += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

        with open(os.path.join(FIG_DIR, 'table_robustness.tex'), 'w') as f:
            f.write('\n'.join(lines2))
        print("  Saved table_robustness.tex")

    # Rollout table
    if 'rollout_pos_rmse' in df.columns:
        lines3 = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{Multi-step rollout prediction accuracy.}',
            r'\label{tab:rollout}',
            r'\begin{tabular}{lcc}',
            r'\toprule',
            r'Method & Rollout RMSE (m) $\downarrow$ & Final Error (m) $\downarrow$ \\',
            r'\midrule',
        ]
        for meth in df.index:
            rr = df.loc[meth, 'rollout_pos_rmse']
            fe = df.loc[meth, 'rollout_final_err']
            lines3.append(f'{meth} & {rr:.4f} & {fe:.4f} ' + r'\\')
        lines3 += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

        with open(os.path.join(FIG_DIR, 'table_rollout.tex'), 'w') as f:
            f.write('\n'.join(lines3))
        print("  Saved table_rollout.tex")


def generate_all_figures():
    """Load results from CSV and generate all figures + tables."""
    _setup()
    summary_file = os.path.join(RESULTS_DIR, 'summary.csv')
    if not os.path.exists(summary_file):
        print("No summary.csv found. Run experiments first.")
        return

    summary = pd.read_csv(summary_file)
    print("Generating figures …")
    fig_spider(summary)
    fig_bar_onestep(summary)
    fig_rollout(summary)
    fig_timing(summary)
    fig_mpc_tracking()
    fig_robustness_heatmap(summary)
    generate_latex_tables(summary)
    print("All figures generated.")


if __name__ == '__main__':
    generate_all_figures()
