"""
High-quality publication figures for the BlueROV2 continual-learning paper.

All figures are saved as PDF (vector) and PNG (raster) with a consistent
colour palette and LaTeX-style typography.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid':         True,
    'grid.alpha':        0.35,
    'lines.linewidth':   1.6,
})

# ── Consistent colour / style mapping ─────────────────────────────────────────
CTRL_STYLE = {
    'PID':       {'color': '#e41a1c', 'ls': '--',  'marker': 'o',  'zorder': 3},
    'Inverse':   {'color': '#ff7f00', 'ls': '-.',  'marker': 's',  'zorder': 4},
    'DNN':       {'color': '#984ea3', 'ls': ':',   'marker': '^',  'zorder': 3},
    'LoRA-ER':   {'color': '#377eb8', 'ls': '-',   'marker': 'D',  'zorder': 5},
    'LoRA-EWC':  {'color': '#4daf4a', 'ls': '-',   'marker': 'v',  'zorder': 5},
    'LoRA-AGEM': {'color': '#a65628', 'ls': '-',   'marker': 'P',  'zorder': 5},
}

PHASE_BOUNDS_FRAC = [0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1.0]
PHASE_LABELS = [
    'Phase 1\nClean',
    'Phase 2\nNoise',
    'Phase 3\nCurrent',
    'Phase 4\nAll',
    'Phase 5\nCurr+Noise',
    'Phase 6\nNoise',
    'Phase 7\nClean',
]
PHASE_COLORS = ['#f7f7f7', '#deebf7', '#fcbba1', '#fb6a4a',
                '#fcbba1', '#deebf7', '#f7f7f7']


def _phase_bands(ax, t_total: float, alpha: float = 0.25):
    """Draw alternating phase background shading."""
    for i, (lo, hi) in enumerate(zip(PHASE_BOUNDS_FRAC[:-1],
                                      PHASE_BOUNDS_FRAC[1:])):
        ax.axvspan(lo * t_total, hi * t_total,
                   color=PHASE_COLORS[i], alpha=alpha, zorder=0)
    # Vertical boundary lines
    for frac in PHASE_BOUNDS_FRAC[1:-1]:
        ax.axvline(frac * t_total, color='grey', lw=0.8,
                   ls='--', alpha=0.6, zorder=1)


def _phase_labels_top(ax, t_total: float):
    """Add phase labels along the top of the axis."""
    ylim = ax.get_ylim()
    for i, (lo, hi) in enumerate(zip(PHASE_BOUNDS_FRAC[:-1],
                                      PHASE_BOUNDS_FRAC[1:])):
        mid = (lo + hi) / 2 * t_total
        ax.text(mid, ylim[1], PHASE_LABELS[i],
                ha='center', va='bottom', fontsize=7,
                transform=ax.get_xaxis_transform())


def _legend(ax):
    handles = [
        Line2D([0], [0], color=v['color'], ls=v['ls'],
               linewidth=2.0, label=k)
        for k, v in CTRL_STYLE.items()
    ]
    ax.legend(handles=handles, loc='upper right',
              framealpha=0.85, ncol=2, fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: 2-D Trajectory (X-Y plane)
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectories(results_dir: str, figures_dir: str):
    ref   = np.load(os.path.join(results_dir, 'reference.npy'))
    t_arr = np.load(os.path.join(results_dir, 'time.npy'))
    T     = t_arr[-1]

    controllers = list(CTRL_STYLE.keys())

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True, sharex=True)
    axes = axes.flatten()

    for ax, ctrl in zip(axes, controllers):
        state_path = os.path.join(results_dir, f'{ctrl}_states.npy')
        if not os.path.exists(state_path):
            ax.set_visible(False)
            continue
        states = np.load(state_path)

        ax.plot(ref[:, 1], ref[:, 0], 'k-', lw=1.2, alpha=0.4, label='Reference')
        ax.plot(states[:, 1], states[:, 0],
                color=CTRL_STYLE[ctrl]['color'],
                lw=1.2, label=ctrl)
        ax.set_title(ctrl, fontsize=11)
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('X (m)')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)

    fig.suptitle('BlueROV2 — Lemniscate Trajectory Tracking (XY plane)', y=1.01)
    plt.tight_layout()
    _save(fig, figures_dir, 'fig_trajectories')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Position Error vs Time (all controllers)
# ─────────────────────────────────────────────────────────────────────────────

def plot_position_error(results_dir: str, figures_dir: str):
    t_arr = np.load(os.path.join(results_dir, 'time.npy'))
    T     = t_arr[-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    _phase_bands(ax, T)

    for ctrl, style in CTRL_STYLE.items():
        state_path = os.path.join(results_dir, f'{ctrl}_states.npy')
        ref_path   = os.path.join(results_dir, 'reference.npy')
        if not os.path.exists(state_path):
            continue
        states = np.load(state_path)
        ref    = np.load(ref_path)
        K      = min(len(t_arr) - 1, states.shape[0] - 1, ref.shape[0] - 1)
        pos_err = np.sqrt(
            (states[:K, 0] - ref[:K, 0])**2 +
            (states[:K, 1] - ref[:K, 1])**2 +
            (states[:K, 2] - ref[:K, 2])**2
        )
        ax.plot(t_arr[:K], pos_err,
                color=style['color'], ls=style['ls'],
                label=ctrl, zorder=style['zorder'], alpha=0.85)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.set_title('Position Tracking Error vs. Time')
    ax.set_ylim(bottom=0)
    _legend(ax)
    _phase_labels_top(ax, T)
    plt.tight_layout()
    _save(fig, figures_dir, 'fig_position_error')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Heading Error vs Time
# ─────────────────────────────────────────────────────────────────────────────

def plot_heading_error(results_dir: str, figures_dir: str):
    t_arr = np.load(os.path.join(results_dir, 'time.npy'))
    T     = t_arr[-1]
    ref   = np.load(os.path.join(results_dir, 'reference.npy'))

    fig, ax = plt.subplots(figsize=(10, 4))
    _phase_bands(ax, T)

    for ctrl, style in CTRL_STYLE.items():
        state_path = os.path.join(results_dir, f'{ctrl}_states.npy')
        if not os.path.exists(state_path):
            continue
        states = np.load(state_path)
        K = min(len(t_arr) - 1, states.shape[0] - 1, ref.shape[0] - 1)
        psi = np.arctan2(states[:K, 4], states[:K, 3])
        e_psi = np.abs(np.arctan2(np.sin(ref[:K, 3] - psi),
                                   np.cos(ref[:K, 3] - psi)))
        ax.plot(t_arr[:K], np.degrees(e_psi),
                color=style['color'], ls=style['ls'],
                label=ctrl, zorder=style['zorder'], alpha=0.85)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading error (deg)')
    ax.set_title('Heading Tracking Error vs. Time')
    ax.set_ylim(bottom=0)
    _legend(ax)
    _phase_labels_top(ax, T)
    plt.tight_layout()
    _save(fig, figures_dir, 'fig_heading_error')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Phase-wise RMSE bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase_rmse(results_dir: str, figures_dir: str):
    controllers = list(CTRL_STYLE.keys())
    phase_dfs = []
    for ctrl in controllers:
        csv_p = os.path.join(results_dir, f'{ctrl}_phase_metrics.csv')
        if os.path.exists(csv_p):
            df = pd.read_csv(csv_p)
            df['controller'] = ctrl
            phase_dfs.append(df)

    if not phase_dfs:
        print("No phase CSVs found — skipping phase RMSE plot.")
        return

    combined = pd.concat(phase_dfs, ignore_index=True)
    phases   = combined['phase'].unique()
    n_phases = len(phases)
    n_ctrls  = len(controllers)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in zip(
            axes,
            ['pos_rmse', 'head_rmse'],
            ['Position RMSE (m)', 'Heading RMSE (rad)']):

        x     = np.arange(n_phases)
        width = 0.8 / n_ctrls

        for ci, ctrl in enumerate(controllers):
            sub = combined[combined['controller'] == ctrl]
            vals = []
            for ph in phases:
                row = sub[sub['phase'] == ph]
                vals.append(row[metric].values[0] if len(row) else 0.0)
            offset = (ci - n_ctrls / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9,
                   label=ctrl, color=CTRL_STYLE[ctrl]['color'],
                   edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('\n', ' ') for p in phases],
                            rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Per-Phase {ylabel.split(" ")[0]} RMSE')
        ax.legend(fontsize=8, ncol=2)

    plt.suptitle('BlueROV2 — Per-Phase Performance Comparison', y=1.01)
    plt.tight_layout()
    _save(fig, figures_dir, 'fig_phase_rmse')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Control Effort & Computation Time
# ─────────────────────────────────────────────────────────────────────────────

def plot_effort_time(results_dir: str, figures_dir: str):
    summary_path = os.path.join(results_dir, 'summary.csv')
    if not os.path.exists(summary_path):
        print("summary.csv not found — skipping effort/time plot.")
        return
    df = pd.read_csv(summary_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = [CTRL_STYLE.get(c, {}).get('color', '#333333')
              for c in df['controller']]

    axes[0].barh(df['controller'], df['effort_mean'], color=colors)
    axes[0].set_xlabel('Mean control effort (N / N·m)')
    axes[0].set_title('Control Effort')
    axes[0].invert_yaxis()

    axes[1].barh(df['controller'], df['time_ms_mean'], color=colors)
    axes[1].set_xlabel('Mean computation time (ms)')
    axes[1].set_title('Computation Time')
    axes[1].invert_yaxis()

    plt.tight_layout()
    _save(fig, figures_dir, 'fig_effort_time')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Depth (Z) tracking
# ─────────────────────────────────────────────────────────────────────────────

def plot_depth(results_dir: str, figures_dir: str):
    t_arr = np.load(os.path.join(results_dir, 'time.npy'))
    T     = t_arr[-1]
    ref   = np.load(os.path.join(results_dir, 'reference.npy'))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t_arr, ref[:, 2], 'k--', lw=1.4, alpha=0.6, label='Reference z')
    _phase_bands(ax, T)

    for ctrl, style in CTRL_STYLE.items():
        state_path = os.path.join(results_dir, f'{ctrl}_states.npy')
        if not os.path.exists(state_path):
            continue
        states = np.load(state_path)
        K = min(len(t_arr), states.shape[0])
        ax.plot(t_arr[:K], states[:K, 2],
                color=style['color'], ls=style['ls'],
                label=ctrl, zorder=style['zorder'], alpha=0.85)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Depth Tracking')
    _legend(ax)
    plt.tight_layout()
    _save(fig, figures_dir, 'fig_depth')


# ─────────────────────────────────────────────────────────────────────────────
# Table: Overall summary LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def generate_latex_table(results_dir: str, figures_dir: str):
    summary_path = os.path.join(results_dir, 'summary.csv')
    if not os.path.exists(summary_path):
        return
    df = pd.read_csv(summary_path)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Overall performance metrics for all controllers on the'
        r' BlueROV2 lemniscate trajectory tracking benchmark.}',
        r'\label{tab:overall_results}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Controller & Pos.\ RMSE (m) $\downarrow$ & Head.\ RMSE (rad) $\downarrow$'
        r' & Effort (N) $\downarrow$ & Time (ms) $\downarrow$ \\',
        r'\midrule',
    ]
    for _, row in df.iterrows():
        lines.append(
            f'{row["controller"]} & {row["pos_rmse"]:.3f} & '
            f'{row["head_rmse"]:.3f} & '
            f'{row["effort_mean"]:.2f} & '
            f'{row["time_ms_mean"]:.3f} \\\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    latex_str = '\n'.join(lines)

    out_path = os.path.join(figures_dir, 'table_overall.tex')
    with open(out_path, 'w') as f:
        f.write(latex_str)
    print(f"LaTeX table → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, figures_dir: str, name: str):
    os.makedirs(figures_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(figures_dir, f'{name}.{ext}')
        fig.savefig(path)
    print(f"  Saved {name}.pdf / .png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(results_dir: str = None, figures_dir: str = None):
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__),
                                   '..', 'results')
    if figures_dir is None:
        figures_dir = os.path.join(results_dir, 'figures')

    print("Generating figures …")
    plot_trajectories(results_dir, figures_dir)
    plot_position_error(results_dir, figures_dir)
    plot_heading_error(results_dir, figures_dir)
    plot_phase_rmse(results_dir, figures_dir)
    plot_effort_time(results_dir, figures_dir)
    plot_depth(results_dir, figures_dir)
    generate_latex_table(results_dir, figures_dir)
    print("Done.")


if __name__ == '__main__':
    generate_all_figures()
