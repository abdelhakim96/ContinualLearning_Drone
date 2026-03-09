"""
BlueROV2 NMPC — Lemniscate Trajectory Following Experiment
===========================================================

Compares:
  1. Analytical NMPC   — pure acados SQP-RTI, no learned model
  2. NMPC + PINN       — disturbance estimate from trained PINN residual
  3. NMPC + SINDy      — disturbance estimate from trained SINDy model
  4. NMPC + GP         — disturbance estimate from trained GP model

Scenarios:
  A. Nominal           — no disturbance
  B. Ocean current     — constant body-frame force disturbance (25% F_max)
  C. Parametric uncert — mass/drag scaled ±50%

Outputs (saved to survey/results/nmpc/):
  - Time-series state trajectory CSV
  - PDF figures: 3D trajectory, position error, heading error,
                 control effort, computation time
  - Summary table (LaTeX)

Usage
─────
  cd /repo && python -m survey.experiments.run_nmpc_lemniscate
  python -m survey.experiments.run_nmpc_lemniscate --scenario nominal
  python -m survey.experiments.run_nmpc_lemniscate --no-learning   # NMPC only
"""

import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('ACADOS_SOURCE_DIR', '/opt/acados')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from survey.src.bluerov_sim   import rk4, ssa, NX, NU
from survey.src.parameters    import MAX_FORCE, MAX_TORQUE

from survey.nmpc.params        import NMPCParams, VehicleParams
from survey.nmpc.casadi_model  import BlueROVCasadiModel
from survey.nmpc.bluerov_nmpc  import BlueROVNMPC
from survey.nmpc.reference     import LemniscateReference
from survey.nmpc.interfaces    import (ResidualEstimator,
                                       ZeroResidualEstimator)
from survey.src.bluerov_sim    import collect_data
from survey.methods            import ALL_METHODS

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'nmpc')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
BUILD_DIR   = os.path.join(RESULTS_DIR, 'acados_build')

DT          = 0.05
T_SIM       = 60.0          # simulation horizon [s]
K_SIM       = int(T_SIM / DT)

COLORS = {'Analytical-NMPC': '#2196F3',
          'NMPC+PINN':  '#4CAF50',
          'NMPC+SINDy': '#9C27B0',
          'NMPC+GP':    '#FF9800'}


# ── Residual estimators wrapping learned models ───────────────────────────────

class LearnedResidualEstimatorFromNextState(ResidualEstimator):
    """
    Converts a BaseModel (predict: x,u → x_next) into a continuous-time
    disturbance estimate by finite difference:
        d_est = (x_next_model - x_next_ana) / dt

    This is the bridge between the discrete-time learned models in
    survey/methods/ and the continuous-time acados dynamics.
    """

    def __init__(self, base_model, analytical_rk4_fn, dt=DT):
        self._model    = base_model
        self._rk4      = analytical_rk4_fn   # rk4(x, u, dt) → x_next
        self._dt       = dt

    def estimate_residual(self, x, u):
        xn_model = self._model.predict(x, u)
        xn_ana   = self._rk4(x, u, self._dt)
        # Finite-difference approximation of continuous residual
        residual = (xn_model - xn_ana) / self._dt
        # Clamp to reasonable bounds (avoid wild extrapolations)
        return np.clip(residual, -5.0, 5.0)


# ── Single closed-loop simulation run ─────────────────────────────────────────

def simulate(nmpc: BlueROVNMPC,
             ref:  LemniscateReference,
             dist: np.ndarray = None,
             unc:  float = 0.0,
             label: str = '') -> dict:
    """
    Run one closed-loop simulation episode.

    Parameters
    ----------
    nmpc  : built BlueROVNMPC solver
    ref   : reference trajectory generator
    dist  : constant disturbance [X, Y, Z, Mz] (body frame) or None
    unc   : parametric uncertainty factor (scales mass/drag by 1+unc)
    label : string label for logging
    """
    N = nmpc.params.N_horizon
    t_arr = np.arange(K_SIM + 1) * DT

    # Initial state aligned to reference start
    ref_full = ref.full_trajectory(T_SIM)
    x = np.zeros(NX)
    x[:3]  = ref_full[0, :3]
    x[IDX_PSI] = ref_full[0, 3]

    states  = [x.copy()]
    ctrls   = []
    pos_err = []
    head_err = []
    comp_ms  = []

    for k in range(K_SIM):
        t_k = k * DT

        # Reference window for this step
        ref_win = ref.get_reference_window(t_k, k, N)

        # Solve NMPC
        t0 = time.perf_counter()
        u_opt, status = nmpc.solve(x, ref_win)
        comp_ms.append((time.perf_counter() - t0) * 1e3)

        # Step ground-truth simulator (may differ from NMPC model)
        x = rk4(x, u_opt, DT, dist=dist, unc=unc)
        x[IDX_PSI] = ssa(x[IDX_PSI])  # wrap heading

        states.append(x.copy())
        ctrls.append(u_opt.copy())

        ref_k = ref_full[k]
        pos_err.append(np.linalg.norm(x[:3] - ref_k[:3]))
        head_err.append(abs(ssa(x[IDX_PSI] - ref_k[3])))

    states   = np.array(states)
    ctrls    = np.array(ctrls)
    pos_err  = np.array(pos_err)
    head_err = np.array(head_err)
    comp_ms  = np.array(comp_ms)

    metrics = {
        'label':          label,
        'pos_rmse':       float(np.sqrt(np.mean(pos_err  ** 2))),
        'head_rmse':      float(np.sqrt(np.mean(head_err ** 2))),
        'max_pos_err':    float(np.max(pos_err)),
        'effort_mean':    float(np.mean(np.linalg.norm(ctrls, axis=1))),
        'comp_ms_mean':   float(np.mean(comp_ms)),
        'comp_ms_max':    float(np.max(comp_ms)),
        'states':         states,
        'ctrls':          ctrls,
        'pos_err':        pos_err,
        'head_err':       head_err,
        'comp_ms':        comp_ms,
        'ref':            ref_full,
        't':              t_arr,
    }
    print(f"  {label:25s}  pos_rmse={metrics['pos_rmse']:.4f}m"
          f"  head_rmse={metrics['head_rmse']:.4f}rad"
          f"  t={metrics['comp_ms_mean']:.1f}ms")
    return metrics


# State index alias
IDX_PSI = 3


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_3d_trajectory(results: dict):
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 200,
                         'savefig.bbox': 'tight'})
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')

    ref_traj = next(iter(results.values()))['ref']
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2],
            'k--', linewidth=1.5, label='Reference', alpha=0.6)

    for label, res in results.items():
        st = res['states']
        ax.plot(st[:, 0], st[:, 1], st[:, 2],
                color=COLORS.get(label, 'gray'), linewidth=2, label=label)

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title('3D Trajectory — NMPC Lemniscate Tracking')
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG_DIR, 'nmpc_3d_traj.pdf'))
    plt.close(fig)
    print("  Saved nmpc_3d_traj.pdf")


def plot_time_series(results: dict, scenario: str):
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 200,
                         'savefig.bbox': 'tight'})
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for label, res in results.items():
        c = COLORS.get(label, 'gray')
        t = res['t'][1:]   # drop t=0 (no error at k=0 before first step)
        axes[0].plot(t, res['pos_err'],  color=c, label=label, linewidth=1.5)
        axes[1].plot(t, res['head_err'], color=c, label=label, linewidth=1.5)
        effort = np.linalg.norm(res['ctrls'], axis=1)
        axes[2].plot(t, effort, color=c, label=label, linewidth=1.5)

    axes[0].set_ylabel('Position Error [m]')
    axes[0].set_title(f'NMPC Tracking — {scenario}')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].set_ylabel('Heading Error [rad]')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    axes[2].set_ylabel('Control Effort [N or N·m]')
    axes[2].set_xlabel('Time [s]')
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'nmpc_timeseries_{scenario}.pdf'))
    plt.close(fig)
    print(f"  Saved nmpc_timeseries_{scenario}.pdf")


def plot_xy_trajectory(results: dict, scenario: str):
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 200,
                         'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(8, 6))

    ref = next(iter(results.values()))['ref']
    ax.plot(ref[:, 0], ref[:, 1], 'k--', lw=1.5, label='Reference', alpha=0.6)

    for label, res in results.items():
        st = res['states']
        ax.plot(st[:, 0], st[:, 1], color=COLORS.get(label, 'gray'),
                linewidth=2, label=label)
        ax.plot(st[0, 0], st[0, 1], 'o', color=COLORS.get(label, 'gray'),
                markersize=6)

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title(f'XY Trajectory — {scenario}')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'nmpc_xy_{scenario}.pdf'))
    plt.close(fig)
    print(f"  Saved nmpc_xy_{scenario}.pdf")


def plot_computation_time(results: dict):
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 200,
                         'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(results.keys())
    means  = [results[l]['comp_ms_mean'] for l in labels]
    maxs   = [results[l]['comp_ms_max']  for l in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, means, 0.4, label='Mean', color=[COLORS.get(l,'gray') for l in labels])
    ax.bar(x + 0.2, maxs,  0.4, label='Max',  color=[COLORS.get(l,'gray') for l in labels],
           alpha=0.5, hatch='//')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel('Computation Time [ms]')
    ax.set_title('NMPC Solve Time per Step')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.axhline(DT * 1e3, color='red', linestyle='--', alpha=0.7,
               label=f'Budget {DT*1e3:.0f}ms')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'nmpc_computation_time.pdf'))
    plt.close(fig)
    print("  Saved nmpc_computation_time.pdf")


def generate_summary_table(all_results: dict):
    """Generate LaTeX summary table across all scenarios."""
    rows = []
    for scenario, res_dict in all_results.items():
        for label, res in res_dict.items():
            rows.append({
                'Scenario': scenario,
                'Controller': label,
                'Pos. RMSE [m]': f"{res['pos_rmse']:.4f}",
                'Head. RMSE [rad]': f"{res['head_rmse']:.4f}",
                'Max Pos. Err [m]': f"{res['max_pos_err']:.4f}",
                'Effort Mean': f"{res['effort_mean']:.2f}",
                'Time [ms]': f"{res['comp_ms_mean']:.2f}",
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'nmpc_summary.csv'), index=False)

    # LaTeX
    lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        r'\caption{BlueROV2 NMPC lemniscate tracking performance.}',
        r'\label{tab:nmpc_summary}',
        r'\begin{tabular}{llccccc}',
        r'\toprule',
        r'Scenario & Controller & Pos.\ RMSE [m] $\downarrow$ & '
        r'Head.\ RMSE [rad] $\downarrow$ & Max Err [m] $\downarrow$ & '
        r'Effort & Time [ms] \\',
        r'\midrule',
    ]
    prev_sc = None
    for _, row in df.iterrows():
        sc = row['Scenario'] if row['Scenario'] != prev_sc else ''
        sc_tex = sc.replace('_', '\\_')
        prev_sc = row['Scenario']
        line = (f"{sc_tex} & {row['Controller']} & "
                f"{row['Pos. RMSE [m]']} & {row['Head. RMSE [rad]']} & "
                f"{row['Max Pos. Err [m]']} & {row['Effort Mean']} & "
                f"{row['Time [ms]']}" + r' \\')
        lines.append(line)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']

    with open(os.path.join(FIG_DIR, 'nmpc_summary_table.tex'), 'w') as f:
        f.write('\n'.join(lines))
    print("  Saved nmpc_summary_table.tex")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  BlueROV2 NMPC — Lemniscate Trajectory Following")
    print("=" * 70)

    # ── Build NMPC solver (compiled once) ─────────────────────────────────────
    print("\n[1/4] Building acados NMPC solver …")
    vehicle = VehicleParams()
    params  = NMPCParams(N_horizon=20, dt=DT,
                         Q_x=50, Q_y=50, Q_z=50, Q_psi=20,
                         R_X=0.005, R_Y=0.005, R_Z=0.005, R_Mz=0.05,
                         Q_x_e=100, Q_y_e=100, Q_z_e=100, Q_psi_e=40,
                         nlp_solver_type='SQP_RTI')

    casadi_model = BlueROVCasadiModel(vp=vehicle, with_disturbance=True)
    nmpc_base    = BlueROVNMPC(casadi_model, params,
                               residual=ZeroResidualEstimator(),
                               build_dir=BUILD_DIR)
    nmpc_base.build(verbose=False)
    print("  NMPC solver built successfully.")

    # ── Train learned models ──────────────────────────────────────────────────
    learned_nmpc_configs = {}
    if not args.no_learning:
        print("\n[2/4] Training learned residual models …")
        X_tr, U_tr, Xn_tr = collect_data(N=5000, dt=DT, seed=42)

        for name in ['PINN', 'SINDy', 'GP']:
            if name not in ALL_METHODS:
                continue
            print(f"  Training {name} …", end=" ", flush=True)
            t0 = time.perf_counter()
            model = ALL_METHODS[name]()
            model.train(X_tr, U_tr, Xn_tr)
            print(f"({time.perf_counter()-t0:.1f}s)")

            estimator = LearnedResidualEstimatorFromNextState(model, rk4, DT)
            nmpc_lrn  = BlueROVNMPC(casadi_model, params,
                                     residual=estimator,
                                     build_dir=BUILD_DIR)
            nmpc_lrn.build(verbose=False)
            learned_nmpc_configs[f'NMPC+{name}'] = nmpc_lrn
    else:
        print("\n[2/4] Skipping learned models (--no-learning).")

    # All controllers for this run
    controllers = {'Analytical-NMPC': nmpc_base}
    controllers.update(learned_nmpc_configs)

    # ── Reference trajectory ───────────────────────────────────────────────────
    ref = LemniscateReference(a=3.0, b=1.5, omega=0.25, z_depth=-2.0, dt=DT)

    # ── Scenarios ──────────────────────────────────────────────────────────────
    scenarios = {
        'nominal': dict(dist=None, unc=0.0),
    }
    if not args.scenario or args.scenario == 'all':
        scenarios['current_25pct'] = dict(
            dist=np.array([0.25 * MAX_FORCE, 0.125 * MAX_FORCE, 0, 0]),
            unc=0.0)
        scenarios['param_50pct'] = dict(dist=None, unc=0.5)
    elif args.scenario != 'nominal':
        if args.scenario == 'current':
            scenarios['current_25pct'] = dict(
                dist=np.array([0.25*MAX_FORCE, 0.125*MAX_FORCE, 0, 0]), unc=0.0)
        elif args.scenario == 'param':
            scenarios['param_50pct'] = dict(dist=None, unc=0.5)

    # ── Run simulations ────────────────────────────────────────────────────────
    print("\n[3/4] Running closed-loop simulations …")
    all_results = {}

    for sc_name, sc_cfg in scenarios.items():
        print(f"\n  Scenario: {sc_name}")
        sc_results = {}
        for ctrl_name, nmpc in controllers.items():
            res = simulate(nmpc, ref,
                           dist=sc_cfg['dist'], unc=sc_cfg['unc'],
                           label=ctrl_name)
            sc_results[ctrl_name] = res
        all_results[sc_name] = sc_results

    # ── Generate figures ───────────────────────────────────────────────────────
    print("\n[4/4] Generating figures …")

    # Use nominal results for trajectory plots
    if 'nominal' in all_results:
        plot_3d_trajectory(all_results['nominal'])
        plot_xy_trajectory(all_results['nominal'], 'nominal')
        plot_time_series(all_results['nominal'],   'nominal')
        plot_computation_time(all_results['nominal'])

    for sc_name, sc_results in all_results.items():
        if sc_name != 'nominal':
            plot_xy_trajectory(sc_results, sc_name)
            plot_time_series(sc_results, sc_name)

    generate_summary_table(all_results)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='all',
                        choices=['all', 'nominal', 'current', 'param'])
    parser.add_argument('--no-learning', action='store_true',
                        help='Skip learned residual models (analytical NMPC only)')
    args = parser.parse_args()
    main(args)
