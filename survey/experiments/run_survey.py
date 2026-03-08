"""
Survey Experiment Runner — Compare data-driven dynamics models for BlueROV2.

Evaluates: Classic (analytical), NN, PINN, SINDy, DMD, GP, Hankel-DMD
on three tasks:
  1. One-step prediction accuracy (nominal + disturbed)
  2. Multi-step rollout accuracy
  3. MPC tracking performance on lemniscate trajectory
  4. Computation time

Results are saved to survey/results/ as CSV and .npy files.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from survey.src.bluerov_sim import (collect_data, rk4, lemniscate_ref,
                                      dynamics, ssa, NX, NU)
from survey.src.parameters import MAX_FORCE, MAX_TORQUE
from survey.methods import ALL_METHODS
from survey.mpc.mpc_controller import MPCController
from survey.mpc.inverse_controller import InverseController

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
DT = 0.05


# ── 1. Data collection ───────────────────────────────────────────────────────

def get_datasets(seed=42):
    """Collect training and test data."""
    X_tr, U_tr, Xn_tr = collect_data(N=5000, dt=DT, seed=seed)
    X_te, U_te, Xn_te = collect_data(N=1000, dt=DT, seed=seed + 1)
    return (X_tr, U_tr, Xn_tr), (X_te, U_te, Xn_te)


# ── 2. One-step prediction evaluation ────────────────────────────────────────

def eval_one_step(model, X, U, Xn):
    """Compute one-step prediction RMSE per state dimension and overall."""
    t0 = time.perf_counter()
    Xn_pred = model.predict_batch(X, U)
    elapsed = (time.perf_counter() - t0) / len(X) * 1e3  # ms per sample

    err = Xn_pred - Xn
    rmse_per_dim = np.sqrt(np.mean(err ** 2, axis=0))
    rmse_pos = np.sqrt(np.mean(err[:, :3] ** 2))
    rmse_vel = np.sqrt(np.mean(err[:, 4:8] ** 2))
    rmse_total = np.sqrt(np.mean(err ** 2))
    return {
        'rmse_pos': rmse_pos,
        'rmse_vel': rmse_vel,
        'rmse_total': rmse_total,
        'rmse_per_dim': rmse_per_dim,
        'time_ms': elapsed,
    }


# ── 3. Disturbed one-step evaluation ─────────────────────────────────────────

def eval_one_step_disturbed(model, N=1000, dist_pct=25.0, unc=0.5, seed=99):
    """Test prediction under disturbance + uncertainty (model trained on clean)."""
    rng = np.random.RandomState(seed)
    dist = np.array([dist_pct/100*MAX_FORCE, dist_pct/100*MAX_FORCE*0.5, 0, 0])

    X, U, Xn_true = [], [], []
    x = np.zeros(NX)
    for _ in range(N):
        u_cmd = rng.uniform([-MAX_FORCE]*3 + [-MAX_TORQUE],
                            [MAX_FORCE]*3 + [MAX_TORQUE])
        xn = rk4(x, u_cmd, DT, dist=dist, unc=unc)
        X.append(x.copy()); U.append(u_cmd.copy()); Xn_true.append(xn.copy())
        x = xn
        if rng.rand() < 0.05:
            x = rng.randn(NX) * np.array([3, 3, 2, np.pi, 1, 1, 0.5, 0.5])
    X = np.array(X); U = np.array(U); Xn_true = np.array(Xn_true)

    Xn_pred = model.predict_batch(X, U)
    err = Xn_pred - Xn_true
    return {
        'rmse_pos_dist': np.sqrt(np.mean(err[:, :3] ** 2)),
        'rmse_vel_dist': np.sqrt(np.mean(err[:, 4:8] ** 2)),
        'rmse_total_dist': np.sqrt(np.mean(err ** 2)),
    }


# ── 4. Multi-step rollout evaluation ─────────────────────────────────────────

def eval_rollout(model, horizon=200, seed=77):
    """Open-loop rollout error over `horizon` steps."""
    rng = np.random.RandomState(seed)
    x0 = np.zeros(NX)
    U_seq = rng.uniform([-MAX_FORCE]*3 + [-MAX_TORQUE],
                        [MAX_FORCE]*3 + [MAX_TORQUE],
                        size=(horizon, NU)) * 0.3  # mild inputs

    # Ground truth rollout
    x = x0.copy()
    gt = [x.copy()]
    for u in U_seq:
        x = rk4(x, u, DT)
        gt.append(x.copy())
    gt = np.array(gt)

    # Model rollout
    if hasattr(model, 'reset_buffers'):
        model.reset_buffers(x0)
    pred = model.rollout(x0, U_seq, DT)

    err = pred - gt
    pos_err = np.sqrt(np.mean(err[:, :3] ** 2, axis=1))
    return {
        'rollout_pos_rmse': float(np.sqrt(np.mean(pos_err ** 2))),
        'rollout_final_err': float(np.linalg.norm(err[-1, :3])),
        'rollout_traj_gt': gt,
        'rollout_traj_pred': pred,
    }


# ── 5. MPC tracking evaluation ───────────────────────────────────────────────

def eval_mpc_tracking(model, T=30.0, mpc_horizon=8, n_samples=150, seed=42):
    """Closed-loop MPC tracking on lemniscate trajectory."""
    np.random.seed(seed)
    K = int(T / DT)
    t = np.linspace(0, T, K + 1)
    ref = lemniscate_ref(t)

    mpc = MPCController(model, horizon=mpc_horizon, n_samples=n_samples, dt=DT)

    x = np.zeros(NX)
    x[:3] = ref[0, :3]; x[3] = ref[0, 3]
    if hasattr(model, 'reset_buffers'):
        model.reset_buffers(x)

    states = [x.copy()]
    pos_errs = []; head_errs = []; efforts = []; times_ms = []

    for k in range(K):
        ref_window = ref[k:k + mpc_horizon]
        if len(ref_window) < mpc_horizon:
            ref_window = np.vstack([ref_window,
                np.tile(ref_window[-1], (mpc_horizon - len(ref_window), 1))])

        t0 = time.perf_counter()
        u_cmd = mpc.control(x, ref_window)
        elapsed = (time.perf_counter() - t0) * 1e3

        x = rk4(x, u_cmd, DT)
        states.append(x.copy())

        pos_errs.append(np.linalg.norm(x[:3] - ref[k, :3]))
        head_errs.append(abs(ssa(x[3] - ref[k, 3])))
        efforts.append(np.linalg.norm(u_cmd))
        times_ms.append(elapsed)

    return {
        'mpc_pos_rmse': float(np.sqrt(np.mean(np.array(pos_errs) ** 2))),
        'mpc_head_rmse': float(np.sqrt(np.mean(np.array(head_errs) ** 2))),
        'mpc_effort': float(np.mean(efforts)),
        'mpc_time_ms': float(np.mean(times_ms)),
        'mpc_states': np.array(states),
        'mpc_ref': ref,
    }


# ── 6. Inverse controller evaluation ─────────────────────────────────────────

def eval_inverse_tracking(T=60.0, dist_pct=0.0, unc=0.0, seed=42):
    """Closed-loop inverse-dynamics tracking on lemniscate."""
    np.random.seed(seed)
    K = int(T / DT)
    t = np.linspace(0, T, K + 1)
    ref = lemniscate_ref(t)

    ctrl = InverseController(dt=DT)
    dist = np.array([dist_pct/100*MAX_FORCE, dist_pct/100*MAX_FORCE*0.5, 0, 0]) \
           if dist_pct > 0 else None

    x = np.zeros(NX)
    x[:3] = ref[0, :3]; x[3] = ref[0, 3]
    states = [x.copy()]
    pos_errs = []; head_errs = []

    for k in range(K):
        u_cmd = ctrl.control(x, ref[k])
        x = rk4(x, u_cmd, DT, dist=dist, unc=unc)
        states.append(x.copy())
        pos_errs.append(np.linalg.norm(x[:3] - ref[k, :3]))
        head_errs.append(abs(ssa(x[3] - ref[k, 3])))

    return {
        'inv_pos_rmse': float(np.sqrt(np.mean(np.array(pos_errs) ** 2))),
        'inv_head_rmse': float(np.sqrt(np.mean(np.array(head_errs) ** 2))),
        'inv_states': np.array(states),
        'inv_ref': ref,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_survey(skip_mpc=False):
    """Run all evaluations and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  BlueROV2 Data-Driven Dynamics — Comparative Survey")
    print("=" * 70)

    # Collect data
    print("\n[1/5] Collecting training and test data …")
    (X_tr, U_tr, Xn_tr), (X_te, U_te, Xn_te) = get_datasets()
    print(f"  Train: {len(X_tr)},  Test: {len(X_te)}")

    # Train all models
    print("\n[2/5] Training models …")
    models = {}
    train_times = {}
    for name, ModelClass in ALL_METHODS.items():
        print(f"  Training {name} …", end=" ", flush=True)
        model = ModelClass()
        t0 = time.perf_counter()
        model.train(X_tr, U_tr, Xn_tr)
        train_times[name] = time.perf_counter() - t0
        models[name] = model
        print(f"({train_times[name]:.2f}s)")

    # One-step evaluation
    print("\n[3/5] One-step prediction evaluation …")
    onestep_rows = []
    for name, model in models.items():
        res = eval_one_step(model, X_te, U_te, Xn_te)
        dist_res = eval_one_step_disturbed(model)
        row = {
            'method': name,
            'train_time_s': train_times[name],
            'rmse_pos': res['rmse_pos'],
            'rmse_vel': res['rmse_vel'],
            'rmse_total': res['rmse_total'],
            'pred_time_ms': res['time_ms'],
            'rmse_pos_dist': dist_res['rmse_pos_dist'],
            'rmse_vel_dist': dist_res['rmse_vel_dist'],
            'rmse_total_dist': dist_res['rmse_total_dist'],
        }
        onestep_rows.append(row)
        print(f"  {name:8s}  pos={res['rmse_pos']:.5f}  vel={res['rmse_vel']:.5f}"
              f"  dist_pos={dist_res['rmse_pos_dist']:.5f}"
              f"  t={res['time_ms']:.3f}ms")

    onestep_df = pd.DataFrame(onestep_rows)
    onestep_df.to_csv(os.path.join(RESULTS_DIR, 'onestep_results.csv'), index=False)

    # Rollout evaluation
    print("\n[4/5] Multi-step rollout evaluation …")
    rollout_rows = []
    rollout_data = {}
    for name, model in models.items():
        if hasattr(model, 'reset_buffers'):
            model.reset_buffers(np.zeros(NX))
        res = eval_rollout(model)
        rollout_rows.append({
            'method': name,
            'rollout_pos_rmse': res['rollout_pos_rmse'],
            'rollout_final_err': res['rollout_final_err'],
        })
        rollout_data[name] = res
        print(f"  {name:8s}  rollout_rmse={res['rollout_pos_rmse']:.4f}"
              f"  final_err={res['rollout_final_err']:.4f}")

    rollout_df = pd.DataFrame(rollout_rows)
    rollout_df.to_csv(os.path.join(RESULTS_DIR, 'rollout_results.csv'), index=False)

    # MPC tracking (optional — slow)
    mpc_data = {}
    if not skip_mpc:
        print("\n[5/5] MPC tracking evaluation …")
        mpc_rows = []
        for name, model in models.items():
            print(f"  MPC with {name} model …", end=" ", flush=True)
            if hasattr(model, 'reset_buffers'):
                model.reset_buffers(np.zeros(NX))
            res = eval_mpc_tracking(model, T=30.0, mpc_horizon=8, n_samples=150)
            mpc_rows.append({
                'method': name,
                'mpc_pos_rmse': res['mpc_pos_rmse'],
                'mpc_head_rmse': res['mpc_head_rmse'],
                'mpc_effort': res['mpc_effort'],
                'mpc_time_ms': res['mpc_time_ms'],
            })
            mpc_data[name] = res
            print(f"pos={res['mpc_pos_rmse']:.3f}  head={res['mpc_head_rmse']:.3f}"
                  f"  t={res['mpc_time_ms']:.1f}ms")

        mpc_df = pd.DataFrame(mpc_rows)
        mpc_df.to_csv(os.path.join(RESULTS_DIR, 'mpc_results.csv'), index=False)
    else:
        print("\n[5/5] Skipping MPC tracking (--skip-mpc).")

    # Save rollout trajectories
    for name in rollout_data:
        np.save(os.path.join(RESULTS_DIR, f'{name}_rollout_gt.npy'),
                rollout_data[name]['rollout_traj_gt'])
        np.save(os.path.join(RESULTS_DIR, f'{name}_rollout_pred.npy'),
                rollout_data[name]['rollout_traj_pred'])

    # Combined summary table
    summary = onestep_df.merge(rollout_df, on='method')
    if not skip_mpc:
        summary = summary.merge(mpc_df, on='method')
    summary.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}/")

    return {
        'models': models,
        'onestep': onestep_df,
        'rollout': rollout_df,
        'rollout_data': rollout_data,
        'mpc_data': mpc_data,
        'summary': summary,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-mpc', action='store_true')
    args = parser.parse_args()
    results = run_survey(skip_mpc=args.skip_mpc)

    print("\nGenerating figures …")
    from survey.utils.plot_survey import generate_all_figures
    generate_all_figures()
    print("Done!")
