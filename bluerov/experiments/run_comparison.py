"""
Main Comparative Experiment — BlueROV2 Lemniscate Trajectory Tracking.

Compares five controllers across seven operational phases:

  Phase 1 (0   – 1/7 T):  Clean — no disturbance, no noise
  Phase 2 (1/7 – 2/7 T):  Sensor noise only
  Phase 3 (2/7 – 3/7 T):  Ocean current + noise
  Phase 4 (3/7 – 4/7 T):  Ocean current + noise + hydrodynamic uncertainty
  Phase 5 (4/7 – 5/7 T):  Ocean current + noise (uncertainty removed)
  Phase 6 (5/7 – 6/7 T):  Noise only (current removed)
  Phase 7 (6/7 – T):      Clean (return to nominal)

Controllers
-----------
  PID       : hierarchical PID
  Inverse   : analytical inverse dynamics (model-based feedforward)
  DNN       : pre-trained MLP, no online adaptation
  LoRA-ER   : LoRA adapters + Experience Replay
  LoRA-EWC  : LoRA adapters + EWC regularisation
  LoRA-AGEM : LoRA adapters + A-GEM gradient projection
"""

import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from bluerov.src.bluerov_model import BlueROV2, ssa
from bluerov.src.trajectory    import lemniscate
from bluerov.src.parameters    import MAX_FORCE, MAX_TORQUE

from bluerov.controllers.pid_bluerov      import PIDBlueROV
from bluerov.controllers.inverse_bluerov  import InverseBlueROV
from bluerov.controllers.dnn_bluerov      import DNNBlueROV
from bluerov.controllers.lora_cl_bluerov  import LoRACLBlueROV

from bluerov.models.mlp  import MLP
from bluerov.models.lora import LoRAMLP

from bluerov.cl.lora_er   import LoRAER
from bluerov.cl.lora_ewc  import LoRAEWC
from bluerov.cl.lora_agem import LoRAAGEM

from bluerov.training.train_inverse import load_trained_model, MODEL_PATH

# ── Experiment parameters ─────────────────────────────────────────────────────
DT          = 0.05          # simulation timestep (s)
T_TOTAL     = 200.0         # total simulation time (s)
K_END       = int(T_TOTAL / DT)

LEMNISCATE_A     = 3.0
LEMNISCATE_B     = 1.5
LEMNISCATE_OMEGA = 0.25     # rad/s
LEMNISCATE_Z     = -2.0     # depth (m)

DISTURBANCE_PCT  = 25.0     # ocean current as % of MAX_FORCE
NOISE_STD        = 0.02     # position noise std (m, rad)
UNCERTAINTY_FAC  = 0.5      # scale factor for hydrodynamic params

LORA_RANK  = 8
LORA_ALPHA = 16.0
LORA_LR    = 5e-4

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Helpers ───────────────────────────────────────────────────────────────────

def position_error(state: np.ndarray, ref: np.ndarray) -> float:
    """Euclidean XYZ position error."""
    return np.sqrt((state[0]-ref[0])**2 + (state[1]-ref[1])**2 + (state[2]-ref[2])**2)


def heading_error(state: np.ndarray, ref: np.ndarray) -> float:
    """Absolute heading error (rad)."""
    psi = np.arctan2(state[4], state[3])
    return abs(ssa(ref[3] - psi))


def phase_flags(k: int, k_end: int):
    """Return (disturbance_pct, uncertainty_factor, noise_std) for step k."""
    frac = k / k_end
    if   frac < 1/7:  return 0.0,               0.0,              0.0
    elif frac < 2/7:  return 0.0,               0.0,              NOISE_STD
    elif frac < 3/7:  return DISTURBANCE_PCT,   0.0,              NOISE_STD
    elif frac < 4/7:  return DISTURBANCE_PCT,   UNCERTAINTY_FAC,  NOISE_STD
    elif frac < 5/7:  return DISTURBANCE_PCT,   0.0,              NOISE_STD
    elif frac < 6/7:  return 0.0,               0.0,              NOISE_STD
    else:             return 0.0,               0.0,              0.0


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_controller(name: str, ctrl, auv: BlueROV2,
                   ref: np.ndarray, is_learning: bool = False):
    """
    Simulate one controller over the full trajectory.

    Returns
    -------
    metrics : dict with lists 'pos_err', 'head_err', 'control_effort', 'time_ms'
    trajectory : (K_END, 9) recorded state history
    """
    auv.reset()
    if hasattr(ctrl, 'reset'):
        ctrl.reset()

    prev_dist = 0.0
    prev_unc  = 0.0

    pos_errs  = []
    head_errs = []
    efforts   = []
    times_ms  = []
    states    = np.zeros((K_END + 1, 9))
    cmds      = np.zeros((K_END, 4))

    # Reset AUV near start of trajectory
    init_state = np.zeros(9)
    init_state[0] = ref[0, 0]
    init_state[1] = ref[0, 1]
    init_state[2] = ref[0, 2]
    init_state[3] = np.cos(ref[0, 3])
    init_state[4] = np.sin(ref[0, 3])
    auv.reset(init_state)

    states[0] = auv.state.copy()

    for k in range(K_END):
        dist_pct, unc_fac, ns = phase_flags(k, K_END)

        # Update disturbance / uncertainty when phase changes
        if dist_pct != prev_dist:
            auv.set_disturbance(dist_pct)
            prev_dist = dist_pct
        if unc_fac != prev_unc:
            if unc_fac > 0:
                auv.set_uncertainty(unc_fac)
            else:
                auv.reset_uncertainty()
            prev_unc = unc_fac

        state_obs = states[k]  # noisy observation (previous step)

        # ── Controller inference ───────────────────────────────────────────
        t0 = time.perf_counter()
        cmd = ctrl.control(state_obs, ref[k])
        elapsed = (time.perf_counter() - t0) * 1e3  # ms

        cmds[k] = cmd

        # ── Simulate one step ──────────────────────────────────────────────
        state_noisy, state_true = auv.step(cmd, noise_std=ns)
        states[k + 1] = state_noisy

        # ── Online learning step ───────────────────────────────────────────
        if is_learning and k > 0:
            ctrl.learn(state_obs, ref[k])

        # ── Metrics ───────────────────────────────────────────────────────
        pos_errs.append(position_error(state_true, ref[k]))
        head_errs.append(heading_error(state_true, ref[k]))
        efforts.append(np.linalg.norm(cmd))
        times_ms.append(elapsed)

    metrics = {
        'pos_err':       np.array(pos_errs),
        'head_err':      np.array(head_errs),
        'control_effort': np.array(efforts),
        'time_ms':       np.array(times_ms),
    }
    return metrics, states, cmds


def build_controllers(base_mlp: MLP) -> dict:
    """Instantiate all controllers."""
    ctrls = {}

    # 1. PID
    ctrls['PID'] = (PIDBlueROV(dt=DT), False)

    # 2. Analytical Inverse
    ctrls['Inverse'] = (InverseBlueROV(dt=DT), False)

    # 3. Pre-trained DNN (no online learning)
    ctrls['DNN'] = (DNNBlueROV(base_mlp, dt=DT, device=DEVICE), False)

    # 4. LoRA-ER
    lora_er_model = LoRAMLP(base_mlp, rank=LORA_RANK, alpha=LORA_ALPHA)
    lora_er_algo  = LoRAER(lora_er_model, lr=LORA_LR, device=DEVICE)
    ctrls['LoRA-ER'] = (LoRACLBlueROV(lora_er_model, lora_er_algo,
                                        dt=DT, device=DEVICE), True)

    # 5. LoRA-EWC
    lora_ewc_model = LoRAMLP(base_mlp, rank=LORA_RANK, alpha=LORA_ALPHA)
    lora_ewc_algo  = LoRAEWC(lora_ewc_model, lr=LORA_LR, device=DEVICE)
    ctrls['LoRA-EWC'] = (LoRACLBlueROV(lora_ewc_model, lora_ewc_algo,
                                          dt=DT, device=DEVICE), True)

    # 6. LoRA-AGEM
    lora_agem_model = LoRAMLP(base_mlp, rank=LORA_RANK, alpha=LORA_ALPHA)
    lora_agem_algo  = LoRAAGEM(lora_agem_model, lr=LORA_LR, device=DEVICE)
    ctrls['LoRA-AGEM'] = (LoRACLBlueROV(lora_agem_model, lora_agem_algo,
                                           dt=DT, device=DEVICE), True)

    return ctrls


def compute_phase_metrics(metrics: dict, k_end: int) -> pd.DataFrame:
    """Break per-step metrics into per-phase RMSE values."""
    boundaries = [0] + [int(k_end * i / 7) for i in range(1, 8)]
    phase_names = [
        'Phase 1\nClean',
        'Phase 2\nNoise',
        'Phase 3\nCurrent',
        'Phase 4\nAll',
        'Phase 5\nCurr+Noise',
        'Phase 6\nNoise',
        'Phase 7\nClean',
    ]
    rows = []
    for i in range(7):
        s, e = boundaries[i], boundaries[i + 1]
        rows.append({
            'phase': phase_names[i],
            'pos_rmse':  float(np.sqrt(np.mean(metrics['pos_err'][s:e]  ** 2))),
            'head_rmse': float(np.sqrt(np.mean(metrics['head_err'][s:e] ** 2))),
            'effort':    float(np.mean(metrics['control_effort'][s:e])),
            'time_ms':   float(np.mean(metrics['time_ms'][s:e])),
        })
    return pd.DataFrame(rows)


def run_all(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Build trajectory ───────────────────────────────────────────────────
    t   = np.linspace(0, T_TOTAL, K_END + 1)
    ref = lemniscate(t, a=LEMNISCATE_A, b=LEMNISCATE_B,
                     z_depth=LEMNISCATE_Z, omega=LEMNISCATE_OMEGA)

    # ── Load base model ────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print("Base model not found — training now …")
        from bluerov.training.generate_dataset import generate_dataset, SAVE_PATH
        import os as _os
        _os.makedirs(_os.path.dirname(SAVE_PATH), exist_ok=True)
        df = generate_dataset(n_episodes=40, episode_len=500, dt=DT)
        df.to_csv(SAVE_PATH, index=False)
        from bluerov.training.train_inverse import train
        train()

    base_mlp = load_trained_model(device=DEVICE)

    controllers = build_controllers(base_mlp)
    auv         = BlueROV2(dt=DT)

    all_metrics = {}
    all_states  = {}
    all_cmds    = {}

    for ctrl_name, (ctrl, is_learning) in controllers.items():
        print(f"\n── Running: {ctrl_name} ──────────────────────────")
        metrics, states, cmds = run_controller(
            ctrl_name, ctrl, auv, ref, is_learning=is_learning)
        all_metrics[ctrl_name] = metrics
        all_states[ctrl_name]  = states
        all_cmds[ctrl_name]    = cmds

        # Compute phase breakdown
        phase_df = compute_phase_metrics(metrics, K_END)
        phase_df.insert(0, 'controller', ctrl_name)
        phase_df.to_csv(os.path.join(RESULTS_DIR,
                                      f'{ctrl_name}_phase_metrics.csv'),
                         index=False)

        overall = {
            'controller':   ctrl_name,
            'pos_rmse_all': float(np.sqrt(np.mean(metrics['pos_err']    ** 2))),
            'head_rmse_all':float(np.sqrt(np.mean(metrics['head_err']   ** 2))),
            'effort_mean':  float(np.mean(metrics['control_effort'])),
            'time_ms_mean': float(np.mean(metrics['time_ms'])),
        }
        print(f"  RMSE pos={overall['pos_rmse_all']:.3f} m  "
              f"head={overall['head_rmse_all']:.3f} rad  "
              f"t={overall['time_ms_mean']:.3f} ms")

    # ── Save consolidated results ──────────────────────────────────────────
    summary_rows = []
    for ctrl_name, metrics in all_metrics.items():
        summary_rows.append({
            'controller':    ctrl_name,
            'pos_rmse':      float(np.sqrt(np.mean(metrics['pos_err']  ** 2))),
            'head_rmse':     float(np.sqrt(np.mean(metrics['head_err'] ** 2))),
            'effort_mean':   float(np.mean(metrics['control_effort'])),
            'time_ms_mean':  float(np.mean(metrics['time_ms'])),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)

    # Save full time series
    np.save(os.path.join(RESULTS_DIR, 'reference.npy'), ref)
    for name, states in all_states.items():
        np.save(os.path.join(RESULTS_DIR, f'{name}_states.npy'), states)
    for name, cmds in all_cmds.items():
        np.save(os.path.join(RESULTS_DIR, f'{name}_cmds.npy'), cmds)

    t_arr = np.linspace(0, T_TOTAL, K_END + 1)
    np.save(os.path.join(RESULTS_DIR, 'time.npy'), t_arr)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(summary_df.to_string(index=False))
    return all_metrics, all_states, ref, t_arr


if __name__ == '__main__':
    run_all()
