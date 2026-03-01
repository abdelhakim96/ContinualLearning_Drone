"""
Dataset Generator for BlueROV2 Inverse-Model Pre-Training.

Strategy:
  1. Fly the analytical inverse controller on random smooth trajectories.
  2. Record (state_error_features, normalised_control) pairs.
  3. Save to CSV for reproducibility.

Features (9,):
    [e_surge, e_sway, e_heave, sin(e_psi), cos(e_psi), u, v, w, r]

Labels (4,):
    [X/F_max, Y/F_max, Z/F_max, Mz/T_max]   (normalised to [-1, 1])
"""

import os
import sys
import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from bluerov.src.bluerov_model import BlueROV2, ssa
from bluerov.src.trajectory import lemniscate, circular
from bluerov.src.parameters import MAX_FORCE, MAX_TORQUE
from bluerov.controllers.inverse_bluerov import InverseBlueROV


SAVE_PATH = os.path.join(os.path.dirname(__file__),
                         '..', 'data', 'dataset_bluerov_inverse.csv')


def collect_episode(auv: BlueROV2, ctrl: InverseBlueROV,
                    ref: np.ndarray, noise_std: float = 0.0) -> list:
    """Run one trajectory episode and return feature-label pairs."""
    auv.reset()
    ctrl.reset()
    N = ref.shape[0]
    rows = []
    for k in range(N - 1):
        state_obs, _ = auv.step(np.zeros(4))   # observe before control
        state_obs = auv.state.copy()            # use true state for features

        cmd = ctrl.control(state_obs, ref[k])

        # Features
        x, y, z = state_obs[0], state_obs[1], state_obs[2]
        cos_psi, sin_psi = state_obs[3], state_obs[4]
        u, v, w, r = state_obs[5], state_obs[6], state_obs[7], state_obs[8]
        psi = np.arctan2(sin_psi, cos_psi)

        e_world = np.array([ref[k, 0] - x, ref[k, 1] - y, ref[k, 2] - z])
        e_surge =  cos_psi * e_world[0] + sin_psi * e_world[1]
        e_sway  = -sin_psi * e_world[0] + cos_psi * e_world[1]
        e_heave =  e_world[2]
        e_psi   = ssa(ref[k, 3] - psi)

        feat = [e_surge, e_sway, e_heave, np.sin(e_psi), np.cos(e_psi),
                u, v, w, r]
        label = [cmd[0] / MAX_FORCE,  cmd[1] / MAX_FORCE,
                 cmd[2] / MAX_FORCE,  cmd[3] / MAX_TORQUE]

        rows.append(feat + label)

        # Step simulation with the computed control
        auv.step(cmd, noise_std=noise_std)

    return rows


def generate_dataset(n_episodes: int = 30,
                     episode_len: int = 400,
                     dt: float = 0.05,
                     noise_std: float = 0.005,
                     seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    auv  = BlueROV2(dt=dt)
    ctrl = InverseBlueROV(dt=dt)
    t    = np.linspace(0, episode_len * dt, episode_len)

    all_rows = []
    for ep in range(n_episodes):
        # Randomise trajectory parameters for variety
        a     = np.random.uniform(1.5, 4.0)
        b     = np.random.uniform(0.5, 2.0)
        omega = np.random.uniform(0.2, 0.5)
        z_d   = np.random.uniform(-3.0, -1.0)

        traj_type = ep % 3
        if traj_type == 0:
            ref = lemniscate(t, a=a, b=b, z_depth=z_d, omega=omega)
        elif traj_type == 1:
            ref = circular(t, radius=a, z_depth=z_d, omega=omega)
        else:
            # Random waypoints (smooth via linear interpolation)
            wpts = np.random.uniform(-3, 3, (8, 4))
            wpts[:, 2] = z_d
            ref = np.zeros((episode_len, 4))
            seg = episode_len // 8
            for s in range(8):
                start = s * seg
                end   = min((s + 1) * seg, episode_len)
                ref[start:end] = wpts[s]
            ref[:, 3] = 0.0  # zero heading for random

        rows = collect_episode(auv, ctrl, ref, noise_std=noise_std)
        all_rows.extend(rows)

        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1}/{n_episodes}  total samples: {len(all_rows)}")

    cols = ['e_surge', 'e_sway', 'e_heave', 'sin_epsi', 'cos_epsi',
            'u', 'v', 'w', 'r',
            'X_norm', 'Y_norm', 'Z_norm', 'Mz_norm']
    df = pd.DataFrame(all_rows, columns=cols)
    return df


if __name__ == '__main__':
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print("Generating BlueROV2 inverse-model dataset …")
    df = generate_dataset(n_episodes=40, episode_len=500)
    df.to_csv(SAVE_PATH, index=False)
    print(f"Saved {len(df)} samples → {SAVE_PATH}")
