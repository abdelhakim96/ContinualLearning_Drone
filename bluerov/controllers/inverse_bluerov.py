"""
Analytical Inverse-Dynamics Controller for BlueROV2.

Given a desired position reference and current state, the controller:
1. Computes position error and derives desired acceleration via a PD law.
2. Inverts the BlueROV2 equations of motion to obtain [X, Y, Z, M_z].

This gives a model-based feedforward controller that compensates
Coriolis, buoyancy, and damping forces.
"""

import numpy as np
from ..src.bluerov_model import ssa
from ..src.parameters import (
    m, g, F_bouy,
    X_ud, Y_vd, Z_wd, N_rd,
    X_u, X_uc, Y_v, Y_vc, Z_w, Z_wc, N_r, N_rc,
    I_zz, MAX_FORCE, MAX_TORQUE,
)


class InverseBlueROV:
    """
    Analytical inverse-dynamics controller.

    Desired accelerations are computed from a PD position law, then
    the BlueROV2 dynamics are inverted:

        X   = M_u * ü_d  −  (m − Y_vd) v r  −  (X_u + X_uc|u|) u
        Y   = M_v * v̈_d  +  (m − X_ud) u r  −  (Y_v + Y_vc|v|) v
        Z   = M_w * ẇ_d  −  (Z_w + Z_wc|w|) w  −  m g  +  F_b
        M_z = J_r * r̈_d  +  (X_ud − Y_vd) u v  −  (N_r + N_rc|r|) r
    """

    def __init__(self, dt: float = 0.05):
        self.dt = dt

        # PD gains for desired acceleration computation (body frame)
        self.Kp = np.array([3.0, 3.0, 2.5, 4.0])   # [surge, sway, heave, yaw]
        self.Kd = np.array([1.5, 1.5, 1.2, 2.0])

        # Effective inertias
        self.M_u = m - X_ud
        self.M_v = m - Y_vd
        self.M_w = m - Z_wd
        self.J_r = I_zz - N_rd

        self.prev_pos_err = np.zeros(4)
        self.prev_vel     = np.zeros(4)

    def control(self, state: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        state : (9,) current state
        ref   : (4,) [x_ref, y_ref, z_ref, ψ_ref]

        Returns
        -------
        cmd : (4,) [X, Y, Z, M_z]
        """
        x, y, z = state[0], state[1], state[2]
        cos_psi, sin_psi = state[3], state[4]
        u, v, w, r = state[5], state[6], state[7], state[8]
        psi = np.arctan2(sin_psi, cos_psi)

        # ── Position errors in body frame ─────────────────────────────────────
        e_world = np.array([ref[0] - x, ref[1] - y, ref[2] - z])
        e_surge =  cos_psi * e_world[0] + sin_psi * e_world[1]
        e_sway  = -sin_psi * e_world[0] + cos_psi * e_world[1]
        e_heave =  e_world[2]
        e_psi   = ssa(ref[3] - psi)

        pos_err = np.array([e_surge, e_sway, e_heave, e_psi])
        d_pos_err = (pos_err - self.prev_pos_err) / self.dt
        self.prev_pos_err = pos_err.copy()

        # ── Desired accelerations (PD law) ────────────────────────────────────
        accel_d = self.Kp * pos_err + self.Kd * d_pos_err

        # ── Feedforward force inversion ───────────────────────────────────────
        X  = (self.M_u * accel_d[0]
              - (m - Y_vd) * v * r
              - (X_u + X_uc * abs(u)) * u)

        Y  = (self.M_v * accel_d[1]
              + (m - X_ud) * u * r
              - (Y_v + Y_vc * abs(v)) * v)

        Z  = (self.M_w * accel_d[2]
              - (Z_w + Z_wc * abs(w)) * w
              - m * g + F_bouy)

        Mz = (self.J_r * accel_d[3]
              + (X_ud - Y_vd) * u * v
              - (N_r + N_rc * abs(r)) * r)

        cmd = np.array([X, Y, Z, Mz])
        cmd[:3] = np.clip(cmd[:3], -MAX_FORCE,  MAX_FORCE)
        cmd[3]  = np.clip(cmd[3],  -MAX_TORQUE, MAX_TORQUE)
        return cmd

    def reset(self):
        self.prev_pos_err[:] = 0.0
        self.prev_vel[:] = 0.0
