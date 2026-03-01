"""
BlueROV2 Simulation Model — 4-DOF (surge, sway, heave, yaw)

State vector x ∈ R^9:
    x[0:3]  = [x, y, z]          — NED position (m)
    x[3:5]  = [cos(ψ), sin(ψ)]   — heading encoded on unit circle
    x[5:9]  = [u, v, w, r]       — body-frame velocities (m/s, rad/s)

Control input u ∈ R^4:
    u[0] = X   — surge force  (N)
    u[1] = Y   — sway force   (N)
    u[2] = Z   — heave force  (N)
    u[3] = M_z — yaw moment   (N·m)
"""

import numpy as np
from .parameters import (
    m   as m_nom,
    g, F_bouy,
    X_ud, Y_vd, Z_wd, N_rd,
    X_u  as X_u_nom,  X_uc as X_uc_nom,
    Y_v  as Y_v_nom,  Y_vc as Y_vc_nom,
    Z_w  as Z_w_nom,  Z_wc as Z_wc_nom,
    N_r  as N_r_nom,  N_rc as N_rc_nom,
    I_zz, MAX_FORCE, MAX_TORQUE,
)


def ssa(angle: float) -> float:
    """Smallest signed angle: maps angle to [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _bluerov_deriv_params(x, u, disturbance,
                           m_p, X_u_p, X_uc_p,
                           Y_v_p, Y_vc_p, Z_w_p, Z_wc_p,
                           N_r_p, N_rc_p):
    """Core derivative computation with explicit parameters."""
    cos_psi = x[3];  sin_psi = x[4]
    u_vel, v_vel, w_vel, r = x[5], x[6], x[7], x[8]

    X, Y, Z, Mz = u[0], u[1], u[2], u[3]
    if disturbance is not None:
        X += disturbance[0]; Y += disturbance[1]
        Z += disturbance[2]; Mz += disturbance[3]

    M_u = m_p - X_ud
    M_v = m_p - Y_vd
    M_w = m_p - Z_wd
    J_r = I_zz - N_rd

    x_dot_pos  = cos_psi * u_vel - sin_psi * v_vel
    y_dot_pos  = sin_psi * u_vel + cos_psi * v_vel
    z_dot_pos  = w_vel
    cos_psi_d  = -sin_psi * r
    sin_psi_d  =  cos_psi * r

    u_dot = (X + (m_p - Y_vd) * v_vel * r
             + (X_u_p + X_uc_p * abs(u_vel)) * u_vel) / M_u

    v_dot = (Y - (m_p - X_ud) * u_vel * r
             + (Y_v_p + Y_vc_p * abs(v_vel)) * v_vel) / M_v

    w_dot = (Z + (Z_w_p + Z_wc_p * abs(w_vel)) * w_vel
             + m_p * g - F_bouy) / M_w

    r_dot = (Mz - (X_ud - Y_vd) * u_vel * v_vel
             + (N_r_p + N_rc_p * abs(r)) * r) / J_r

    return np.array([
        x_dot_pos, y_dot_pos, z_dot_pos,
        cos_psi_d, sin_psi_d,
        u_dot, v_dot, w_dot, r_dot,
    ])


class BlueROV2:
    """
    BlueROV2 simulator with support for parameter uncertainty,
    ocean-current disturbances, and sensor noise.
    """

    def __init__(self, dt: float = 0.05, init_state: np.ndarray = None):
        self.dt = dt

        # Nominal parameters (copied so we can scale them)
        self._m   = m_nom
        self._X_u  = X_u_nom;  self._X_uc = X_uc_nom
        self._Y_v  = Y_v_nom;  self._Y_vc = Y_vc_nom
        self._Z_w  = Z_w_nom;  self._Z_wc = Z_wc_nom
        self._N_r  = N_r_nom;  self._N_rc = N_rc_nom

        # Effective (possibly perturbed) parameters
        self.m   = self._m
        self.X_u  = self._X_u;  self.X_uc = self._X_uc
        self.Y_v  = self._Y_v;  self.Y_vc = self._Y_vc
        self.Z_w  = self._Z_w;  self.Z_wc = self._Z_wc
        self.N_r  = self._N_r;  self.N_rc = self._N_rc

        if init_state is None:
            self.state = np.zeros(9)
            self.state[3] = 1.0  # cos(0) = 1
        else:
            self.state = init_state.copy()

        self.disturbance_body = np.zeros(4)

    def _deriv(self, x, u):
        return _bluerov_deriv_params(
            x, u, self.disturbance_body,
            self.m, self.X_u, self.X_uc,
            self.Y_v, self.Y_vc,
            self.Z_w, self.Z_wc,
            self.N_r, self.N_rc,
        )

    def set_disturbance(self, force_pct: float = 0.0):
        """Set ocean-current disturbance as % of MAX_FORCE."""
        self.disturbance_body = np.array([
            force_pct / 100.0 * MAX_FORCE,
            force_pct / 100.0 * MAX_FORCE * 0.5,
            0.0, 0.0,
        ])

    def set_uncertainty(self, factor: float = 0.0):
        """Scale hydrodynamic parameters. factor=0.5 → 1.5× nominal."""
        scale = 1.0 + factor
        self.m    = self._m   * scale
        self.X_u  = self._X_u  * scale;  self.X_uc = self._X_uc * scale
        self.Y_v  = self._Y_v  * scale;  self.Y_vc = self._Y_vc * scale
        self.Z_w  = self._Z_w  * scale;  self.Z_wc = self._Z_wc * scale
        self.N_r  = self._N_r  * scale;  self.N_rc = self._N_rc * scale

    def reset_uncertainty(self):
        self.m   = self._m
        self.X_u  = self._X_u;  self.X_uc = self._X_uc
        self.Y_v  = self._Y_v;  self.Y_vc = self._Y_vc
        self.Z_w  = self._Z_w;  self.Z_wc = self._Z_wc
        self.N_r  = self._N_r;  self.N_rc = self._N_rc

    def step(self, control: np.ndarray,
             noise_std: float = 0.0,
             clip: bool = True) -> tuple:
        """Advance simulation by one dt using RK4."""
        if clip:
            u_clip = np.clip(control[:3], -MAX_FORCE,  MAX_FORCE)
            t_clip = np.clip(control[3:], -MAX_TORQUE, MAX_TORQUE)
            control = np.concatenate([u_clip, t_clip])

        # RK4
        k1 = self._deriv(self.state, control)
        k2 = self._deriv(self.state + self.dt/2 * k1, control)
        k3 = self._deriv(self.state + self.dt/2 * k2, control)
        k4 = self._deriv(self.state + self.dt   * k3, control)
        self.state = self.state + self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

        # Re-normalise heading
        c, s = self.state[3], self.state[4]
        norm  = np.sqrt(c**2 + s**2) + 1e-9
        self.state[3] = c / norm
        self.state[4] = s / norm

        state_true = self.state.copy()
        if noise_std > 0.0:
            noise = np.random.randn(9) * noise_std
            noise[3:5] = 0.0
            state_noisy = state_true + noise
            c2, s2 = state_noisy[3], state_noisy[4]
            n2 = np.sqrt(c2**2 + s2**2) + 1e-9
            state_noisy[3] = c2 / n2
            state_noisy[4] = s2 / n2
        else:
            state_noisy = state_true.copy()

        return state_noisy, state_true

    def get_psi(self) -> float:
        return np.arctan2(self.state[4], self.state[3])

    def reset(self, init_state: np.ndarray = None):
        if init_state is None:
            self.state = np.zeros(9)
            self.state[3] = 1.0
        else:
            self.state = init_state.copy()
        self.disturbance_body = np.zeros(4)
        self.reset_uncertainty()
