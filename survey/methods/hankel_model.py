"""Hankel-DMD model.

Extends standard DMD by constructing a time-delay (Hankel) embedding
of the state and input history, capturing higher-order dynamics that
a single-step linear model cannot represent.

x_{k+1} ≈ A_H · z_k + B_H · u_k
where z_k = [x_k; x_{k-1}; ...; x_{k-d+1}; u_{k-1}; ...; u_{k-d+1}]
"""
import numpy as np
from .base import BaseModel


class HankelDMDModel(BaseModel):
    name = "Hankel"

    def __init__(self, nx=8, nu=4, delay=3, rank=None):
        self.nx = nx; self.nu = nu
        self.delay = delay
        self.rank = rank
        self.A_H = None; self.B_H = None
        self._z_dim = None

    def _build_hankel(self, X, U):
        """Build time-delay embedded dataset."""
        d = self.delay
        N = len(X) - d
        z_dim = self.nx * d + self.nu * (d - 1)
        self._z_dim = z_dim

        Z = np.zeros((N, z_dim))
        U_cur = np.zeros((N, self.nu))
        Xn = np.zeros((N, self.nx))

        for i in range(N):
            # Stack delayed states
            states = X[i:i+d].ravel()  # d * nx
            # Stack delayed controls (exclude current)
            controls = U[i:i+d-1].ravel() if d > 1 else np.array([])  # (d-1) * nu
            Z[i] = np.concatenate([states, controls])
            U_cur[i] = U[i + d - 1]
            Xn[i] = X[i + d]  # next state after the window

        return Z, U_cur, Xn

    def train(self, X, U, Xn):
        # Re-create sequential data from X, Xn
        # X[i] → Xn[i] means Xn can serve as X shifted by 1
        all_states = np.vstack([X, Xn[-1:]])
        all_inputs = U

        Z, U_cur, Xn_h = self._build_hankel(all_states, all_inputs)

        Omega = np.hstack([Z, U_cur]).T
        Xn_T = Xn_h.T

        Uo, So, Vt = np.linalg.svd(Omega, full_matrices=False)
        r = self.rank if self.rank else min(len(So), Omega.shape[0])
        r = min(r, len(So))
        Uo = Uo[:, :r]; So = So[:r]; Vt = Vt[:r, :]

        AB = Xn_T @ Vt.T @ np.diag(1.0 / So) @ Uo.T
        self.A_H = AB[:, :self._z_dim]
        self.B_H = AB[:, self._z_dim:]

        # Store last (delay-1) states and controls for prediction init
        self._state_buf = list(all_states[-(self.delay):])
        self._ctrl_buf = list(all_inputs[-(self.delay-1):]) if self.delay > 1 else []

    def predict(self, x, u):
        # Update buffers
        self._state_buf.append(x.copy())
        if len(self._state_buf) > self.delay:
            self._state_buf.pop(0)

        if len(self._state_buf) < self.delay:
            # Not enough history, pad with current state
            while len(self._state_buf) < self.delay:
                self._state_buf.insert(0, x.copy())

        states = np.concatenate(self._state_buf)
        if self.delay > 1:
            self._ctrl_buf.append(u.copy())
            if len(self._ctrl_buf) > self.delay - 1:
                self._ctrl_buf.pop(0)
            while len(self._ctrl_buf) < self.delay - 1:
                self._ctrl_buf.insert(0, u.copy())
            controls = np.concatenate(self._ctrl_buf)
        else:
            controls = np.array([])

        z = np.concatenate([states, controls])
        xn = self.A_H @ z + self.B_H @ u
        xn[3] = (xn[3] + np.pi) % (2 * np.pi) - np.pi
        return xn

    def reset_buffers(self, x0):
        """Reset delay buffers for a new rollout."""
        self._state_buf = [x0.copy()] * self.delay
        self._ctrl_buf = [np.zeros(self.nu)] * max(0, self.delay - 1)
