"""Dynamic Mode Decomposition (DMD) model.

Learns a linear discrete map  x_{k+1} = A x_k + B u_k  via DMD.
This is the simplest data-driven approach and serves as a linear baseline.
"""
import numpy as np
from .base import BaseModel


class DMDModel(BaseModel):
    name = "DMD"

    def __init__(self, nx=8, nu=4, rank=None):
        self.nx = nx; self.nu = nu
        self.rank = rank
        self.A = None; self.B = None

    def train(self, X, U, Xn):
        # Stack [X; U] and solve  Xn = [A B] [X; U]^T
        Omega = np.hstack([X, U]).T          # (nx+nu, N)
        Xn_T = Xn.T                          # (nx, N)

        # Truncated SVD for robustness
        Uo, So, Vt = np.linalg.svd(Omega, full_matrices=False)
        r = self.rank if self.rank else min(len(So), self.nx + self.nu)
        r = min(r, len(So))
        Uo = Uo[:, :r]; So = So[:r]; Vt = Vt[:r, :]

        AB = Xn_T @ Vt.T @ np.diag(1.0 / So) @ Uo.T   # (nx, nx+nu)
        self.A = AB[:, :self.nx]
        self.B = AB[:, self.nx:]

    def predict(self, x, u):
        xn = self.A @ x + self.B @ u
        xn[3] = (xn[3] + np.pi) % (2 * np.pi) - np.pi
        return xn
