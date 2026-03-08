"""Sparse Identification of Nonlinear Dynamics (SINDy) model.

Builds a library of candidate nonlinear functions and uses
sequential thresholded least-squares (STLSQ) to find a sparse
dynamics model  dx/dt = Θ(x,u) · ξ.
"""
import numpy as np
from .base import BaseModel

def _build_library(X, U):
    """Build a feature library Θ(x,u) with polynomial and trig terms."""
    n = X.shape[0]
    ones = np.ones((n, 1))
    # State and input columns
    cols = [ones, X, U]
    # Quadratic state terms
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
    # Cross terms x_i * u_j
    for i in range(X.shape[1]):
        for j in range(U.shape[1]):
            cols.append((X[:, i] * U[:, j]).reshape(-1, 1))
    # |vel| * vel  (quadratic drag pattern)
    for i in range(4, 8):  # velocity states
        cols.append((np.abs(X[:, i]) * X[:, i]).reshape(-1, 1))
    # sin/cos of heading
    cols.append(np.sin(X[:, 3:4]))
    cols.append(np.cos(X[:, 3:4]))
    return np.hstack(cols)


def _build_library_single(x, u):
    """Single-sample version of _build_library."""
    X = x.reshape(1, -1); U = u.reshape(1, -1)
    return _build_library(X, U).ravel()


class SINDyModel(BaseModel):
    name = "SINDy"

    def __init__(self, nx=8, nu=4, dt=0.05, threshold=0.05, max_iter=20):
        self.nx = nx; self.nu = nu; self.dt = dt
        self.threshold = threshold
        self.max_iter = max_iter
        self.Xi = None  # coefficient matrix

    def train(self, X, U, Xn):
        # Compute discrete derivatives
        dXdt = (Xn - X) / self.dt
        Theta = _build_library(X, U)
        # Sequential thresholded least squares
        Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]
        for _ in range(self.max_iter):
            small = np.abs(Xi) < self.threshold
            Xi[small] = 0.0
            for j in range(self.nx):
                big = ~small[:, j]
                if big.sum() == 0:
                    continue
                Xi[big, j] = np.linalg.lstsq(
                    Theta[:, big], dXdt[:, j], rcond=None)[0]
        self.Xi = Xi
        self._n_terms = int(np.count_nonzero(Xi))

    def predict(self, x, u):
        theta = _build_library_single(x, u)
        dxdt = theta @ self.Xi
        xn = x + self.dt * dxdt
        # Wrap heading
        xn[3] = (xn[3] + np.pi) % (2 * np.pi) - np.pi
        return xn
