"""Gaussian Process Regression dynamics model.

Learns x_{k+1} = GP(x_k, u_k) using scikit-learn GPs.
One GP per output dimension (independently trained).
Uses a subset of training data for scalability.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from .base import BaseModel


class GPModel(BaseModel):
    name = "GP"

    def __init__(self, nx=8, nu=4, max_train=800):
        self.nx = nx; self.nu = nu
        self.max_train = max_train
        self.gps = []
        self._x_mean = None; self._x_std = None
        self._u_mean = None; self._u_std = None

    def train(self, X, U, Xn):
        # Subsample for GP scalability
        n = len(X)
        if n > self.max_train:
            idx = np.random.choice(n, self.max_train, replace=False)
            X = X[idx]; U = U[idx]; Xn = Xn[idx]

        self._x_mean = X.mean(0); self._x_std = X.std(0) + 1e-8
        self._u_mean = U.mean(0); self._u_std = U.std(0) + 1e-8

        Xn_ = (X - self._x_mean) / self._x_std
        Un_ = (U - self._u_mean) / self._u_std
        inp = np.hstack([Xn_, Un_])

        self.gps = []
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(0.01)
        for j in range(self.nx):
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=2,
                normalize_y=True, alpha=1e-6)
            gp.fit(inp, Xn[:, j])
            self.gps.append(gp)

    def predict(self, x, u):
        xn = (x - self._x_mean) / self._x_std
        un = (u - self._u_mean) / self._u_std
        inp = np.concatenate([xn, un]).reshape(1, -1)
        out = np.array([gp.predict(inp)[0] for gp in self.gps])
        return out

    def predict_with_std(self, x, u):
        xn = (x - self._x_mean) / self._x_std
        un = (u - self._u_mean) / self._u_std
        inp = np.concatenate([xn, un]).reshape(1, -1)
        means, stds = [], []
        for gp in self.gps:
            m, s = gp.predict(inp, return_std=True)
            means.append(m[0]); stds.append(s[0])
        return np.array(means), np.array(stds)
