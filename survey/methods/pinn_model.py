"""Physics-Informed Neural Network dynamics model.

The PINN learns a residual correction on top of the analytical dynamics,
plus enforces physics-consistency via a dynamics residual penalty.
"""
import numpy as np
import torch
import torch.nn as nn
from .base import BaseModel
from survey.src.bluerov_sim import dynamics as analytical_dynamics, rk4

class PINNModel(BaseModel):
    name = "PINN"

    def __init__(self, nx=8, nu=4, hidden=128, epochs=200, lr=1e-3,
                 physics_weight=0.1, dt=0.05):
        self.nx = nx; self.nu = nu
        self.epochs = epochs; self.lr = lr
        self.dt = dt
        self.physics_weight = physics_weight
        self.residual_net = nn.Sequential(
            nn.Linear(nx + nu, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, nx),
        )
        self._trained = False
        self._x_mean = None; self._x_std = None
        self._u_mean = None; self._u_std = None
        self._r_mean = None; self._r_std = None

    def train(self, X, U, Xn):
        # Compute analytical predictions and residuals
        Xn_ana = np.array([rk4(X[i], U[i], self.dt) for i in range(len(X))])
        R = Xn - Xn_ana  # residual the NN must learn

        self._x_mean = X.mean(0); self._x_std = X.std(0) + 1e-8
        self._u_mean = U.mean(0); self._u_std = U.std(0) + 1e-8
        self._r_mean = R.mean(0); self._r_std = R.std(0) + 1e-8

        Xn_ = (X - self._x_mean) / self._x_std
        Un_ = (U - self._u_mean) / self._u_std
        Rn_ = (R - self._r_mean) / self._r_std

        inp = torch.tensor(np.hstack([Xn_, Un_]), dtype=torch.float32)
        tgt = torch.tensor(Rn_, dtype=torch.float32)

        opt = torch.optim.Adam(self.residual_net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        ds = torch.utils.data.TensorDataset(inp, tgt)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

        self.residual_net.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad()
                pred = self.residual_net(xb)
                data_loss = nn.functional.mse_loss(pred, yb)
                # Physics loss: residual should be small (close to analytical)
                physics_loss = torch.mean(pred ** 2)
                loss = data_loss + self.physics_weight * physics_loss
                loss.backward()
                opt.step()
            sched.step()
        self._trained = True

    def predict(self, x, u):
        x_ana = rk4(x, u, self.dt)
        xn = (x - self._x_mean) / self._x_std
        un = (u - self._u_mean) / self._u_std
        inp = torch.tensor(np.concatenate([xn, un]), dtype=torch.float32)
        self.residual_net.eval()
        with torch.no_grad():
            rn = self.residual_net(inp).numpy()
        residual = rn * self._r_std + self._r_mean
        return x_ana + residual
