"""Standard Neural Network forward dynamics model."""
import numpy as np
import torch
import torch.nn as nn
from .base import BaseModel

class NNModel(BaseModel):
    name = "NN"

    def __init__(self, nx=8, nu=4, hidden=128, epochs=200, lr=1e-3):
        self.nx = nx
        self.nu = nu
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cpu')
        self.net = nn.Sequential(
            nn.Linear(nx + nu, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, nx),
        ).to(self.device)
        self._trained = False
        self._x_mean = None; self._x_std = None
        self._u_mean = None; self._u_std = None
        self._y_mean = None; self._y_std = None

    def _normalize(self, X, U, Xn):
        self._x_mean = X.mean(0); self._x_std = X.std(0) + 1e-8
        self._u_mean = U.mean(0); self._u_std = U.std(0) + 1e-8
        self._y_mean = Xn.mean(0); self._y_std = Xn.std(0) + 1e-8
        Xn_ = (X - self._x_mean) / self._x_std
        Un_ = (U - self._u_mean) / self._u_std
        Yn_ = (Xn - self._y_mean) / self._y_std
        return Xn_, Un_, Yn_

    def train(self, X, U, Xn):
        Xn_, Un_, Yn_ = self._normalize(X, U, Xn)
        inp = torch.tensor(np.hstack([Xn_, Un_]), dtype=torch.float32)
        tgt = torch.tensor(Yn_, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        ds = torch.utils.data.TensorDataset(inp, tgt)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad()
                nn.functional.mse_loss(self.net(xb), yb).backward()
                opt.step()
            sched.step()
        self._trained = True

    def predict(self, x, u):
        xn = (x - self._x_mean) / self._x_std
        un = (u - self._u_mean) / self._u_std
        inp = torch.tensor(np.concatenate([xn, un]), dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            yn = self.net(inp).numpy()
        return yn * self._y_std + self._y_mean
