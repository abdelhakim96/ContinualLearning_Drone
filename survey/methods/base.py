"""Base class for all learned dynamics models."""
import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    All models implement:
      train(X, U, Xn)  — learn from data
      predict(x, u)    — one-step prediction  x_{k+1} = f(x_k, u_k)
    """
    name: str = "base"

    @abstractmethod
    def train(self, X: np.ndarray, U: np.ndarray, Xn: np.ndarray):
        ...

    @abstractmethod
    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        ...

    def predict_batch(self, X, U):
        return np.array([self.predict(X[i], U[i]) for i in range(len(X))])

    def rollout(self, x0, U_seq, dt=0.05):
        """Multi-step open-loop rollout."""
        traj = [x0.copy()]
        x = x0.copy()
        for u in U_seq:
            x = self.predict(x, u)
            traj.append(x.copy())
        return np.array(traj)
