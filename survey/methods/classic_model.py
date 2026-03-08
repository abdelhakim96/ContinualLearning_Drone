"""Classic (analytical) BlueROV2 dynamics model — ground truth baseline."""
import numpy as np
from .base import BaseModel
from survey.src.bluerov_sim import rk4

class ClassicModel(BaseModel):
    name = "Classic"

    def __init__(self, dt=0.05):
        self.dt = dt

    def train(self, X, U, Xn):
        pass  # no training needed — uses known equations

    def predict(self, x, u):
        return rk4(x, u, self.dt)
