"""
Experience Replay Buffer using reservoir sampling.

Maintains a fixed-size memory of past (input, output) pairs
sampled uniformly over the stream of encountered data.
"""

import torch
import numpy as np


class ReplayBuffer:
    """
    Reservoir-sampling replay buffer.

    Guarantees a uniform distribution over all seen examples
    regardless of arrival order.
    """

    def __init__(self, capacity: int = 500, in_dim: int = 9, out_dim: int = 4,
                 device: str = 'cpu'):
        self.capacity = capacity
        self.device   = device
        self.in_dim   = in_dim
        self.out_dim  = out_dim

        self.buf_x = torch.zeros(capacity, in_dim,  device=device)
        self.buf_y = torch.zeros(capacity, out_dim, device=device)
        self.n_seen = 0    # total number of samples observed
        self.size   = 0    # current buffer size (≤ capacity)

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Add a mini-batch to the buffer via reservoir sampling.

        Parameters
        ----------
        x : (B, in_dim)
        y : (B, out_dim)
        """
        B = x.size(0)
        for i in range(B):
            self.n_seen += 1
            if self.size < self.capacity:
                self.buf_x[self.size] = x[i]
                self.buf_y[self.size] = y[i]
                self.size += 1
            else:
                j = np.random.randint(0, self.n_seen)
                if j < self.capacity:
                    self.buf_x[j] = x[i]
                    self.buf_y[j] = y[i]

    def sample(self, batch_size: int):
        """
        Randomly sample `batch_size` examples from the buffer.

        Returns
        -------
        (x, y) tensors of shape (batch_size, *) or empty tensors if buffer empty.
        """
        if self.size == 0:
            empty_x = torch.zeros(0, self.in_dim,  device=self.device)
            empty_y = torch.zeros(0, self.out_dim, device=self.device)
            return empty_x, empty_y
        idx = torch.randint(0, self.size, (min(batch_size, self.size),))
        return self.buf_x[idx], self.buf_y[idx]

    def __len__(self):
        return self.size
