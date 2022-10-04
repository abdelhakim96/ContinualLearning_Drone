
import torch


class Cache:
    def __init__(self, size):
        self.size = size
        self.queue = []
        self.counter = 0

    def initialize(self, samples):
        self.queue.append(samples)
        print(self.queue)

    @property
    def n_samples(self):
        return len(self.queue)

    def update(self, x):
        if self.n_samples < self.size:
            self.queue.append(x)
        else:
            del self.queue[0]
            self.queue.append(x)
        self.counter += 1

    def load_batch(self):
        x, y = self.queue[0]  # tensor
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        if self.n_samples > 1:
            for i in range(1, self.n_samples):
                x = torch.cat([x, self.queue[i][0].unsqueeze(0)], dim=0)
                y = torch.cat([y, self.queue[i][1].unsqueeze(0)], dim=0)

        return x, y
