
from utils.utils import maybe_cuda
import torch
import random
import numpy as np


class Buffer(torch.nn.Module):
    """
    Totally online, and there is no concept of "task" here.
    Regression, no classes.
    """
    def __init__(self, model, buffer_size, input_size, output_size):
        super().__init__()

        self.model = model
        self.buffer_size = buffer_size
        # self.device = "cuda" if self.params.cuda else "cpu"
        self.current_index = 0
        self.n_seen_so_far = 0

        buffer_input = torch.FloatTensor(buffer_size, input_size).fill_(0)
        buffer_target = torch.FloatTensor(buffer_size, output_size).fill_(0)

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_input', buffer_input)
        self.register_buffer('buffer_target', buffer_target)

    # def update(self, x, y, **kwargs):
    #     pass
    #
    # def retrieve(self, **kwargs):
    #     pass

    def reservoir_update(self, x, y):
        """
        (x, y) is the single current sample
        """

        # when buffer is not full, add the sample
        if self.current_index < self.buffer_size:
            self.buffer_input[self.current_index] = x
            self.buffer_target[self.current_index] = y

            self.current_index += 1

        else:
            r = random.randint(0, self.n_seen_so_far)
            if r < self.buffer_size:
                self.buffer_input[r] = x
                self.buffer_target[r] = y

        self.n_seen_so_far += 1

    def random_retrieve(self, num_retrieve, excl_indices=None, return_indices=False):
        filled_indices = np.arange(self.current_index)
        if excl_indices is not None:
            excl_indices = list(excl_indices)
        else:
            excl_indices = []
        valid_indices = np.setdiff1d(filled_indices, np.array(excl_indices))
        num_retrieve = min(num_retrieve, valid_indices.shape[0])
        indices = torch.from_numpy(np.random.choice(valid_indices, num_retrieve, replace=False)).long()

        x = self.buffer_input[indices]
        y = self.buffer_target[indices]

        if return_indices:
            return x, y, indices
        else:
            return x, y
