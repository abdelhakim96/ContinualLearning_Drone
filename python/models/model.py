from torch import nn


class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden_units, num_outputs):
        super().__init__()
        if len(num_hidden_units) == 1:
            self.layers = nn.Sequential(
                nn.Linear(num_inputs, num_hidden_units[0]),
                nn.Tanh(),
                nn.Linear(num_hidden_units[0], num_outputs)
            )
        if len(num_hidden_units) == 2:
            self.layers = nn.Sequential(
                nn.Linear(num_inputs, num_hidden_units[0]),
                nn.ReLU(),
                nn.Linear(num_hidden_units[0], num_hidden_units[1]),
                nn.Tanh(),
                nn.Linear(num_hidden_units[1], num_outputs)
            )
        if len(num_hidden_units) == 3:
            self.layers = nn.Sequential(
                nn.Linear(num_inputs, num_hidden_units[0]),
                nn.ReLU(),
                nn.Linear(num_hidden_units[0], num_hidden_units[1]),
                nn.ReLU(),
                nn.Linear(num_hidden_units[1], num_hidden_units[2]),
                nn.Tanh(),
                nn.Linear(num_hidden_units[2], num_outputs)
            )

    def forward(self, x):
        return self.layers(x)