import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, MSELoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.generators.benchmark_generators import dataset_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive


class SimpleDataset(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return torch.from_numpy(self.input[index]).float(), torch.from_numpy(self.output[index]).float()


scenario = SplitMNIST(n_experiences=5)

# MODEL CREATION
model = SimpleMLP(num_classes=scenario.n_classes)

# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.

# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True),
    confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False,
                             stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    #loggers=[interactive_logger, text_logger, tb_logger]
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    MSELoss(), train_mb_size=10, train_epochs=1, eval_mb_size=100,
    evaluator=eval_plugin)

# define data and class labels
input = np.array([[1, 2],
                  [3, 4]])
output = np.array([[3],
                   [7]])

dataset = SimpleDataset(input, output)
loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)

print(type(dataset))
print(type(loader))

# dnn_input = torch.from_numpy(dnn_input).float()
# dnn_output = torch.from_numpy(dnn_output).float()
#
# trainLoader = DataLoader(train, batch_size=1000, shuffle=True, pin_memory=True)

# TRAINING LOOP
print('Starting experiment...')
results = []
scenario = dataset_benchmark(dataset, dataset)
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience.float())
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(scenario.test_stream))