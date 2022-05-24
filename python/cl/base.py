from abc import abstractmethod
import abc
import numpy as np
import torch


class ContinualLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract module which is inherited by each and every continual learning algorithm.
    '''

    def __init__(self, model, opt, params):  # opt : optimizer
        super(ContinualLearner, self).__init__()
        self.model = model
        self.criterion = torch.nn.MSELoss()  # Qz
        self.opt = opt
        self.epoch = params.epoch
        self.batch = params.batch


    # # training of each agent is different, so 'train_learner' will be defined in the py file of each agent
    @abstractmethod
    def train_learner(self, x_train, y_train):
        pass

    def forward(self, x):
        return self.model.forward(x)
