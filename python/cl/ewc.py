from utils.utils import AverageMeter
import torch
from termcolor import colored


class EWC_pp:
    def __init__(self, model, opt, epoch, lambda_, alpha, steps1=1000, steps2=20):
        self.model = model
        self.opt = opt
        self.criterion = torch.nn.MSELoss()
        self.epoch = epoch
        self.losses_batch = AverageMeter()
        self.weights = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.lambda_ = lambda_
        self.alpha = alpha
        self.batch_counter = 0
        self.update_after = steps1
        self.running_fisher_update_after = steps2
        self.prev_params = {}  # parameters of previous model
        self.running_fisher = self.init_fisher()
        self.tmp_fisher = self.init_fisher()
        # normalized fisher. Calculated from running_fisher when finishing a task. Used in Loss
        self.normalized_fisher = self.init_fisher()

        print(colored('EWC initialised', 'green'))

    def train(self, batch_x, batch_y):
        """
        modify it to one step, not one task
        So x_train, y_train are data & targets of 1 mini batch --> which is cache

        """

        # set up model
        self.model = self.model.train()

        for ep in range(self.epoch):
            self.batch_counter += 1

            # update the running fisher --> Update after many steps, not every step
            if self.batch_counter % self.running_fisher_update_after == 0:
                self.update_running_fisher()

            out = self.model(batch_x)
            loss = self.total_loss(out, batch_y)

            # update tracker
            self.losses_batch.update(loss.item(), batch_y.size(0))

            # backward
            self.opt.zero_grad()
            loss.backward()

            # accumulate the fisher of current batch
            self.accum_fisher()
            self.opt.step()

        # save params after learning sufficient batches
        if self.batch_counter % self.update_after == 0:
            # save the params of the previous model
            for n, p in self.weights.items():
                self.prev_params[n] = p.clone().detach()

            # update normalized fisher
            max_fisher = max([torch.max(m) for m in
                              self.running_fisher.values()])  # global maximum imp value across all the parameters
            min_fisher = min([torch.min(m) for m in self.running_fisher.values()])
            for n, p in self.running_fisher.items():
                self.normalized_fisher[n] = (p - min_fisher) / (max_fisher - min_fisher + 1e-32)

    def total_loss(self, inputs, targets):
        loss = self.criterion(inputs, targets)
        if len(self.prev_params) > 0:
            # add regularization loss
            reg_loss = 0
            for n, p in self.weights.items():
                reg_loss += (self.normalized_fisher[n] * (p - self.prev_params[n]) ** 2).sum()
            loss += self.lambda_ * reg_loss
        return loss

    def init_fisher(self):
        """
        Fisher Information Matrix: one matrix for each parameter
        Dictionary: key: learnable parameter's name, value: its importance matrix, FIM
        """
        return {n: p.clone().detach().fill_(0) for n, p in self.model.named_parameters() if p.requires_grad}

    def update_running_fisher(self):
        for n, p in self.running_fisher.items():
            self.running_fisher[n] = (1. - self.alpha) * p \
                                     + 1. / self.running_fisher_update_after * self.alpha * self.tmp_fisher[n]
        # reset the accumulated fisher
        self.tmp_fisher = self.init_fisher()

    def accum_fisher(self):
        for n, p in self.tmp_fisher.items():
            p += self.weights[n].grad ** 2