import torch
from utils.utils import AverageMeter
from termcolor import colored


class ExperienceReplay:
    def __init__(self, model, opt, buffer, epoch, eps_mem_batch, mem_iters=1):
        self.model = model
        self.opt = opt
        self.criterion = torch.nn.MSELoss()
        self.buffer = buffer
        # self.device = 'cuda' if torch.cuda.is_available() else 'gpu'
        self.epoch = epoch
        self.eps_mem_batch = eps_mem_batch
        self.mem_iters = mem_iters

        print(colored('ER initialised', 'green'))

    def train(self, batch_x, batch_y):
        """
        modify it to one step, not one task
        So x_train, y_train are data & targets of 1 mini batch --> which is cache

        """

        # set up model
        self.model = self.model.train()

        # # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()

        for ep in range(self.epoch):

            for j in range(self.mem_iters):
                logits = self.model.forward(batch_x)
                loss = self.criterion(logits, batch_y)
                losses_batch.update(loss, batch_y.size(0))
                self.opt.zero_grad()
                loss.backward()

                # Note here !!!
                mem_x, mem_y = self.buffer.random_retrieve(num_retrieve=self.eps_mem_batch)

                if mem_x.size(0) > 0:
                    mem_logits = self.model.forward(mem_x)
                    loss_mem = self.criterion(mem_logits, mem_y)
                    losses_mem.update(loss_mem, mem_y.size(0))
                    loss_mem.backward()

                self.opt.step()

