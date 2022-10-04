import torch
from utils.utils import AverageMeter
import copy
from termcolor import colored


class LwF:
    def __init__(self, model, opt, epoch, lambda_=0.5, step=100, buffer=None, eps_mem_batch=32, mem_iters=1):
        self.model = model
        self.opt = opt
        self.epoch = epoch
        self.criterion = torch.nn.MSELoss()

        # The use of Buffer is optional.
        self.buffer = buffer
        self.eps_mem_batch = eps_mem_batch
        self.mem_iters = mem_iters

        self.batch_counter = 0
        self.update_teacher_after = step  # The number of steps after which updates the teacher model
        self.lambda_ = lambda_
        self.kd_manager = KdManager()

        print(colored('LwF initialised', 'green'))

    def train(self, batch_x, batch_y):
        """
        modify it to one step, not one task
        So x_train, y_train are data & targets of 1 mini batch --> which is cache

        """

        # set up model
        self.model = self.model.train()

        # # setup tracker
        losses_batch = AverageMeter()

        for ep in range(self.epoch):
            self.batch_counter += 1

            for j in range(self.mem_iters):
                outputs = self.model.forward(batch_x)
                loss_new = self.criterion(outputs, batch_y)
                loss_old = self.kd_manager.get_kd_loss(self.model, batch_x)
                loss = loss_new + self.lambda_ * loss_old  # Try someway to make lambda dynamic/adaptive?

                if self.buffer:
                    mem_x, mem_y = self.buffer.random_retrieve(num_retrieve=self.eps_mem_batch)

                    if mem_x.size(0) > 0:
                        mem_logits = self.model.forward(mem_x)
                        loss_mem = self.criterion(mem_logits, mem_y)
                        loss += loss_mem

                # update tracker
                losses_batch.update(loss, batch_y.size(0))
                # backward
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        if self.batch_counter % self.update_teacher_after == 0:
            self.kd_manager.update_teacher(self.model)


class KdManager:
    def __init__(self):
        self.teacher_model = None

    def update_teacher(self, model):
        self.teacher_model = copy.deepcopy(model)

    def get_kd_loss(self, cur_model, x):
        loss = torch.nn.MSELoss()
        if self.teacher_model is not None:

            with torch.no_grad():
                target_scores = self.teacher_model.forward(x)
            scores = cur_model.forward(x)
            dist_loss = loss(scores, target_scores)

        else:
            dist_loss = 0
        return dist_loss
