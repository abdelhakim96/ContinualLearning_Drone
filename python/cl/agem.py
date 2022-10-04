import torch
from utils.utils import AverageMeter
from termcolor import colored


class AGEM:
    def __init__(self, model, opt, buffer, epoch, eps_mem_batch, mem_iters=1):
        self.model = model
        self.opt = opt
        self.criterion = torch.nn.MSELoss()
        self.buffer = buffer
        self.epoch = epoch
        self.eps_mem_batch = eps_mem_batch
        self.mem_iters = mem_iters

        self.losses_batch = AverageMeter()
        self.losses_mem = AverageMeter()

        print(colored('AGEM initialised', 'green'))

    def train(self, batch_x, batch_y):
        """
        modify it to one step, not one task
        So x_train, y_train are data & targets of 1 mini batch --> which is cache

        """

        # set up model
        self.model = self.model.train()

        for ep in range(self.epoch):

            for j in range(self.mem_iters):
                logits = self.model.forward(batch_x)
                loss = self.criterion(logits, batch_y)
                self.losses_batch.update(loss, batch_y.size(0))
                self.opt.zero_grad()
                loss.backward()

                mem_x, mem_y = self.buffer.random_retrieve(num_retrieve=self.eps_mem_batch)

                if mem_x.size(0) > 0:
                    params = [p for p in self.model.parameters() if p.requires_grad]

                    # gradient computed using current batch
                    grad = [p.grad.clone() for p in params]

                    mem_logits = self.model.forward(mem_x)
                    loss_mem = self.criterion(mem_logits, mem_y)
                    self.losses_mem.update(loss_mem, mem_y.size(0))
                    loss_mem.backward()

                    # gradient computed using memory samples
                    grad_ref = [p.grad.clone() for p in params]

                    # inner product of grad and grad_ref
                    prod = sum([torch.sum(g * g_r) for g, g_r in zip(grad, grad_ref)])
                    if prod < 0:
                        prod_ref = sum([torch.sum(g_r ** 2) for g_r in grad_ref])
                        # do projection
                        grad = [g - prod / prod_ref * g_r for g, g_r in zip(grad, grad_ref)]
                    # replace params' grad
                    for g, p in zip(grad, params):
                        p.grad.data.copy_(g)

                self.opt.step()
