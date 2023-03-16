import random
import torch
from torch.utils.tensorboard import SummaryWriter

class FastSpeech2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(FastSpeech2Logger, self).__init__(logdir)

    def log_training(self, loss, iteration):
            self.add_scalar("training.loss", loss, iteration)
            # self.add_scalar("grad.norm", grad_norm, iteration)
            # self.add_scalar("learning.rate", learning_rate, iteration)
            # self.add_scalar("duration", duration, iteration)
