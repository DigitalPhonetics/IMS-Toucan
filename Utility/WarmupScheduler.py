from torch.optim.lr_scheduler import _LRScheduler


# This is rather suboptimal, because we need to import a protected class. Unfortunately, I don't see another way.


class ToucanWarmupScheduler(_LRScheduler):
    """
    A warmup scheduler that should be called after every batch.
    """

    def __init__(self, optimizer, peak_lr=0.0002, warmup_steps=20000, max_steps=200000, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.max_steps = max_steps
        self.plateau = self.warmup_steps * 4
        self.last_lr = 0.0
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if step_num <= self.warmup_steps:
            lr = self.peak_lr * min(step_num / self.warmup_steps, 1.0)
            self.last_lr = lr
            return [lr for _ in self.base_lrs]
        elif step_num < self.warmup_steps + self.plateau:
            self.last_lr = self.peak_lr
            return [self.peak_lr for _ in self.base_lrs]
        else:
            scale = 1 - (((step_num - (self.warmup_steps + self.plateau)) / self.max_steps) / (self.max_steps / 10))
            self.last_lr = max(self.last_lr * scale, 1e-7)
            return [self.last_lr for _ in self.base_lrs]


class WarmupScheduler(_LRScheduler):
    """
    The WarmupLR scheduler
    This scheduler is almost same as NoamLR Scheduler except for following difference:
    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    Note that the maximum lr equals to optimizer.lr in this scheduler.

    Taken from ESPnet
    """

    def __init__(self, optimizer, warmup_steps=25000, last_epoch=-1):
        self.warmup_steps = warmup_steps
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [lr * self.warmup_steps ** 0.5 * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5) for lr in
                self.base_lrs]


if __name__ == '__main__':
    lrs = list()
    warmup_steps = 30000
    peak_lr = 0.0005
    max_steps = 800000
    plateau_size = warmup_steps * 5
    for step_num in range(max_steps):
        if step_num <= warmup_steps:
            lr = peak_lr * min(step_num / warmup_steps, 1.0)
            lrs.append(lr)
        elif step_num < warmup_steps + plateau_size:
            lrs.append(peak_lr)
        else:
            scale = 1 - (((step_num - (warmup_steps + plateau_size)) / max_steps) / (max_steps / 10))
            lrs.append(max(lrs[-1] * scale, 1e-7))
    import matplotlib.pyplot as plt

    plt.plot(lrs)
    plt.show()
