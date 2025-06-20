from copy import deepcopy
from pathlib import Path

import torch
from torch.optim import Adam
from torch.nn import Module, ModuleList

from Architectures.RectifiedFlow.rectified_flow import RectifiedFlow

from ema_pytorch import EMA
#from accelerate import Accelerator

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# reflow wrapper

class Reflow(Module):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        frozen_model: RectifiedFlow | None = None,
        *,
        batch_size = 16,

    ):
        super().__init__()
        model, data_shape = rectified_flow.model, rectified_flow.data_shape
        assert exists(data_shape), '`data_shape` must be defined in RectifiedFlow'

        self.batch_size = batch_size
        self.data_shape = data_shape

        self.model = rectified_flow

        if not exists(frozen_model):
            # make a frozen copy of the model and set requires grad to be False for all parameters for safe measure

            frozen_model = deepcopy(rectified_flow)

            for p in frozen_model.parameters():
                p.detach_()

        self.frozen_model = frozen_model

    def device(self):
        return next(self.parameters()).device

    def parameters(self):
        return self.model.parameters() # omit frozen model

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def forward(self):

        noise = torch.randn((self.batch_size, *self.data_shape), device = self.device)
        sampled_output = self.frozen_model.sample(noise = noise)

        # the coupling in the paper is (noise, sampled_output)

        loss = self.model(sampled_output, noise = noise)

        return loss

# reflow trainer
"""
class ReflowTrainer(Module):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        *,
        num_train_steps = 70_000,
        learning_rate = 3e-4,
        batch_size = 16,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int = 100,
        checkpoint_every: int = 1000,
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        assert not rectified_flow.use_consistency, 'reflow is not needed if using consistency flow matching'

        self.model = Reflow(rectified_flow)

        if self.is_main:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        self.optimizer = Adam(rectified_flow.parameters(), lr = learning_rate, **adam_kwargs)

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        self.batch_size = batch_size
        self.num_train_steps = num_train_steps

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def log(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def log_images(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )

        torch.save(save_package, str(self.checkpoints_folder / path))

    def forward(self):

        for ind in range(self.num_train_steps):
            step = ind + 1

            self.model.train()

            loss = self.model(batch_size = self.batch_size)

            self.log(loss, step = step)

            self.accelerator.print(f'[{step}] reflow loss: {loss.item():.3f}')
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main:
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:
                if divisible_by(step, self.save_results_every):
                    self.ema_model.ema_model.data_shape = self.model.data_shape

                    with torch.no_grad():
                        sampled = self.ema_model.sample(batch_size = self.num_samples)

                    sampled = rearrange(sampled, '(row col) c h w -> c (row h) (col w)', row = self.num_sample_rows)
                    sampled.clamp_(0., 1.)

                    self.log_images(sampled, step = step)

                    save_image(sampled, str(self.results_folder / f'results.{step}.png'))

                if divisible_by(step, self.checkpoint_every):
                    self.save(f'checkpoint.{step}.pt')

            self.accelerator.wait_for_everyone()

        print('reflow training complete')
"""