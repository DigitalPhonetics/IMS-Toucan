import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cvxopt import matrix
from cvxopt import solvers
from cvxopt import sparse
from cvxopt import spmatrix
from torch.autograd import grad as torch_grad
from tqdm import tqdm


class WassersteinGanQuadraticCost:

    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, criterion, epochs, n_max_iterations,
                 data_dimensions, batch_size, device, gamma=0.1, K=-1, milestones=[150000, 250000], lr_anneal=1.0):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {
            'D' : [],
            'WD': [],
            'G' : []
        }
        self.num_steps = 0
        self.gen_steps = 0
        self.epochs = epochs
        self.n_max_iterations = n_max_iterations
        # put in the shape of a dataset sample
        self.data_dim = data_dimensions[0] * data_dimensions[1] * data_dimensions[2]
        self.batch_size = batch_size
        self.device = device
        self.criterion = criterion
        self.mone = torch.FloatTensor([-1]).to(device)
        self.tensorboard_counter = 0

        if K <= 0:
            self.K = 1 / self.data_dim
        else:
            self.K = K
        self.Kr = np.sqrt(self.K)
        self.LAMBDA = 2 * self.Kr * gamma * 2

        self.G = nn.DataParallel(self.G.to(self.device))
        self.D = nn.DataParallel(self.D.to(self.device))

        self.schedulerD = self._build_lr_scheduler_(self.D_opt, milestones, lr_anneal)
        self.schedulerG = self._build_lr_scheduler_(self.G_opt, milestones, lr_anneal)

        self.c, self.A, self.pStart = self._prepare_linear_programming_solver_(self.batch_size)

    def _build_lr_scheduler_(self, optimizer, milestones, lr_anneal, last_epoch=-1):
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_anneal, last_epoch=-1)
        return scheduler

    def _quadratic_wasserstein_distance_(self, real, generated):
        num_r = real.size(0)
        num_f = generated.size(0)
        real_flat = real.view(num_r, -1)
        fake_flat = generated.view(num_f, -1)

        real3D = real_flat.unsqueeze(1).expand(num_r, num_f, self.data_dim)
        fake3D = fake_flat.unsqueeze(0).expand(num_r, num_f, self.data_dim)
        # compute squared L2 distance
        dif = real3D - fake3D
        dist = 0.5 * dif.pow(2).sum(2).squeeze()

        return self.K * dist

    def _prepare_linear_programming_solver_(self, batch_size):
        A = spmatrix(1.0, range(batch_size), [0] * batch_size, (batch_size, batch_size))
        for i in range(1, batch_size):
            Ai = spmatrix(1.0, range(batch_size), [i] * batch_size, (batch_size, batch_size))
            A = sparse([A, Ai])

        D = spmatrix(-1.0, range(batch_size), range(batch_size), (batch_size, batch_size))
        DM = D
        for i in range(1, batch_size):
            DM = sparse([DM, D])

        A = sparse([[A], [DM]])

        cr = matrix([-1.0 / batch_size] * batch_size)
        cf = matrix([1.0 / batch_size] * batch_size)
        c = matrix([cr, cf])

        pStart = {}
        pStart['x'] = matrix([matrix([1.0] * batch_size), matrix([-1.0] * batch_size)])
        pStart['s'] = matrix([1.0] * (2 * batch_size))

        return c, A, pStart

    def _linear_programming_(self, distance, batch_size):
        b = matrix(distance.cpu().double().detach().numpy().flatten())
        sol = solvers.lp(self.c, self.A, b, primalstart=self.pStart, solver='glpk',
                         options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})
        offset = 0.5 * (sum(sol['x'])) / batch_size
        sol['x'] = sol['x'] - offset
        self.pStart['x'] = sol['x']
        self.pStart['s'] = sol['s']

        return sol

    def _approx_OT_(self, sol):
        # Compute the OT mapping for each fake dataset
        ResMat = np.array(sol['z']).reshape((self.batch_size, self.batch_size))
        mapping = torch.from_numpy(np.argmax(ResMat, axis=0)).long().to(self.device)

        return mapping

    def _optimal_transport_regularization_(self, output_fake, fake, real_fake_diff):
        output_fake_grad = torch.ones(output_fake.size()).to(self.device)
        gradients = torch_grad(outputs=output_fake, inputs=fake,
                               grad_outputs=output_fake_grad,
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
        n = gradients.size(0)
        RegLoss = 0.5 * ((gradients.view(n, -1).norm(dim=1) / (2 * self.Kr) - self.Kr / 2 * real_fake_diff.view(n,
                                                                                                                -1).norm(
            dim=1)).pow(2)).mean()
        fake.requires_grad = False

        return RegLoss

    def _critic_deep_regression_(self, images, opt_iterations=1):
        images = images.to(self.device)

        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        self.G.train()
        self.D.train()

        # Get generated fake dataset
        generated_data = self.sample_generator(self.batch_size)

        # compute wasserstein distance
        distance = self._quadratic_wasserstein_distance_(images, generated_data)
        # solve linear programming problem
        sol = self._linear_programming_(distance, self.batch_size)
        # approximate optimal transport
        mapping = self._approx_OT_(sol)
        real_ordered = images[mapping]  # match real and fake
        real_fake_diff = real_ordered - generated_data

        # construct target
        target = torch.from_numpy(np.array(sol['x'])).float()
        target = target.squeeze().to(self.device)

        for i in range(opt_iterations):
            self.D.zero_grad()  # ???
            self.D_opt.zero_grad()
            generated_data.requires_grad_()
            if generated_data.grad is not None:
                generated_data.grad.data.zero_()
            output_real = self.D(images)
            output_fake = self.D(generated_data)
            output_real, output_fake = output_real.squeeze(), output_fake.squeeze()
            output_R_mean = output_real.mean(0).view(1)
            output_F_mean = output_fake.mean(0).view(1)

            L2LossD_real = self.criterion(output_R_mean[0], target[:self.batch_size].mean())
            L2LossD_fake = self.criterion(output_fake, target[self.batch_size:])
            L2LossD = 0.5 * L2LossD_real + 0.5 * L2LossD_fake

            reg_loss_D = self._optimal_transport_regularization_(output_fake, generated_data, real_fake_diff)

            total_loss = L2LossD + self.LAMBDA * reg_loss_D

            self.losses['D'].append(float(total_loss.data))

            total_loss.backward()
            self.D_opt.step()

        # this is supposed to be the wasserstein distance
        wasserstein_distance = output_R_mean - output_F_mean
        self.losses['WD'].append(float(wasserstein_distance.data))

    def _generator_train_iteration(self, batch_size):
        for p in self.D.parameters():
            p.requires_grad = False  # freeze critic

        self.G.zero_grad()
        self.G_opt.zero_grad()

        if isinstance(self.G, torch.nn.parallel.DataParallel):
            z = self.G.module.sample_latent(batch_size, self.G.module.z_dim)
        else:
            z = self.G.sample_latent(batch_size, self.G.z_dim)
        z.requires_grad = True

        fake = self.G(z)
        output_fake = self.D(fake)
        output_F_mean_after = output_fake.mean(0).view(1)

        self.losses['G'].append(float(output_F_mean_after.data))

        output_F_mean_after.backward(self.mone)
        self.G_opt.step()

        self.schedulerD.step()
        self.schedulerG.step()

    def _train_epoch(self, data_loader, writer, experiment):
        for i, data in enumerate(tqdm(data_loader)):
            images = data[0]
            speaker_ids = data[1]
            self.num_steps += 1
            # self.tensorboard_counter += 1
            if self.gen_steps >= self.n_max_iterations:
                return
            self._critic_deep_regression_(images)
            self._generator_train_iteration(images.size(0))

            D_loss_avg = np.average(self.losses['D'])
            G_loss_avg = np.average(self.losses['G'])
            wd_avg = np.average(self.losses['WD'])

    def train(self, data_loader, writer, experiment=None):
        self.G.train()
        self.D.train()

        for epoch in range(self.epochs):
            if self.gen_steps >= self.n_max_iterations:
                return
            time_start_epoch = time.time()
            self._train_epoch(data_loader, writer, experiment)

            D_loss_avg = np.average(self.losses['D'])

            time_end_epoch = time.time()

        return self

    def sample_generator(self, num_samples, nograd=False, return_intermediate=False):
        self.G.eval()
        if isinstance(self.G, torch.nn.parallel.DataParallel):
            latent_samples = self.G.module.sample_latent(num_samples, self.G.module.z_dim)
        else:
            latent_samples = self.G.sample_latent(num_samples, self.G.z_dim)
        latent_samples = latent_samples.to(self.device)
        if nograd:
            with torch.no_grad():
                generated_data = self.G(latent_samples, return_intermediate=return_intermediate)
        else:
            generated_data = self.G(latent_samples)
        self.G.train()
        if return_intermediate:
            return generated_data[0].detach(), generated_data[1], latent_samples
        return generated_data.detach()

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]

    def save_model_checkpoint(self, model_path, model_parameters, timestampStr):
        # dateTimeObj = datetime.now()
        # timestampStr = dateTimeObj.strftime("%d-%m-%Y-%H-%M-%S")
        name = '%s_%s' % (timestampStr, 'wgan')
        model_filename = os.path.join(model_path, name)
        torch.save({
            'generator_state_dict'       : self.G.state_dict(),
            'critic_state_dict'          : self.D.state_dict(),
            'gen_optimizer_state_dict'   : self.G_opt.state_dict(),
            'critic_optimizer_state_dict': self.D_opt.state_dict(),
            'model_parameters'           : model_parameters,
            'iterations'                 : self.num_steps
        }, model_filename)
