import numpy as np
import torch
import torch.nn as nn
from cvxopt import matrix
from cvxopt import solvers
from cvxopt import sparse
from cvxopt import spmatrix
from torch.autograd import grad as torch_grad


def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        # nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class VarianceDiscriminator(torch.nn.Module):
    def __init__(self, data_dim, gamma=0.1, K=-1):

        super().__init__()
        parameters = {
            "z_dim"         : 32,
            "size"          : 16,
            "nfilter"       : 4,
            "nfilter_max"   : 16,
            "data_dim"      : data_dim,
            "conv_filters"  : 20,
        }

        self.D = ResNet_D(parameters['data_dim'][-1],
                          parameters['size'],
                          nfilter=parameters['nfilter'],
                          nfilter_max=parameters['nfilter_max'])

        self.D.apply(weights_init_D)
        self.losses = {
            'D' : [],
            'WD': [],
            'G' : []
        }
        self.num_steps = 0
        self.gen_steps = 0
        # put in the shape of a dataset sample
        self.data_dim = parameters['data_dim'][0] * parameters['data_dim'][1] * parameters['data_dim'][2]
        self.criterion = torch.nn.MSELoss()
        self.mone = torch.FloatTensor([-1])
        self.tensorboard_counter = 0

        if K <= 0:
            self.K = 1 / self.data_dim
        else:
            self.K = K
        self.Kr = np.sqrt(self.K)
        self.LAMBDA = 2 * self.Kr * gamma * 2

        # the following can be initialized once batch_size is known, so use the function initialize_solver for this.
        self.batch_size = None
        self.A = None
        self.pStart = None
        self.batch_size = None

    def initialize_solver(self, batch_size):
        self.batch_size = batch_size
        self.c, self.A, self.pStart = self._prepare_linear_programming_solver_(self.batch_size)

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

    def _approx_OT_(self, sol, device):
        # Compute the OT mapping for each fake dataset
        ResMat = np.array(sol['z']).reshape((self.batch_size, self.batch_size))
        mapping = torch.from_numpy(np.argmax(ResMat, axis=0)).long().to(device)

        return mapping

    def _optimal_transport_regularization_(self, output_fake, fake, real_fake_diff):
        output_fake_grad = torch.ones(output_fake.size()).to(fake.device)
        gradients = torch_grad(outputs=output_fake,
                               inputs=fake,
                               grad_outputs=output_fake_grad,
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True)[0]
        n = gradients.size(0)
        RegLoss = 0.5 * ((gradients.view(n, -1).norm(dim=1) / (2 * self.Kr) - self.Kr / 2 * real_fake_diff.view(n, -1).norm(dim=1)).pow(2)).mean()

        return RegLoss

    def _critic_deep_regression_(self, data_generated, data_real, data_condition, opt_iterations=1):
        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        self.D.train()

        # Get generated fake dataset
        generated_data = torch.cat((data_generated, data_condition), dim=3)

        real_data = torch.cat((data_real, data_condition), dim=3)

        # compute wasserstein distance
        distance = self._quadratic_wasserstein_distance_(real_data, generated_data)
        # solve linear programming problem
        sol = self._linear_programming_(distance, self.batch_size)
        # approximate optimal transport
        mapping = self._approx_OT_(sol, data_generated.device)
        real_ordered = real_data[mapping]  # match real and fake
        real_fake_diff = real_ordered - generated_data

        # construct target
        target = torch.from_numpy(np.array(sol['x'])).float()
        target = target.squeeze().to(data_generated.device)

        for i in range(opt_iterations):
            generated_data.requires_grad_()
            if generated_data.grad is not None:
                generated_data.grad.data.zero_()
            output_real = self.D(real_data)
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

        # this is supposed to be the wasserstein distance
        wasserstein_distance = output_R_mean - output_F_mean
        self.losses['WD'].append(float(wasserstein_distance.data))

        return total_loss

    def _generator_train_iteration(self, data_generated, data_condition):
        for p in self.D.parameters():
            p.requires_grad = False  # freeze critic

        fake = torch.cat((data_generated, data_condition), dim=3)
        output_fake = self.D(fake)
        output_F_mean_after = output_fake.mean(0).view(1)

        self.losses['G'].append(float(output_F_mean_after.data))

        return output_F_mean_after

    def train_step(self, data_generated, data_real, data_condition):
        self.num_steps += 1
        loss_critic = self._critic_deep_regression_(data_generated.detach(), data_real, data_condition)
        loss_generator = self._generator_train_iteration(data_generated, data_condition)

        return loss_critic, loss_generator * -1


class ResNet_D(nn.Module):
    def __init__(self, data_dim, size, nfilter=64, nfilter_max=512, res_ratio=0.1):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.size = size

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio)
        ]

        self.fc_input = nn.Linear(data_dim, 3 * size * size)

        for i in range(1, nlayers + 1):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]

        self.conv_img = nn.Conv2d(1, 1 * nf, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)

        self.fc = nn.Linear(1568, 1)  # this needs to be changes everytime the window length is changes. It would be nice if this could be done dynamically.

    def forward(self, x):
        batch_size = x.size(0)

        # out = self.fc_input(x)
        # out = self.relu(out).view(batch_size, 3, self.size, self.size)

        out = self.relu((self.conv_img(x)))
        out = self.resnet(out)
        out = out.view(batch_size, -1)
        out = self.fc(out)

        return out


class ResNetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, bn=True, res_ratio=0.1):
        super().__init__()
        # Attributes
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.res_ratio = res_ratio

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_0 = nn.BatchNorm2d(self.fhidden)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_1 = nn.BatchNorm2d(self.fout)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)
            if self.bn:
                self.bn2d_s = nn.BatchNorm2d(self.fout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn2d_0(dx)
        dx = self.relu(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn2d_1(dx)
        out = self.relu(x_s + self.res_ratio * dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn2d_s(x_s)
        else:
            x_s = x
        return x_s
