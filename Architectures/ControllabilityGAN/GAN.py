import torch

from Architectures.ControllabilityGAN.wgan.init_wgan import create_wgan


class GanWrapper:

    def __init__(self, path_wgan, device):
        self.device = device
        self.path_wgan = path_wgan

        self.mean = None
        self.std = None
        self.wgan = None
        self.normalize = True

        self.load_model(path_wgan)

        self.U = self.compute_controllability()

        self.z_list = list()

        for _ in range(1100):
            self.z_list.append(self.wgan.G.sample_latent(1, self.wgan.G.z_dim, temperature=0.8))
        self.z = self.z_list[0]

    def set_latent(self, seed):
        self.z = self.z = self.z_list[seed]

    def reset_default_latent(self):
        self.z = self.wgan.G.sample_latent(1, self.wgan.G.z_dim, temperature=0.8)

    def load_model(self, path):
        gan_checkpoint = torch.load(path, map_location="cpu")

        self.wgan = create_wgan(parameters=gan_checkpoint['model_parameters'], device=self.device)
        # Create a new state dict without 'module.' prefix
        new_state_dict_G = {}
        for key, value in gan_checkpoint['generator_state_dict'].items():
            # Remove 'module.' prefix
            new_key = key.replace('module.', '')
            new_state_dict_G[new_key] = value

        new_state_dict_D = {}
        for key, value in gan_checkpoint['critic_state_dict'].items():
            # Remove 'module.' prefix
            new_key = key.replace('module.', '')
            new_state_dict_D[new_key] = value

        self.wgan.G.load_state_dict(new_state_dict_G)
        self.wgan.D.load_state_dict(new_state_dict_D)

        self.mean = gan_checkpoint["dataset_mean"]
        self.std = gan_checkpoint["dataset_std"]

    def compute_controllability(self, n_samples=100000):
        _, intermediate, z = self.wgan.sample_generator(num_samples=n_samples, nograd=True, return_intermediate=True)
        intermediate = intermediate.cpu()
        z = z.cpu()
        U = self.controllable_speakers(intermediate, z)
        return U

    def controllable_speakers(self, intermediate, z):
        pca = torch.pca_lowrank(intermediate)
        mu = intermediate.mean()
        X = torch.matmul((intermediate - mu), pca[2])
        U = torch.linalg.lstsq(X, z)
        return U

    def get_original_embed(self):
        self.wgan.G.eval()
        embed_original = self.wgan.G.module.forward(self.z.to(self.device))

        if self.normalize:
            embed_original = inverse_normalize(
                embed_original.cpu(),
                self.mean.cpu().unsqueeze(0),
                self.std.cpu().unsqueeze(0)
            )
        return embed_original

    def modify_embed(self, x):
        self.wgan.G.eval()
        z_new = self.z.squeeze() + torch.matmul(self.U.solution.t(), x)
        embed_modified = self.wgan.G.forward(z_new.unsqueeze(0).to(self.device))
        if self.normalize:
            embed_modified = inverse_normalize(
                embed_modified.cpu(),
                self.mean.cpu().unsqueeze(0),
                self.std.cpu().unsqueeze(0)
            )
        return embed_modified


def inverse_normalize(tensor, mean, std):
    return tensor * std + mean
