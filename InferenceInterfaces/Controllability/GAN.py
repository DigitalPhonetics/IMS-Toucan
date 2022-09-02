import torch

from InferenceInterfaces.Controllability.dataset.speaker_embeddings_dataset import SpeakerEmbeddingsDataset
from InferenceInterfaces.Controllability.wgan.init_wgan import create_wgan


class GanWrapper:
    def __init__(self, path_dataset, path_wgan) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path_dataset = path_dataset
        self.path_wgan = path_wgan

        self.dataset = SpeakerEmbeddingsDataset(feature_path=path_dataset, device=self.device)

        self.wgan = self.load_model(path_wgan)
        self.U = self.compute_controllability()

        self.z = self.wgan.G.module.sample_latent(1, 32)

    def load_model(self, path):
        gan_checkpoint = torch.load(path, map_location="cpu")
        gan_parameters = gan_checkpoint['model_parameters']
        gan = create_wgan(parameters=gan_parameters, device=self.device)
        
        gan.G.load_state_dict(gan_checkpoint['generator_state_dict'])
        gan.D.load_state_dict(gan_checkpoint['critic_state_dict'])

        return gan
    
    def compute_controllability(self, n_samples=500000):
        samples_generated, intermediate, z = self.wgan.sample_generator(num_samples=n_samples, nograd=True, return_intermediate=True)
        samples_generated = samples_generated.cpu()
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
        embed_original = self.wgan.G.module.forward(self.z.cuda())

        embed_original = self.inverse_normalize(
            embed_original.cpu(), 
            self.dataset.mean.cpu().unsqueeze(0),
            self.dataset.std.cpu().unsqueeze(0)
        )
        return embed_original
    
    def modify_embed(self, x):
        self.wgan.G.eval()
        z_new = self.z.squeeze() + torch.matmul(self.U.solution.t(), x)
        embed_modified = self.wgan.G.module.forward(z_new.unsqueeze(0).cuda())
        embed_modified = self.inverse_normalize(
            embed_modified.cpu(), 
            self.dataset.mean.cpu().unsqueeze(0),
            self.dataset.std.cpu().unsqueeze(0)
        )
        return embed_modified

    def inverse_normalize(self, tensor, mean, std):
        return tensor * std + mean
