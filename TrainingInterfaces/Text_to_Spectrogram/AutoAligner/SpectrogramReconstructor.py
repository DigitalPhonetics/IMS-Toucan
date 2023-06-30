import torch
import torch.multiprocessing

from Utility.utils import make_non_pad_mask


class SpectrogramReconstructor(torch.nn.Module):

    def __init__(self,
                 n_mels=72,
                 num_symbols=145,
                 speaker_embedding_dim=192,
                 hidden_dim=128):
        super().__init__()
        self.spectral_reconstruction = torch.nn.Sequential(torch.nn.Linear(num_symbols + speaker_embedding_dim, hidden_dim),
                                                           torch.nn.Tanh(),
                                                           torch.nn.Linear(hidden_dim, hidden_dim),
                                                           torch.nn.Tanh(),
                                                           torch.nn.Linear(hidden_dim, hidden_dim),
                                                           torch.nn.Tanh(),
                                                           torch.nn.Linear(hidden_dim, n_mels))
        self.l1_criterion = torch.nn.L1Loss(reduction="none")
        self.l2_criterion = torch.nn.MSELoss(reduction="none")

    def forward(self, x, lens, ys):
        x = self.spectral_reconstruction(x)
        out_masks = make_non_pad_mask(lens).unsqueeze(-1).to(ys.device)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= ys.size(0) * ys.size(2)
        l1_loss = self.l1_criterion(x, ys).mul(out_weights).masked_select(out_masks).sum()
        l2_loss = self.l2_criterion(x, ys).mul(out_weights).masked_select(out_masks).sum()
        return l1_loss + l2_loss


if __name__ == '__main__':
    print(sum(p.numel() for p in SpectrogramReconstructor().parameters() if p.requires_grad))