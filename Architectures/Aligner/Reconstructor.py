import torch
import torch.multiprocessing

from Utility.utils import make_non_pad_mask


class Reconstructor(torch.nn.Module):

    def __init__(self,
                 n_features=128,
                 num_symbols=145,
                 speaker_embedding_dim=192,
                 hidden_dim=256):
        super().__init__()
        self.in_proj = torch.nn.Linear(num_symbols + speaker_embedding_dim, hidden_dim)
        self.hidden_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, n_features)
        self.l1_criterion = torch.nn.L1Loss(reduction="none")

    def forward(self, x, lens, ys):
        x = self.in_proj(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.hidden_proj(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.out_proj(x)
        out_masks = make_non_pad_mask(lens).unsqueeze(-1).to(ys.device)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= ys.size(0) * ys.size(2)
        return self.l1_criterion(x, ys).mul(out_weights).masked_select(out_masks).sum()


if __name__ == '__main__':
    print(sum(p.numel() for p in Reconstructor().parameters() if p.requires_grad))