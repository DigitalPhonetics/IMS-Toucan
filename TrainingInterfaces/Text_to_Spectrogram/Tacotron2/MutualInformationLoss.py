"""

https://github.com/bfs18/tacotron2

https://arxiv.org/pdf/1909.01145.pdf

Still need to figure out how to use this in conjunction with the articulatory features

"""

from torch import nn


class MIEsitmator(nn.Module):
    def __init__(self, vocab_size, decoder_dim, hidden_size, dropout=0.5):
        super(MIEsitmator, self).__init__()
        self.proj = nn.Sequential(
            LinearNorm(decoder_dim, hidden_size, bias=True, w_init_gain='relu'),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.ctc_proj = LinearNorm(hidden_size, vocab_size + 1, bias=True)
        self.ctc = nn.CTCLoss(blank=vocab_size, reduction='mean')

    def forward(self, decoder_outputs, target_phones, decoder_lengths, target_lengths):
        out = self.proj(decoder_outputs)
        log_probs = self.ctc_proj(out).log_softmax(dim=2)
        log_probs = log_probs.transpose(1, 0)
        ctc_loss = self.ctc(log_probs, target_phones, decoder_lengths, target_lengths)
        return ctc_loss


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
