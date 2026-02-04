import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x):
        S = x.size(1)
        assert S <= self.pos.num_embeddings, f"Sequence length {S} exceeds max_len {self.pos.num_embeddings}"
        return x + self.pos(torch.arange(S, device=x.device))[None, :, :]


class CausalEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=4, ff=256, dropout=0.0,
                 out_dim=1, in_dim=5, max_len=195, masktype="causal", windowsize=5):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.in_proj = nn.Linear(in_dim, d_model)
        self.out_proj = nn.Linear(d_model, out_dim)
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len)
        self.masktype = masktype  # or "windowed_causal"
        self.windowsize = windowsize  # for windowed causal mask

    def mask(self, masktype, S, n, device):
        if masktype == "causal":
            return self.causal_mask(S, device)
        elif masktype == "windowed_causal":
            return self.windowed_causal_mask(S, n, device)
        else:
            raise ValueError(f"Unknown mask type: {masktype}")

    @staticmethod
    def causal_mask(S, device):
        return torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)

    @staticmethod
    def windowed_causal_mask(S, n, device):
        # Mask everything outside the window [i - (n-1), i]
        # i.e., look back n steps (and also consider the current one, therefore n-1)
        return (
            torch.tril(torch.full((S, S), float("-inf"), device=device), diagonal=-(n - 1)) +
            torch.triu(torch.full((S, S), float(
                "-inf"), device=device), diagonal=1)
        )

    def forward(self, x):
        S = x.size(1)
        mask = self.mask(self.masktype, S, self.windowsize, x.device)

        h = self.in_proj(x)
        h = self.pos_enc(h)
        h = self.enc(h, mask=mask)

        return self.out_proj(h)
