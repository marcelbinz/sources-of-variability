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
    def __init__(self, d_model=64, nhead=4, num_layers=4, ff=256, dropout=0.0, out_dim=1, in_dim=5, max_len=195):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.in_proj = nn.Linear(in_dim, d_model)
        self.out_proj = nn.Linear(d_model, out_dim)
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len)

    @staticmethod
    def causal_mask(S, device):
        return torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)

    def forward(self, x):
        S = x.size(1)
        mask = self.causal_mask(S, x.device)

        h = self.in_proj(x)
        h = self.pos_enc(h)
        h = self.enc(h, mask=mask)

        return self.out_proj(h)
