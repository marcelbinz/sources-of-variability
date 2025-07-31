import pandas as pd
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import wandb

# home-grown
import utils as ut

# TODO different data loaders for different conditions
# load all four dataframes and concat into one df
# each data frame from a different 2020 month (i.e., march, april, june, november)
#df_itc_all = ut.load_and_concat_raw_data()
#df_itc_all.drop("Unnamed: 0", axis=1, inplace=True)

run = wandb.init(project="source-of-variablity")

batch_size = 32
num_epochs = 100
lr = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create pytorch dataset
num_features = 4
num_targets = 1
num_participants = 1000
num_trials = 100
# careful: assumes all participants have equal number of trials
X = torch.randn(num_participants, num_trials, num_features)
Y = torch.bernoulli(0.5 * torch.ones(num_participants, num_trials, 1))
Y_shifted = torch.cat([torch.zeros(num_participants, 1, 1), Y[:, :-1, :]], dim = 1)
X = torch.cat([X, Y_shifted], dim = -1)

# create torch data loader
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x):
        S = x.size(1)
        return x + self.pos(torch.arange(S, device=x.device))[None, :, :]

class CausalEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=4, ff=256, dropout=0.0, out_dim=1, in_dim=5, max_len=100):
        super().__init__()
        layer = TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.enc = TransformerEncoder(layer, num_layers)
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

model = CausalEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

for epoch in tqdm(range(num_epochs)):
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"train/loss_step": loss.item()})

    torch.save(model, 'models/epoch=' + str(epoch) + '.pth')
