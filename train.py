import pandas as pd
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import wandb

# home-grown
import utils as ut
import model as mod

# TODO different data loaders for different conditions
# load all four dataframes and concat into one df
# each data frame from a different 2020 month (i.e., march, april, june, november)

# select which months to load
# slice(0,4,1) selects all months
filter = slice(0, 4, 1)
df_itc_all = ut.load_and_concat_raw_data(filter)
df_itc_all.drop("Unnamed: 0", axis=1, inplace=True)
# make participant ids unique (i.e., they were re-used across sessions, but likely not from the same participant)
df_itc_all = ut.unique_participant_ids(df_itc_all)
n_participants_total = df_itc_all["sid_unique"].nunique()
# note. all participants provided exactly 195 trials
n_trials = 195

# shuffle trial ids and participant ids to create dfs with theoretical upper bounds on learning
df_itc_original, df_itc_id_nohist, df_itc_shared_hist, df_itc_shared_nohist = ut.make_conditions(
    df_itc_all, n_trials, n_participants_total)

# convert to 3d torch arrays that can be used by model
col_pid = ["sid_unique"]
cols_x = ["right_time", "right_val", "left_time", "left_val"]
col_y = ["right_picked"]
X_original, y_original = ut.format_to_torch(
    df_itc_original, col_pid, cols_x, col_y)
X_id_nohist, y_id_nohist = ut.format_to_torch(
    df_itc_id_nohist, col_pid, cols_x, col_y)
X_shared_hist, y_shared_hist = ut.format_to_torch(
    df_itc_shared_hist, col_pid, cols_x, col_y)
X_shared_nohist, y_shared_nohist = ut.format_to_torch(
    df_itc_shared_nohist, col_pid, cols_x, col_y)


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
Y_shifted = torch.cat(
    [torch.zeros(num_participants, 1, 1), Y[:, :-1, :]], dim=1)
X = torch.cat([X, Y_shifted], dim=-1)

# create torch data loader
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = mod.CausalEncoder().to(device)
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
