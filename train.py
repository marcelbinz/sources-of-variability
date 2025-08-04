import os

import pandas as pd
import torch
import wandb

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import zscore
from tqdm import tqdm
from functools import partial

# home-grown
import utils as ut
import model as mod


load_from_osf = False  # when local file is available

# Create Model Dirs
l_save_dirs = ['models/condition_original', "models/condition_id_nohist",
               "models/condition_shared_hist", "models/condition_shared_nohist"]
partial_makedirs = partial(os.makedirs, exist_ok=True)
_ = list(map(partial_makedirs, l_save_dirs))


# ===================== Load Data =====================


# load all four dataframes and concat into one df
# each data frame from a different 2020 month (i.e., march, april, june, november)
if load_from_osf:
    # select which months to load
    # slice(0,4,1) selects all months
    filter = slice(0, 4, 1)
    df_itc_all = ut.load_and_concat_raw_data(filter)
    df_itc_all.drop("Unnamed: 0", axis=1, inplace=True)
    df_itc_all.to_csv("data/full-data.csv")
else:
    df_itc_all = pd.read_csv("data/full-data.csv")
# make participant ids unique (i.e., they were re-used across sessions, but likely not from the same participant)
df_itc_all = ut.unique_participant_ids(df_itc_all)
n_participants_total = df_itc_all["sid_unique"].nunique()
# note. all participants provided exactly 195 trials
n_trials = 195


# ===================== Prepare Data =====================

col_pid = ["sid_unique"]
cols_x = ["right_time", "right_val", "left_time", "left_val"]
col_y = ["right_picked"]

# zscale X
df_itc_all[cols_x] = df_itc_all[cols_x].apply(zscore)

# shuffle trial ids and participant ids to create dfs with theoretical upper bounds on learning
df_itc_original, df_itc_id_nohist, df_itc_shared_hist, df_itc_shared_nohist = ut.make_conditions(
    df_itc_all, n_trials, n_participants_total)

# convert to 3d torch arrays that can be used by model

X_original, y_original = ut.format_to_torch(
    df_itc_original, col_pid, cols_x, col_y)
X_id_nohist, y_id_nohist = ut.format_to_torch(
    df_itc_id_nohist, col_pid, cols_x, col_y)
X_shared_hist, y_shared_hist = ut.format_to_torch(
    df_itc_shared_hist, col_pid, cols_x, col_y)
X_shared_nohist, y_shared_nohist = ut.format_to_torch(
    df_itc_shared_nohist, col_pid, cols_x, col_y)


batch_size = 32
num_epochs = 10  # 100
lr = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 1. Model: By-Participant and History =====================

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="mirkothalmann-helmholtz-munich",
    # Set the wandb project where this run will be logged.
    project="source-of-variablity",
    # Track hyperparameters and run metadata.
    name="run_condition_original",
    config={
        "learning_rate": lr,
        "architecture": "CausalTF with learned Pos. Encoding",
        "dataset": "Agrawal et al. (2023) JEP:G",
        "epochs": num_epochs,
    }
)
wandb.run.tags += ("condition_original", "subset_participants",)

n_participants_total = 100
y_original = y_original[0:n_participants_total, :, :]
X_original = X_original[0:n_participants_total, :, :]

y_original_shifted = torch.cat(
    [torch.zeros(n_participants_total, 1, 1), y_original[:, :-1, :]], dim=1)
X_original = torch.cat([X_original, y_original_shifted], dim=-1)


# create torch data loader
dataset = TensorDataset(X_original, y_original)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = mod.CausalEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

for epoch in tqdm(range(num_epochs)):
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"train/loss_step": loss.item()})

    torch.save(model, 'models/condition_original/epoch=' + str(epoch) + '.pth')


# ===================== 2. Model: By-Participant and History =====================


wandb.run.tags += ("condition_id_nohist", "subset_participants",)

X_id_nohist, y_id_nohist


y_id_nohist = y_id_nohist[0:n_participants_total, :, :]
X_id_nohist = X_id_nohist[0:n_participants_total, :, :]

y_id_nohist_shifted = torch.cat(
    [torch.zeros(n_participants_total, 1, 1), y_id_nohist[:, :-1, :]], dim=1)
X_id_nohist = torch.cat([X_id_nohist, y_id_nohist_shifted], dim=-1)


# create torch data loader
dataset = TensorDataset(X_original, y_id_nohist_shifted)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = mod.CausalEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

for epoch in tqdm(range(num_epochs)):
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"train/loss_step": loss.item()})

    torch.save(model, 'models/condition_id_nohist/epoch=' +
               str(epoch) + '.pth')


# ===================== 3. Model: Shared and History =====================


wandb.run.tags += ("condition_shared_hist", "subset_participants",)


y_shared_hist = y_shared_hist[0:n_participants_total, :, :]
X_shared_hist = X_shared_hist[0:n_participants_total, :, :]

y_shared_hist_shifted = torch.cat(
    [torch.zeros(n_participants_total, 1, 1), y_shared_hist[:, :-1, :]], dim=1)
X_shared_hist = torch.cat([X_shared_hist, y_shared_hist_shifted], dim=-1)


# create torch data loader
dataset = TensorDataset(X_original, y_shared_hist_shifted)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = mod.CausalEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

for epoch in tqdm(range(num_epochs)):
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"train/loss_step": loss.item()})

    torch.save(model, 'models/condition_shared_hist/epoch=' +
               str(epoch) + '.pth')


# ===================== 4. Model: Shared and No History =====================


wandb.run.tags += ("condition_shared_nohist", "subset_participants",)


y_shared_nohist = y_shared_nohist[0:n_participants_total, :, :]
X_shared_nohist = X_shared_nohist[0:n_participants_total, :, :]

y_shared_nohist_shifted = torch.cat(
    [torch.zeros(n_participants_total, 1, 1), y_shared_nohist[:, :-1, :]], dim=1)
X_shared_nohist = torch.cat([X_shared_nohist, y_shared_nohist_shifted], dim=-1)


# create torch data loader
dataset = TensorDataset(X_original, y_shared_nohist_shifted)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = mod.CausalEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

for epoch in tqdm(range(num_epochs)):
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"train/loss_step": loss.item()})

    torch.save(model, 'models/condition_shared_nohist/epoch=' +
               str(epoch) + '.pth')

wandb.finish()
