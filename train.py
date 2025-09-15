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
col_y_shifted = ["right_picked_prev"]

# zscale X
df_itc_all[cols_x] = df_itc_all[cols_x].apply(zscore)

# shuffle trial ids and participant ids to create dfs with theoretical upper bounds on learning
df_itc_original, df_itc_id_nohist, df_itc_shared_hist, df_itc_shared_nohist = ut.make_conditions(
    df_itc_all, n_trials, n_participants_total)

# shift y by one trial such that on trial t, model gets info about y from trial t-1
l_dfs = list(map(ut.shift_y, [
             df_itc_original, df_itc_id_nohist, df_itc_shared_hist, df_itc_shared_nohist]))
df_itc_original, df_itc_id_nohist, df_itc_shared_hist, df_itc_shared_nohist = l_dfs[
    0], l_dfs[1], l_dfs[2], l_dfs[3]

# train dev split and convert to 3d torch arrays that can be used by model

partial_split_and_format = partial(
    ut.split_and_format, splittype="first_vs_second_half", n_trial_split=100,
    col_pid=col_pid, cols_x=cols_x, col_y=col_y, col_y_shifted=col_y_shifted
)

l_df_train_dev = list(map(
    partial_split_and_format,
    [df_itc_original, df_itc_id_nohist, df_itc_shared_hist, df_itc_shared_nohist],
    ["original", "id_nohist", "shared_hist", "shared_nohist"]
))


batch_size = 32
num_epochs = 100
lr = 3e-4
# just to subset and test, comment otherwise
n_participants_total = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 1. Model: By-Participant and History =====================

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="mirkothalmann-helmholtz-munich",
    # Set the wandb project where this run will be logged.
    project="source-of-variablity",
    # Track hyperparameters and run metadata.
    name="condition=original",
    config={
        "subset_participants": n_participants_total,
        "learning_rate": lr,
        "architecture": "CausalTF with learned Pos. Encoding",
        "dataset": "Agrawal et al. (2023) JEP:G",
        "epochs": num_epochs,
    }
)

y_original_train = l_df_train_dev[0]["y_train"][0:n_participants_total, :, :]
X_original_train = l_df_train_dev[0]["X_train"][0:n_participants_total, :, :]
y_original_dev = l_df_train_dev[0]["y_dev"][0:n_participants_total, :, :]
X_original_dev = l_df_train_dev[0]["X_dev"][0:n_participants_total, :, :]

# create torch data loader
dataset_train = TensorDataset(X_original_train, y_original_train)
dataloader_train = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True)

dataset_dev = TensorDataset(X_original_dev, y_original_dev)
dataloader_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True)


model = mod.CausalEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()


num_train_batches = len(dataloader_train)
num_dev_batches = len(dataloader_dev)

for epoch in tqdm(range(num_epochs)):
    train_loss_total = 0
    dev_loss_total = 0
    for batch_x, batch_y in dataloader_train:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        train_loss_total += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for batch_x, batch_y in dataloader_dev:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()

        outputs = model(batch_x)
        loss_dev = criterion(outputs, batch_y)
        dev_loss_total += loss_dev.item()

    wandb.log({
        "epoch": epoch,
        "train/loss_epoch": train_loss_total / num_train_batches,
        "dev/loss_epoch": dev_loss_total / num_dev_batches,
    })

    torch.save(model, 'models/condition_original/epoch=' + str(epoch) + '.pth')
wandb.finish()

# ===================== 2. Model: By-Participant and History =====================

""" 
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="mirkothalmann-helmholtz-munich",
    # Set the wandb project where this run will be logged.
    project="source-of-variablity",
    # Track hyperparameters and run metadata.
    name="condition=ID-NoHistory",
    config={
        "subset_participants": n_participants_total,
        "learning_rate": lr,
        "architecture": "CausalTF with learned Pos. Encoding",
        "dataset": "Agrawal et al. (2023) JEP:G",
        "epochs": num_epochs,
    }
)


y_id_nohist = y_id_nohist[0:n_participants_total, :, :]
X_id_nohist = X_id_nohist[0:n_participants_total, :, :]

y_id_nohist_shifted = torch.cat(
    [torch.zeros(n_participants_total, 1, 1), y_id_nohist[:, :-1, :]], dim=1)
X_id_nohist = torch.cat([X_id_nohist, y_id_nohist_shifted], dim=-1)


# create torch data loader
dataset = TensorDataset(X_id_nohist, y_id_nohist)
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
wandb.finish()

# ===================== 3. Model: Shared and History =====================


run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="mirkothalmann-helmholtz-munich",
    # Set the wandb project where this run will be logged.
    project="source-of-variablity",
    # Track hyperparameters and run metadata.
    name="condition=Shared-History",
    config={
        "subset_participants": n_participants_total,
        "learning_rate": lr,
        "architecture": "CausalTF with learned Pos. Encoding",
        "dataset": "Agrawal et al. (2023) JEP:G",
        "epochs": num_epochs,
    }
)

y_shared_hist = y_shared_hist[0:n_participants_total, :, :]
X_shared_hist = X_shared_hist[0:n_participants_total, :, :]

y_shared_hist_shifted = torch.cat(
    [torch.zeros(n_participants_total, 1, 1), y_shared_hist[:, :-1, :]], dim=1)
X_shared_hist = torch.cat([X_shared_hist, y_shared_hist_shifted], dim=-1)


# create torch data loader
dataset = TensorDataset(X_shared_hist, y_shared_hist)
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
wandb.finish()

# ===================== 4. Model: Shared and No History =====================


run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="mirkothalmann-helmholtz-munich",
    # Set the wandb project where this run will be logged.
    project="source-of-variablity",
    # Track hyperparameters and run metadata.
    name="condition=Shared-NoHistory",
    config={
        "subset_participants": n_participants_total,
        "learning_rate": lr,
        "architecture": "CausalTF with learned Pos. Encoding",
        "dataset": "Agrawal et al. (2023) JEP:G",
        "epochs": num_epochs,
    }
)

y_shared_nohist = y_shared_nohist[0:n_participants_total, :, :]
X_shared_nohist = X_shared_nohist[0:n_participants_total, :, :]

y_shared_nohist_shifted = torch.cat(
    [torch.zeros(n_participants_total, 1, 1), y_shared_nohist[:, :-1, :]], dim=1)
X_shared_nohist = torch.cat([X_shared_nohist, y_shared_nohist_shifted], dim=-1)


# create torch data loader
dataset = TensorDataset(X_shared_nohist, y_shared_nohist)
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
 """
