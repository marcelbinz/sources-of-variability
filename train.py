import os
import argparse
import random
import numpy as np
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


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--condition_name",
        type=str,
        default="original",
        choices=["original", "id_nohist", "shared_hist", "shared_nohist"],
        help="Which condition to run.",
    )
    aa(
        "--rnd_seed",
        type=int,
        default=1,
        help="Random seed for reproducibility.",
    )
    aa(
        "--d_model",
        type=int,
        default=64,
        choices=[4, 8, 16, 32, 64],
    )
    aa(
        "--d_ff",
        type=int,
        default=256,
        choices=[32, 64, 128, 256],
    )
    aa(
        "--is_testcase",
        type=bool,
        default=True,
        help="If True, only use 200 participants to test the code. If False, use all participants.",
    )
    aa(
        "--num_layers",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help="Number of transformer encoder layers.",
    )
    args = parser.parse_args()
    return args

    

def run(condition_name, d_model=64, d_ff=256, is_testcase=True, rnd_seed=1, num_layers=4):

    # ===================== Setup =====================

    condition_names = ["original", "id_nohist", "shared_hist", "shared_nohist"]
    condition_id = condition_names.index(condition_name)

    load_from_osf = False  # when local file is available

    # Create Model Dirs
    save_dir = f"models/condition_{condition_name}/d_model={d_model}/d_ff={d_ff}/num_layers={num_layers}/"
    os.makedirs(save_dir, exist_ok=True)

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
        [df_itc_original, df_itc_id_nohist,
            df_itc_shared_hist, df_itc_shared_nohist],
        condition_names
    ))

    batch_size = 32
    num_epochs = 300
    lr = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===================== 1. Model: By-Participant and History =====================

    # just to subset and test, comment otherwise
    if is_testcase:
        n_participants_subset = 10
    else:
        n_participants_subset = n_participants_total

    run = wandb.init(
        entity="mirkothalmann-helmholtz-munich",
        project="source-of-variablity",
        name=f"condition={condition_name}_dmodel={d_model}_dff={d_ff}_numlayers={num_layers}",
        config={
            "subset_participants": n_participants_subset,
            "learning_rate": lr,
            "architecture": "CausalTF with learned Pos. Encoding",
            "dataset": "Agrawal et al. (2023) JEP:G",
            "condition": condition_name,
            "epochs": num_epochs,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "rnd_seed": rnd_seed,
            "is_testcase": is_testcase,
        }
    )

    # select the data from the relevant condition
    y_original_train = l_df_train_dev[condition_id]["y_train"][0:n_participants_subset, :, :]
    X_original_train = l_df_train_dev[condition_id]["X_train"][0:n_participants_subset, :, :]
    y_original_dev = l_df_train_dev[condition_id]["y_dev"][0:n_participants_subset, :, :]
    X_original_dev = l_df_train_dev[condition_id]["X_dev"][0:n_participants_subset, :, :]

    # create torch data loader
    dataset_train = TensorDataset(X_original_train, y_original_train)
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)

    dataset_dev = TensorDataset(X_original_dev, y_original_dev)
    dataloader_dev = DataLoader(
        dataset_dev, batch_size=batch_size, shuffle=True)

    model = mod.CausalEncoder(d_model=d_model, ff=d_ff, num_layers=num_layers).to(device)
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

        if epoch % 50 == 0:
            torch.save(model, save_dir + '/epoch=' + str(epoch) + '.pth')
    wandb.finish()


if __name__ == "__main__":
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    run(
        condition_name=args.condition_name,
        d_model=args.d_model,
        d_ff=args.d_ff,
        is_testcase=args.is_testcase,
        rnd_seed=args.rnd_seed,
        num_layers=args.num_layers
    )
