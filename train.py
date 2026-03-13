import os
import argparse
import random
import numpy as np
import pandas as pd
import polars as pl
import torch
import wandb
import logging
import ast


import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import zscore
from tqdm import tqdm
from functools import partial


# home-grown
import utils as ut
import model as mod
import lazydata as lzdt

os.environ["PYTHONIOENCODING"] = "UTF-8"
os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--dataset_name",
        type=str,
        choices=["itc", "risky"],
        help="'itc' for Agrawal et al. (2023) JEP:G, 'risky' for Peterson et al. (2021) Science",
    )
    aa(
        "--dataset_select",
        type=str,
        choices=["", "repetitions", "no_repetitions"],
        help="Only relevant for 'risky' dataset, whether problem repetitions should be included or not.",
    )
    aa(
        "--condition_name",
        type=str,
        default="original",
        choices=["original", "id_nohist", "shared_hist", "shared_nohist"],
        help="Which condition to run.",
    )
    aa(
        "--swap_colnames",
        type=str,
        help='Quoted list of column name pairs (in a list) to swap within participants, e.g., "[["right_val", "left_val"], ["right_time", "left_time"]]"',
    )
    aa(
        "--shuffle_single_colnames",
        type=str,
        help="Quoted list of single column names to shuffle within participants.",
    )
    aa(
        "--tf",
        type=str,
        default="z",
        choices=["z", "mean_center_only", "log_values"],
        help=""
        "Type of transformation to apply to the data. 'z' for z-scoring, 'mean_center_only' for mean-centering only, "
        "'log_values' for log-transforming followed by z-scoring."
        "Note that log scaling only necessary for itc dataset, not for risky dataset.",
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
        type=str,
        default="True",
        choices=["True", "False"],
        help="If True, only use 200 participants to test the code. If False, use all participants.",
    )
    aa(
        "--num_layers",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help="Number of transformer encoder layers.",
    )
    aa(
        "--masktype",
        type=str,
        default="causal",
        choices=["causal", "windowed_causal"],
        help="Type of attention mask to use in the transformer.",
    )
    aa(
        "--windowsize",
        type=int,
        default=5,
        help="Window size for windowed causal mask.",
    )
    args = parser.parse_args()
    return args


def init_logger(log_file="train-itc.log", level=logging.INFO):
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Prevent duplicate handlers if re-initialized
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def run(
    dataset_name,
    dataset_select,
    condition_name,
    swap_colnames,
    shuffle_single_colnames,
    tf,
    d_model=64,
    d_ff=256,
    is_testcase="True",
    rnd_seed=1,
    num_layers=4,
    masktype="causal",
    windowsize=5,
):

    # ===================== Setup =====================

    condition_names = ["original", "id_nohist", "shared_hist", "shared_nohist"]
    # can be handed over as argument when different data sets are used, but for now, just hardcoded for the itc data set
    condition_id = condition_names.index(condition_name)

    l_swap_colnames = ast.literal_eval(swap_colnames)
    l_shuffle_single_colnames = ast.literal_eval(shuffle_single_colnames)

    # when applied to different tasks as well, likely needs to be adapted to be more generalizable
    swap_varnames = [sc[0].split("_")[-1] for sc in l_swap_colnames if len(sc) > 0]
    # unique variable names that are swapped
    swap_varnames = list(set(swap_varnames))
    l_shuffle_varnames = [
        "_".join(ssc.split("_")[1:]) for ssc in l_shuffle_single_colnames
    ]
    shuffle_variables = swap_varnames + l_shuffle_varnames

    # Create Model Dirs
    more_shuffle = "-".join(shuffle_variables)
    if more_shuffle == "":
        more_shuffle = "nothing"

    if dataset_name == "risky":
        logger = init_logger(
            f"""logs/train-{dataset_name}_{dataset_select}/condition_{condition_name}/more_shuffle_{more_shuffle}-dmodel={d_model}-dff={d_ff}-numlayers={num_layers}.log"""
        )
        save_dir = f"models/dataset_{dataset_name}/condition_{condition_name}/more_shuffle_{more_shuffle}/d_model={d_model}/d_ff={d_ff}/num_layers={num_layers}/masktype={masktype}/windowsize={windowsize}/"
    elif dataset_name == "itc":
        logger = init_logger(
            f"""logs/train-{dataset_name}/condition_{condition_name}/more_shuffle_{more_shuffle}-dmodel={d_model}-dff={d_ff}-numlayers={num_layers}.log"""
        )
        save_dir = f"models/dataset_{dataset_name}_{dataset_select}/condition_{condition_name}/more_shuffle_{more_shuffle}/d_model={d_model}/d_ff={d_ff}/num_layers={num_layers}/masktype={masktype}/windowsize={windowsize}/"

    os.makedirs(save_dir, exist_ok=True)
    logger.info("Logger initialized and ready to roll.")

    logger.info(f"""l_swap_colnames = {l_swap_colnames}""")
    logger.info(f"""l_shuffle_single_colnames = {l_shuffle_single_colnames}""")

    # ===================== Load Data =====================

    df, dict_info = ut.load_sov_dataset(dataset_name, dataset_select)

    # ===================== Prepare Data =====================

    #### SCALING ####
    # zscale individual xs
    # df[dict_info["cols_x"]] = df[dict_info["cols_x"]].apply(zscore)

    # tf of X only required in itc and risky
    if dataset_name in ["itc", "risky"]:
        df = ut.zscore_grouped_cols(df, dict_info["l_colnames_grouped_zscale"], tf=tf)
    # for mm dataset, we only apply z per col, not pairs of cols
    elif dataset_name == "mm":
        df = ut.zscore_single_col_lazy(df, dict_info["l_colnames_single_zscale"])

    #### CONDITIONS ####

    if dataset_name in ["itc", "risky"]:
        df_original, df_id_nohist, df_shared_hist, df_shared_nohist = (
            ut.make_conditions(df)
        )
    elif dataset_name == "mm":
        df_original, df_id_nohist, df_shared_hist, df_shared_nohist = (
            ut.make_conditions_lazy(df)
        )

    batch_size = 32
    num_epochs = 250  # 150
    lr = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===================== 1. Model: By-Participant and History =====================

    # just to subset and test, comment otherwise
    if is_testcase == "True":
        n_participants_subset = 10
    else:
        n_participants_subset = dict_info["n_participants_total"]

    if dataset_name == "risky":
        wandb_name = f"""dataset={dataset_name}_{dataset_select}_condition={condition_name}_more_shuffle={more_shuffle}_tf={tf}_dmodel={d_model}_dff={d_ff}_numlayers={num_layers}_ntrials_train={dict_info["n_trials_train"]}_masktype={masktype}_windowsize={windowsize}"""
        dataset_desc = f"{dataset_name}_{dataset_select}"
    elif dataset_name == "itc":
        wandb_name = f"""dataset={dataset_name}_condition={condition_name}_more_shuffle={more_shuffle}_tf={tf}_dmodel={d_model}_dff={d_ff}_numlayers={num_layers}_ntrials_train={dict_info["n_trials_train"]}_masktype={masktype}_windowsize={windowsize}"""
        dataset_desc = dataset_name
    run = wandb.init(
        entity="mirkothalmann-helmholtz-munich",
        project="source-of-variablity",
        name=wandb_name,
        config={
            "subset_participants": n_participants_subset,
            "learning_rate": lr,
            "architecture": "CausalTF with learned Pos. Encoding",
            "dataset": dataset_desc,
            "condition": condition_name,
            "more_shuffle": more_shuffle,
            "tf": tf,
            "epochs": num_epochs,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "rnd_seed": rnd_seed,
            "is_testcase": is_testcase,
            "n_trials_train": dict_info["n_trials_train"],
            "masktype": masktype,
            "windowsize": windowsize,
        },
    )

    #### LOAD TRAIN AND DEV DATA ####

    if dataset_name in ["itc", "risky"]:
        # can process all data in memory for itc and risky datasets, but not for mm dataset
        # loads all conditions and only then selects the relevant condition
        # this is not possible for the mm dataset
        # also could be improved, in principle

        # shift y by one trial such that on trial t, model gets info about y from trial t-1
        # ordering of dfs in l_dfs remains the same (original, id_nohist, shared_hist, shared_nohist)
        l_dfs = list(
            map(
                ut.shift_y,
                [df_original, df_id_nohist, df_shared_hist, df_shared_nohist],
            )
        )

        #### SHUFFLING INDEPENDENT VARIABLES ####
        if len(l_swap_colnames[0]) > 0:
            for idx, colnames in enumerate(l_swap_colnames):
                f_partial_swap = partial(ut.swap_two_cols, colnames=colnames, rs=idx)
                l_dfs = list(map(f_partial_swap, l_dfs))
        if len(l_shuffle_single_colnames) > 0:
            for colname in l_shuffle_single_colnames:
                f_partial_shuffle = partial(ut.shuffle_single_column, colname=colname)
                l_dfs = list(map(f_partial_shuffle, l_dfs))

        #### SPLIT AND FORMAT ####
        partial_split_and_format = partial(
            ut.split_and_format,
            splittype="first_vs_second_half",
            n_trial_split=dict_info["n_trials_train"],
            col_pid=dict_info["col_pid"],
            cols_x=dict_info["cols_x"],
            col_y=dict_info["col_y"],
            col_y_shifted=dict_info["col_y_shifted"],
        )

        l_df_train_dev = list(map(partial_split_and_format, l_dfs, condition_names))

        #### SELECT CONDITION AND CREATE DATALOADER ####
        # select the data from the relevant condition
        y_train = l_df_train_dev[condition_id]["y_train"][0:n_participants_subset, :, :]
        X_train = l_df_train_dev[condition_id]["X_train"][0:n_participants_subset, :, :]
        mask_train = l_df_train_dev[condition_id]["mask_train"][
            0:n_participants_subset, :
        ]
        y_dev = l_df_train_dev[condition_id]["y_dev"][0:n_participants_subset, :, :]
        X_dev = l_df_train_dev[condition_id]["X_dev"][0:n_participants_subset, :, :]
        mask_dev = l_df_train_dev[condition_id]["mask_dev"][0:n_participants_subset, :]

        # create torch data loader
        dataset_train = TensorDataset(X_train, y_train, mask_train)
        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )

        dataset_dev = TensorDataset(X_dev, y_dev, mask_dev)
        dataloader_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True)

    elif dataset_name == "mm":
        df_use = ut.shift_y_lazy(l_dfs[condition_id])

        ## TODO: implement shuffling for mm dataset as well, currently only implemented for itc and risky datasets

        #### SPLIT AND FORMAT ####
        df_train, df_dev = ut.train_dev_split_lazy(
            df_use,
            splittype="first_vs_second_half",
            n_trial_split=dict_info["n_trials_train"],
        )
        participants = (
            df_train.select(pl.col(dict_info["col_pid"])).unique().to_numpy().flatten()
        )
        dataset_train = lzdt.PolarsLazyDataset(
            lazyframe=df_train,
            participants=participants,
            col_pid=dict_info["col_pid"],
            col_y=dict_info["col_y"],
            cols_x=dict_info["cols_x"],
            col_y_shifted=dict_info["col_y_shifted"],
            participant_subsets=1000,
        )
        # note. batch size is ignored, because dataset returns already sequences from participants
        dataloader_train = DataLoader(dataset_train)

        dataset_dev = lzdt.PolarsLazyDataset(
            lazyframe=df_dev,
            participants=participants,
            col_pid=dict_info["col_pid"],
            col_y=dict_info["col_y"],
            cols_x=dict_info["cols_x"],
            col_y_shifted=dict_info["col_y_shifted"],
            participant_subsets=1000,
        )
        # note. batch size is ignored, because dataset returns already sequences from participants
        dataloader_dev = DataLoader(dataset_dev)

    model = mod.CausalEncoder(
        d_model=d_model,
        ff=d_ff,
        num_layers=num_layers,
        in_dim=dict_info["in_dim"],
        masktype=masktype,
        windowsize=windowsize,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    num_train_batches = len(dataloader_train)
    num_dev_batches = len(dataloader_dev)

    for epoch in tqdm(range(num_epochs)):
        train_loss_total = 0
        dev_loss_total = 0
        n_correct_total_train = 0
        n_correct_total_dev = 0
        n_total_train = 0
        n_total_dev = 0
        for batch_x, batch_y, batch_mask in dataloader_train:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_mask = batch_mask.to(device).bool()

            outputs = model(batch_x)
            # loss
            loss = criterion(outputs[batch_mask], batch_y[batch_mask])
            train_loss_total += loss.item()
            # accuracy
            n_problems = batch_mask.int().sum().item()
            pred_right = (outputs[batch_mask] > 0).int()
            correct = (pred_right == batch_y[batch_mask].int()).sum().item()
            # update total correct and total problems for train accuracy
            n_correct_total_train += correct
            n_total_train += n_problems

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = n_correct_total_train / n_total_train
        # (batch_y.shape[0] * batch_y.shape[1] * batch_y.shape[2])

        for batch_x, batch_y, batch_mask in dataloader_dev:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_mask = batch_mask.to(device).bool()
            outputs = model(batch_x)
            loss_dev = criterion(outputs[batch_mask], batch_y[batch_mask])
            dev_loss_total += loss_dev.item()
            # accuracy
            n_problems = batch_mask.int().sum().item()
            pred_right = (outputs[batch_mask] > 0).int()
            correct = (pred_right == batch_y[batch_mask].int()).sum().item()
            # update total correct and total problems for dev accuracy
            n_correct_total_dev += correct
            n_total_dev += n_problems

        dev_acc = n_correct_total_dev / n_total_dev
        # (batch_y.shape[0] * batch_y.shape[1] * batch_y.shape[2])

        wandb.log(
            {
                "epoch": epoch,
                "train/loss_epoch": train_loss_total / num_train_batches,
                "dev/loss_epoch": dev_loss_total / num_dev_batches,
                "train/acc_epoch": train_acc,
                "dev/acc_epoch": dev_acc,
            }
        )

        if epoch % 50 == 0:
            torch.save(model, save_dir + "/epoch=" + str(epoch) + ".pth")
    wandb.finish()


if __name__ == "__main__":
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    run(
        dataset_name=args.dataset_name,
        dataset_select=args.dataset_select,
        condition_name=args.condition_name,
        swap_colnames=args.swap_colnames,
        shuffle_single_colnames=args.shuffle_single_colnames,
        tf=args.tf,
        d_model=args.d_model,
        d_ff=args.d_ff,
        is_testcase=args.is_testcase,
        rnd_seed=args.rnd_seed,
        num_layers=args.num_layers,
        masktype=args.masktype,
        windowsize=args.windowsize,
    )
