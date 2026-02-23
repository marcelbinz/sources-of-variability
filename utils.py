import requests
from io import StringIO
import numpy as np
import pandas as pd
import torch
from functools import reduce
from itertools import chain

from datasets import load_dataset


def load_raw_data(filter):
    """
    Downloads raw CSV files from OSF using predefined file IDs.

    Args:
        filter (list): List indices to indicate which data should be loaded. i.e., 0=first month, 1=second month, ...

    Returns:
        list of requests.Response: A list containing the HTTP response objects
        for each file retrieved from OSF.
    """
    l_file_ids = ["mv823/", "28ygn/", "86nar/", "byunk/"][filter]
    file_url_base = "https://osf.io/download/"

    l_responses = []
    for file_id in l_file_ids:
        file_url = file_url_base + file_id
        response = requests.get(file_url)
        l_responses.append(response)
    return l_responses


def to_dataframe(response, month):
    """
    Converts an HTTP response containing CSV text into a pandas DataFrame
    and annotates it with the month of data collection.

    Args:
        response (requests.Response): HTTP response with CSV data.
        month (str): Month label to assign to the DataFrame.

    Returns:
        pandas.DataFrame: Parsed data with an additional 'month' column.
    """
    csv_content = StringIO(response.text)
    df = pd.read_csv(csv_content)
    df["month"] = month
    return df


def concat_raw_data(l_responses, filter):
    """
    Transforms a list of HTTP responses into DataFrames with annotated months
    and concatenates them into a single DataFrame.

    Args:
        l_responses (list): List of requests.Response objects containing CSV text.
        filter (list): List indices to indicate which data should be loaded. i.e., 0=first month, 1=second month, ...

    Returns:
        pandas.DataFrame: Combined DataFrame containing all raw data across months.
    """
    l_months = ["March", "April", "June", "November"][filter]
    l_dfs_itc = list(map(to_dataframe, l_responses, l_months))
    df_itc_all = reduce(lambda x, y: pd.concat(
        [x, y], ignore_index=True), l_dfs_itc)
    return df_itc_all


def load_and_concat_raw_data(filter):
    """
    Orchestrates the full data ingestion pipeline:
    downloads raw files and combines them into a single DataFrame.

    Args:
        filter (list): List indices to indicate which data should be loaded. i.e., 0=first month, 1=second month, ...

    Returns:
        pandas.DataFrame: Aggregated dataset composed of all files retrieved and parsed.
    """
    l_raw = load_raw_data(filter)
    df_itc_all = concat_raw_data(l_raw, filter)
    return df_itc_all


def unique_participant_ids_itc(df):
    """
    Generates unique participant identifiers based on the 'sid' column across specific months.

    The function:
    - Aggregates 'sid' by month
    - Sorts months chronologically using a custom order
    - Computes cumulative and shifted sid values
    - Adds a unique sid identifier by offsetting and mapping to a sequential integer

    Parameters:
        df (pd.DataFrame): Input DataFrame containing at least 'sid' and 'month' columns.

    Returns:
        pd.DataFrame: Modified DataFrame with a new 'sid_unique' column of sequential participant IDs.
    """

    month_order = ["March", "April", "June", "November"]

    # Convert the 'month' column to a Categorical type with that specific order
    df_ids = pd.DataFrame(df.groupby("month")["sid"].max()).reset_index()
    df_ids["month"] = pd.Categorical(
        df_ids["month"], categories=month_order, ordered=True)
    df_ids.sort_values("month", inplace=True)
    df_ids["sid_prev"] = df_ids["sid"].cumsum()
    df_ids["sid_continue"] = df_ids["sid_prev"].shift(1).fillna(0)
    df = pd.merge(df, df_ids.drop(
        columns=["sid", "sid_prev"]), how="left", on="month")
    df["sid_unique"] = (df["sid"] + df["sid_continue"]).astype(int)

    # Map each value to a unique integer starting from 1
    unique_vals = df["sid_unique"].unique()
    mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}

    # Convert the series using the mapping
    df["sid_unique"] = df["sid_unique"].map(
        mapping).astype("category").astype(int)

    return df


def make_conditions(df):
    """
    Creates multiple versions of a dataset based on different trial and participant reshuffling conditions.

    It produces:
    - The original dataset with trial indexing
    - A version where trials are shuffled within each participant (no history)
    - A version where participants are shuffled within each trial (shared history)
    - A fully randomized version where both trials and participants are reshuffled (shared no history)

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'sid_unique' column.
        n_trials (int): Number of trials per participant.
        n_participants_total (int): Total number of unique participants.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple of four datasets in the above order.
    """

    # original
    df["trial_id"] = df.groupby("sid_unique").cumcount() + 1
    df_itc_original = df.copy()
    # id no hist
    df_itc_id_nohist = id_nohist(df)
    df_itc_shared_hist = shared_hist(df)
    df_itc_shared_nohist = shared_nohist(df)
    return df_itc_original, df_itc_id_nohist, df_itc_shared_hist, df_itc_shared_nohist


def id_nohist(df):
    """
    Randomizes trial order within each participant group, preserving participant identity.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'sid_unique' and 'trial_id'.

    Returns:
        pd.DataFrame: Shuffled DataFrame with reassigned 'trial_id' per participant.
    """

    df_new = df.copy()
    df_new["trial_id_old"] = df_new["trial_id"].copy()
    df_new = df_new.groupby("sid_unique").sample(frac=1).reset_index(drop=True)
    df_new["trial_id"] = df_new.groupby("sid_unique").cumcount() + 1
    return df_new


def shared_hist(df):
    """
    Randomizes participant identity within each trial, maintaining shared trial history across participants.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'sid_unique' and 'trial_id'.

    Returns:
        pd.DataFrame: Shuffled DataFrame with reassigned 'sid_unique' per trial.
    """

    df_new = df.copy()
    df_new["sid_unique_old"] = df_new["sid_unique"].copy(
    )
    df_new = df_new.groupby(
        "trial_id").sample(frac=1).reset_index(drop=True)
    df_new["sid_unique"] = df_new.groupby(
        "trial_id").cumcount() + 1
    df_new.sort_values(
        ["sid_unique", "trial_id"], inplace=True)
    return df_new


def shared_nohist(df):
    """
    Fully randomizes both trial and participant IDs, creating a new structure where all combinations are reshuffled.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'trial_id' and 'sid_unique'.

    Returns:
        pd.DataFrame: Completely randomized DataFrame with reassigned 'trial_id' and 'sid_unique'.
    """

    df_new = df.copy()
    df_new["trial_id_old"] = df_new["trial_id"].copy()
    df_new["sid_unique_old"] = df_new["sid_unique"].copy()
    df_new = df_new.sample(frac=1).reset_index(drop=True)
    df_new["sid_unique"] = df_new["sid_unique"].sample(frac=1).reset_index(drop=True)
    df_new["trial_id"] = df_new.groupby("sid_unique").cumcount() + 1
    df_new.sort_values(["sid_unique", "trial_id"], inplace=True)
    return df_new


def format_to_torch(df, col_pid, cols_x, col_y, col_y_shifted):
    """
    Converts a pandas DataFrame into 3D NumPy arrays suitable for PyTorch input.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing all relevant data.
    col_pid : str
        Column name used to identify unique groups (e.g., patient or session IDs).
    cols_x : list of str
        List of feature column names to be used as input (X).
    col_y : str
        Name of target column to be used as output (y).
    col_y_shifted : list of str
        Name of shifted target column to be appended to X.

    Returns:
    -------
    X_3d : numpy.ndarray
        A 3D array of shape (num_participants, num_trials, num_features) representing input features.
    y_3d : numpy.ndarray
        A 3D array of shape (num_participants, num_trials, num_features) representing target labels.

    Notes:
    -----
    - The function groups the DataFrame by `col_pid`, then converts each group into a 2D NumPy array.
    - It stacks all groups into a single 3D array for both inputs and targets.
    - Useful for preparing time-series or sequential data for deep learning models in PyTorch.
    """
    df_x = df[col_pid + cols_x + col_y_shifted].copy()
    df_y = df[col_pid + col_y].copy()
    grouped_x = df_x.groupby(
        col_pid)[cols_x + col_y_shifted].apply(lambda x: x.to_numpy())
    grouped_y = df_y.groupby(col_pid)[col_y].apply(lambda x: x.to_numpy())
    seqs_x = [torch.tensor(arr) for arr in grouped_x]
    seqs_y = [torch.tensor(arr) for arr in grouped_y]
    X_3d = torch.nn.utils.rnn.pad_sequence(seqs_x, batch_first=True)
    y_3d = torch.nn.utils.rnn.pad_sequence(seqs_y, batch_first=True)
    lengths = torch.tensor([len(s) for s in seqs_x])
    mask = torch.arange(X_3d.size(1))[None, :] < lengths[:, None]
    return X_3d, y_3d, mask


def train_dev_split(df, splittype, n_trial_split=None):
    """
    Splits a DataFrame into training and development sets based on a specified strategy.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing a 'trial_id' column used for splitting.
    splittype : str
        The strategy used for splitting. Options:
        - "first_vs_second_half": uses `n_trial_split` to divide trials.
        - "every_second": assigns every second trial to the training set.
    n_trial_split : int, optional
        The trial ID threshold used when `splittype` is "first_vs_second_half".

    Returns:
    -------
    df_train : pandas.DataFrame
        Subset of the original DataFrame marked as training data.
    df_dev : pandas.DataFrame
        Subset of the original DataFrame marked as development data.
    """
    match splittype:
        case "first_vs_second_half":
            df["is_train"] = df["trial_id"] <= n_trial_split
        case "every_second":
            df["is_train"] = df["trial_id"] % 2 == 1
    dict_groups = dict(tuple(df.groupby('is_train')))
    df_train = dict_groups[True]
    df_dev = dict_groups[False]
    return df_train, df_dev


def split_and_format(my_df, condition, splittype, n_trial_split, col_pid, cols_x, col_y, col_y_shifted):
    """
    Splits a DataFrame into training and development sets, formats them for PyTorch input,
    and returns a dictionary containing the split and formatted data along with metadata.

    Parameters:
    ----------
    my_df : pd.DataFrame
        The input DataFrame containing the full dataset.
    condition : str
        A label or identifier for the condition or experimental setting.
    splittype : str
        The type of splitting strategy to use (e.g., 'random', 'sequential').
    n_trial_split : int
        Number of trials or samples to include in the training split.
    col_pid : str
        Column name representing participant or subject ID.
    cols_x : list of str
        List of column names to be used as input features.
    col_y : str
        Column name representing the target variable.
    col_y_shifted : str
        Column name for the shifted target variable (e.g., for sequence prediction).

    Returns:
    -------
    dict
        A dictionary containing:
            - "condition": str, the input condition label
            - "df_train": pd.DataFrame, training subset of the original DataFrame
            - "df_dev": pd.DataFrame, development subset of the original DataFrame
            - "X_train": torch.Tensor, formatted training input features
            - "y_train": torch.Tensor, formatted training target values
            - "X_dev": torch.Tensor, formatted development input features
            - "y_dev": torch.Tensor, formatted development target values
    """
    my_df_train, my_df_dev = train_dev_split(
        my_df, splittype, n_trial_split=n_trial_split
    )
    X_train, y_train, mask_train = format_to_torch(
        my_df_train, col_pid, cols_x, col_y, col_y_shifted=col_y_shifted
    )
    X_dev, y_dev, mask_dev = format_to_torch(
        my_df_dev, col_pid, cols_x, col_y, col_y_shifted=col_y_shifted
    )
    dict_out = {
        "condition": condition,
        "df_train": my_df_train,
        "df_dev": my_df_dev,
        "X_train": X_train,
        "y_train": y_train,
        "X_dev": X_dev,
        "y_dev": y_dev,
        "mask_train": mask_train,
        "mask_dev": mask_dev,
    }
    return dict_out


def shift_y(df):
    """
    Adds a lagged version of the 'right_picked' column to the DataFrame.

    The lag is computed within each 'sid' group, shifting values by one trial.
    Missing values (e.g., first trial in each group) are filled with 0.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing 'sid' and 'right_picked' columns.

    Returns:
    -------
    df : pandas.DataFrame
        The modified DataFrame with a new column 'right_picked_prev'.
    """
    df["right_picked_prev"] = df.groupby(
        "sid")["right_picked"].shift(1)
    df.fillna(value={"right_picked_prev": 0}, inplace=True)
    df["right_picked_prev"] = df["right_picked_prev"].astype(int)
    return df

def swap_two_cols(df, colnames, rs, prop_swap=0.5):
    """
    Swap values between two columns for a random subset of rows within each participant group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the data.
    colnames : list or tuple of str
        Two column names whose values should be swapped.
    rs : int
        Random seed for reproducibility of the swapping process.
    prop_swap : float, optional (default=0.5)
        Proportion of rows (per participant group) in which the values 
        of the two columns will be swapped.

    Returns
    -------
    pandas.DataFrame
        The dataframe with swapped values in the specified columns.
    """
    idxs_swap = (
        df.groupby("sid_unique")
          .sample(frac=prop_swap, random_state=rs) # , random_state=1
          .index
    )

    to_be_swapped0 = df.loc[idxs_swap, colnames[0]]
    to_be_swapped1 = df.loc[idxs_swap, colnames[1]]

    df.loc[idxs_swap, colnames[0]] = to_be_swapped1
    df.loc[idxs_swap, colnames[1]] = to_be_swapped0

    return df


def shuffle_single_column(df, colname):
    """
    Replace values in a column with random draws from its unique values,
    applied independently within each participant group.

    This produces a per-group shuffle where each row receives a random
    value drawn uniformly from the set of unique values in the column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the data.
    colname : str
        Name of the column to shuffle.

    Returns
    -------
    pandas.DataFrame
        The dataframe with the specified column shuffled within groups.
    """
    unique_vals = df[colname].unique()
    rng = np.random.default_rng(seed=1)

    def sample_group(g):
        return rng.choice(unique_vals, size=len(g))

    df[colname] = df.groupby("sid_unique")[colname].transform(sample_group)
    return df


def scale_fixed(x, mn, sd):
    """ compute a scaled version of a variable by using predefined mean and sd """
    return (x - mn) / sd

def zscore_grouped_cols(df, l_colnames_grouped_zscale, tf="z"):
    """
    compute a z score across several columns by using the mean and sd of the combined values of all columns
    
    :param df: dataframe containing the columns to be z-scored
    :param l_colnames_grouped_zscale: pairs of column names for which to compute z scores across the combined values of the two columns
    :param mean_center_only: mean-center only, i.e., do not divide by sd, default False
    :return: dataframe with z-scaled columns
    """
    scale_colnames_flat = list(chain.from_iterable(l_colnames_grouped_zscale))

    # apply log transformation to value columns if specified
    # for risky data set, add absolute value of the minimum value plus a small constant to avoid log(0) and log of negative values
    if tf == "log_values":
        min_val = df[[c for c in df.columns if c.endswith("_val")]].melt()["value"].min()
        add_val = np.abs(min_val) + 0.000001
        for cn in df.columns:
            if cn.endswith("_val"):
                df[cn] = np.log(df[cn] + add_val)  # add abs(min val) and small constant to avoid log(0)

    stats = [df[vs].reset_index().melt(id_vars="index").value.agg(["mean", "std"]) for vs in l_colnames_grouped_zscale]
    d_stats = dict()
    for idx, cn in enumerate(scale_colnames_flat):
        d_stats[cn] = stats[int(np.floor(idx / len(l_colnames_grouped_zscale[0])))]
    for k, v in d_stats.items():
        if tf == "mean_center_only":
            v["std"] = 1
        df[k] = df[k].apply(scale_fixed, mn=v["mean"], sd=v["std"])

    return df


def load_sov_dataset(dataset_name):
    """Load datasets based on the specified dataset name."""
    match dataset_name:
        case "risky":
            df, dict_info = load_risky_dataset()
        case "itc":
            df, dict_info = load_itc_dataset()
    return df, dict_info



def load_itc_dataset(load_from_osf=False):
    """
    Load and preprocess the ITC dataset from Agrawal et al. (2023) JEP:G containing four different months of 2020.

    :param load_from_osf: If True, loads raw data from OSF, otherwise, loads data from disk.
    
    """

    # load all four dataframes and concat into one df
    # each data frame from a different 2020 month (i.e., march, april, june, november)
    if load_from_osf:
        # select which months to load
        # slice(0,4,1) selects all months
        filter = slice(0, 4, 1)
        df_itc_all = load_and_concat_raw_data(filter)
        df_itc_all.drop("Unnamed: 0", axis=1, inplace=True)
        df_itc_all.to_csv("data/full-data.csv")
    else:
        df_itc_all = pd.read_csv("data/full-data.csv")
    # make participant ids unique (i.e., they were re-used across sessions, but likely not from the same participant)
    df_itc_all = unique_participant_ids_itc(df_itc_all)

    col_pid = ["sid_unique"]
    cols_x = ["right_time", "right_val", "left_time", "left_val"]
    col_y = ["right_picked"]
    col_y_shifted = ["right_picked_prev"]

    colnames_zscale = [["right_time", "left_time"], ["right_val", "left_val"]]

    n_participants_total = df_itc_all["sid_unique"].nunique()
    n_trials_train = 130

    in_dim = len(cols_x) + len(col_y_shifted)

    dict_out = {
        "n_participants_total":n_participants_total,
        "n_trials_train":n_trials_train,
        "col_pid":col_pid,
        "cols_x":cols_x,
        "col_y":col_y,
        "col_y_shifted":col_y_shifted,
        "l_colnames_grouped_zscale":colnames_zscale,
        "in_dim":in_dim
    }

    return df_itc_all, dict_out


def load_risky_dataset():
    """Load and preprocess the risky dataset from Peterson et al. (2021) Science."""

    # minimal number of trials available per participant to be included in the data set
    n_minimal = 80
    n_trials_train = 60
    
    ds = load_dataset("marcelbinz/peterson2021using", "exp1")
    df_risky = ds['train'].to_pandas()

    df_risky, df_counts, df_participants_use = participants_with_enough_trials(df_risky, n_minimal)
    df_risky = participant_id_counter(df_risky, df_counts, df_participants_use, n_minimal)
    df_risky, dict_colnames = rename_risky_cols(df_risky)

    n_participants_total = df_risky["sid"].unique().shape[0]

    in_dim = len(dict_colnames["cols_x"]) + len(dict_colnames["col_y_shifted"])

    dict_out = {
        "n_participants_total":n_participants_total,
        "n_trials_train":n_trials_train,
        "col_pid":dict_colnames["col_pid"],
        "cols_x":dict_colnames["cols_x"],
        "col_y":dict_colnames["col_y"],
        "col_y_shifted":dict_colnames["col_y_shifted"],
        "l_colnames_grouped_zscale":dict_colnames["l_colnames_grouped_zscale"],
        "in_dim":in_dim
    }

    return df_risky, dict_out


def participants_with_enough_trials(df_risky, n_minimal):
    df_counts_trials = df_risky.groupby("participant")["trial"].count().reset_index()
    df_counts = df_counts_trials.groupby("trial").count().sort_values("trial", ascending=False)
    df_counts["n_cum"] = df_counts["participant"].cumsum()

    df_participants_use = df_counts_trials.query(f"trial >= {n_minimal}").copy()
    df_risky = pd.merge(df_participants_use.loc[:,"participant"], df_risky, on="participant", how="inner")

    return df_risky, df_counts, df_participants_use

    
def participant_id_counter(df_risky, df_counts, df_participants_use, n_minimal):
    df_participants_use["sid"] = np.arange(1, 1 + df_counts.loc[n_minimal,"n_cum"], 1)
    df_risky = pd.merge(df_risky, df_participants_use[["participant", "sid"]], on="participant").drop(columns=["participant"])
    df_risky = df_risky[["sid"] + [c for c in df_risky.columns if c != "sid"]]

    # in the risky dataset, sid and sid_unique are the same, but we create a sid_unique column for consistency with the ITC dataset and to allow for potential future modifications where they might differ
    df_risky["sid_unique"] = df_risky["sid"].copy()

    return df_risky

def rename_risky_cols(df_risky):
    # original column names in the dataset, and new column names that are more descriptive and consistent with the ITC dataset
    colnames_old = [
        "probA1", "probA2", "probA3", "probA4", "probA5", "probA6", "probA7", "probA8", "probA9",
        "probB1", "probB2", "probB3", "probB4", "probB5", "probB6", "probB7", "probB8", "probB9",
        "outA1", "outA2", "outA3", "outA4", "outA5", "outA6", "outA7", "outA8", "outA9",
        "outB1", "outB2", "outB3", "outB4", "outB5", "outB6", "outB7", "outB8", "outB9",
        "trial", "choice"
    ]
    colnames_new = [
        "left_1_prob", "left_2_prob", "left_3_prob", "left_4_prob", "left_5_prob", 
        "left_6_prob", "left_7_prob", "left_8_prob", "left_9_prob",
        "right_1_prob", "right_2_prob", "right_3_prob", "right_4_prob", "right_5_prob", 
        "right_6_prob", "right_7_prob", "right_8_prob", "right_9_prob",
        "left_1_val", "left_2_val", "left_3_val", "left_4_val", "left_5_val", 
        "left_6_val", "left_7_val", "left_8_val", "left_9_val",
        "right_1_val", "right_2_val", "right_3_val", "right_4_val", "right_5_val", 
        "right_6_val", "right_7_val", "right_8_val", "right_9_val",
        "trial_id", "right_picked"
    ]
    colnames_zscale = [[c for c in colnames_new if "val" in c]]
    cols_x = [c for c in colnames_new if "prob" in c or "val" in c]
    col_y = ["right_picked"]
    col_y_shifted = ["right_picked_prev"]
    col_pid = ["sid"]

    dict_colnames = {
        "l_colnames_grouped_zscale": colnames_zscale,
        "cols_x": cols_x,
        "col_y": col_y,
        "col_y_shifted": col_y_shifted,
        "col_pid": col_pid
    }


    dict_map_cols = {old:colnames_new[idx] for idx, old in enumerate(colnames_old)}
    df_risky.rename(columns=dict_map_cols, inplace=True)

    return df_risky, dict_colnames