import requests
from io import StringIO
import numpy as np
import pandas as pd
import torch
from functools import reduce


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


def unique_participant_ids(df):
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


def make_conditions(df, n_trials, n_participants_total):
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
    df_itc_shared_nohist = shared_nohist(df, n_trials, n_participants_total)
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


def shared_nohist(df, n_trials, n_participants_total):
    """
    Fully randomizes both trial and participant IDs, creating a new structure where all combinations are reshuffled.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'trial_id' and 'sid_unique'.
        n_trials (int): Number of trials per participant.
        n_participants_total (int): Total number of unique participants.

    Returns:
        pd.DataFrame: Completely randomized DataFrame with reassigned 'trial_id' and 'sid_unique'.
    """

    df_new = df.copy()
    df_new["trial_id_old"] = df_new["trial_id"].copy()
    df_new["sid_unique_old"] = df_new["sid_unique"].copy()
    df_new = df_new.sample(frac=1).reset_index(drop=True)
    df_new["trial_id"] = np.tile(range(1, n_trials+1), n_participants_total)
    df_new["sid_unique"] = np.tile(
        range(1, n_participants_total + 1), n_trials)
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
    X_3d = torch.from_numpy(np.stack(grouped_x.to_numpy()))
    y_3d = torch.from_numpy(np.stack(grouped_y.to_numpy()))
    return X_3d, y_3d


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
    X_train, y_train = format_to_torch(
        my_df_train, col_pid, cols_x, col_y, col_y_shifted=col_y_shifted
    )
    X_dev, y_dev = format_to_torch(
        my_df_dev, col_pid, cols_x, col_y, col_y_shifted=col_y_shifted
    )
    df_out = {
        "condition": condition,
        "df_train": my_df_train,
        "df_dev": my_df_dev,
        "X_train": X_train,
        "y_train": y_train,
        "X_dev": X_dev,
        "y_dev": y_dev
    }
    return df_out


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

def swap_two_cols(df, colnames, prop_swap=0.5):
    """
    Swap values between two columns for a random subset of rows within each participant group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the data.
    colnames : list or tuple of str
        Two column names whose values should be swapped.
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
          .sample(frac=prop_swap, random_state=1)
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

