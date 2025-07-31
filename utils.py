import requests
from io import StringIO
import numpy as np
import pandas as pd
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
