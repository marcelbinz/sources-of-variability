import requests
from io import StringIO
import pandas as pd
from functools import reduce


def load_raw_data():
    """
    Downloads raw CSV files from OSF using predefined file IDs.

    Returns:
        list of requests.Response: A list containing the HTTP response objects
        for each file retrieved from OSF.
    """
    l_file_ids = ["mv823/", "28ygn/", "86nar/", "byunk/"]
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


def concat_raw_data(l_responses):
    """
    Transforms a list of HTTP responses into DataFrames with annotated months
    and concatenates them into a single DataFrame.

    Args:
        l_responses (list): List of requests.Response objects containing CSV text.

    Returns:
        pandas.DataFrame: Combined DataFrame containing all raw data across months.
    """
    l_months = ["March", "April", "June", "November"]
    l_dfs_itc = list(map(to_dataframe, l_responses, l_months))
    df_itc_all = reduce(lambda x, y: pd.concat(
        [x, y], ignore_index=True), l_dfs_itc)
    return df_itc_all


def load_and_concat_raw_data():
    """
    Orchestrates the full data ingestion pipeline:
    downloads raw files and combines them into a single DataFrame.

    Returns:
        pandas.DataFrame: Aggregated dataset composed of all files retrieved and parsed.
    """
    l_raw = load_raw_data()
    df_itc_all = concat_raw_data(l_raw)
    return df_itc_all
