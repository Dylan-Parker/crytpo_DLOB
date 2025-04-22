import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def reduce_mem_usage(df, precision=np.float32):
    """
    Iterate through all the columns of a dataframe and downcast numerical data types
    to lower precision versions. Floats are converted to np.float16 and integers are
    downcast if possible.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe whose memory usage we want to reduce.

    Returns
    -------
    df : pd.DataFrame
        The dataframe with optimized memory usage.
    """
    # Downcast floats to float16 and integers to the smallest possible integer type.
    for col in df.columns:
        col_type = df[col].dtype

        # If the column is of float type, cast to float16 if possible.
        if pd.api.types.is_float_dtype(col_type):
            df[col] = df[col].astype(precision)
        # If the column is an integer type, try to downcast to integer.
        elif pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        # Optionally, if you have object-type columns that have a low cardinality,
        # you might convert them to 'category' types.
        elif pd.api.types.is_object_dtype(col_type):
            num_unique_values = df[col].nunique()
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
    return df

def load_deep_orderbook_dataset(
        num_files: int = None,
        precision: np.dtype = np.float64,
        path : str = 'data/coinbase_btc_usd/coinbase/btc_usd/l2_snapshots/100ms/'
) -> pd.DataFrame:
    """
    :param num_files: number of files you want to load, default behavior loads all files
    :param precision: np.dtype for floating point precision, defaults to np.float32
    :param path: string containing the directory with parquet files you wish to laod
    :return: dataframe with loaded dataset
    """

    dfs = []
    i = 0
    for x in os.listdir(path):
        # If needed, remove or modify this limit—here it stops after 10 files.
        if num_files is not None:
            if i > num_files:
                break

        # Build the complete filepath
        file_path = os.path.join(path, x)

        # Load the parquet file into a DataFrame.
        temp = pd.read_parquet(file_path)
        # Downcast numeric columns (floats and integers) and optionally convert objects.
        temp = reduce_mem_usage(temp, precision=precision)

        # Append the reduced DataFrame to our list.
        dfs.append(temp)

        # Clean up and increment the counter.
        del temp
        gc.collect()
        i += 1

    # Concatenate all DataFrames once the loop is complete.
    l2_snapshot = pd.concat(dfs, ignore_index=True).dropna()

    print('Memory Usage: {:.2f} MB'.format(l2_snapshot.memory_usage().sum() / (1024 ** 2)))
    return l2_snapshot

def construct_target_prices(dataset : pd.DataFrame, target_type : str, num_levels=1)-> pd.DataFrame:
    """
    Constructs target price(s) from an orderbook DataFrame.

    The DataFrame is assumed to have columns with the following naming convention:
      - 'b1', 'b2', ...: bid price levels,
      - 'a1', 'a2', ...: ask price levels,
      - 'bq1', 'bq2', ...: bid volumes,
      - 'aq1', 'aq2', ...: ask volumes.

    Parameters
    ----------
    dataset : pd.DataFrame
        A DataFrame containing the orderbook snapshots.
    target_type : str
        The type of target to construct. Supported choices are:
            - 'simple_midpoint': (b1 + a1) / 2.
            - 'volume_weighted': Volume-weighted average price over num_levels levels,
              computed as: (Σ[b_i*bq_i + a_i*aq_i]) / (Σ[bq_i + aq_i]).
            - 'micro_price': Micro-price over num_levels levels, computed as:
              (Σ[a_i*bq_i + b_i*aq_i]) / (Σ[bq_i + aq_i]).
    num_levels : int, optional (default=1)
        The number of order book levels to include in the calculation.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with an additional column 'target_price' containing the calculated target.

    Raises
    ------
    ValueError
        If target_type is not one of the supported types.
    """


    if target_type == 'simple_midpoint':
        # Use only level 1 for a simple mid price.
        dataset['target_price'] = (dataset['b0'] + dataset['a0']) / 2

    elif target_type == 'volume_weighted':
        total_value = 0
        total_volume = 0
        # Iterate through levels 0 to num_levels.
        for level in range(num_levels):
            bid_price = dataset[f'b{level}']
            ask_price = dataset[f'a{level}']
            bid_volume = dataset[f'bq{level}']
            ask_volume = dataset[f'aq{level}']

            total_value += bid_price * bid_volume + ask_price * ask_volume
            total_volume += bid_volume + ask_volume

        # Avoid division by zero if total_volume happens to be 0.
        dataset['target_price'] = total_value / total_volume

    elif target_type == 'micro_price':
        numerator = 0
        denominator = 0
        # Iterate through levels 0 to num_levels.
        for level in range(num_levels):
            bid_price = dataset[f'b{level}']
            ask_price = dataset[f'a{level}']
            bid_volume = dataset[f'bq{level}']
            ask_volume = dataset[f'aq{level}']

            # Compute numerator = a_i * bq_i + b_i * aq_i for this level.
            numerator += ask_price * bid_volume + bid_price * ask_volume
            # Denominator = bq_i + aq_i.
            denominator += bid_volume + ask_volume

        dataset['target_price'] = numerator / denominator

    else:
        raise ValueError(f"target_type '{target_type}' not recognized. Please choose among 'simple_midpoint', 'volume_weighted', or 'micro_price'.")

    return dataset.loc[:,["target_price"]].copy()


def construct_labels_slow(
        targets: pd.DataFrame,
        theta: float,
        horizon: int = 20,
) -> pd.DataFrame :
    """
    Constructs 0/1/2 labels from the percentage change between
    the average of [t-horizon ... t] and [t+1 ... t+horizon].

    0 : Δ < -θ
    1 : -θ ≤ Δ ≤ θ
    2 : Δ > θ

    If theta is None, choose θ = 33rd percentile of |Δ| for balance.

    Returns
    -------
    labels : pd.DataFrame
        Single‐column DataFrame indexed like `targets`, dtype "Int64" (nullable).
    theta  : float
        The threshold actually used.
    """
    # Prepare output
    labels = pd.DataFrame(index=targets.index, columns=["label"], dtype="Int64")

    # Extract the series of values (assumes your price column is the first one)
    vals = targets.iloc[:, 0].to_numpy(dtype=float)
    N = len(vals)

    # Compute returns array (NaN where we don't have full windows)
    delta = np.full(N, np.nan, dtype=float)
    for t in range(N):
        if t >= horizon and t + horizon < N:
            past = vals[t - horizon: t + 1]  # includes t
            fut = vals[t + 1: t + 1 + horizon]  # next horizon points
            delta[t] = (fut.mean() - past.mean()) / past.mean()


    # Assign labels
    lab = np.full(N, pd.NA, dtype="Int64")
    lab[delta < -theta] = 0
    lab[(delta >= -theta) & (delta <= theta)] = 1
    lab[delta > theta] = 2

    labels["label"] = lab
    return labels


def construct_labels(
        targets: pd.DataFrame,
        horizon: int,
        theta: float
) -> pd.DataFrame:
    """
    Vectorized label construction using sliding_window_view.

    Parameters
    ----------
    targets : pd.DataFrame
        DataFrame with a single column of prices (e.g. mid‐price), indexed by time.
    horizon : int
        Look‐ahead horizon h.
    theta : float
        Threshold for classifying returns.

    Returns
    -------
    labels_df : pd.DataFrame
        Single‐column DataFrame "label" with Int64 dtype, containing 0/1/2 or <NA> at the edges.
    """
    vals = targets.iloc[:, 0].to_numpy(dtype=float)
    N = len(vals)

    # past windows covering [t-horizon ... t], length=horizon+1
    Wp = np.lib.stride_tricks.sliding_window_view(vals, window_shape=horizon + 1)
    past_means = Wp.mean(axis=1)  # shape: (N - horizon,)

    # future windows covering [t+1 ... t+horizon], length=horizon
    Wf = np.lib.stride_tricks.sliding_window_view(vals, window_shape=horizon)
    future_means = Wf.mean(axis=1)  # shape: (N - horizon + 1,)

    # allocate delta and fill for valid t = horizon ... N-horizon-1
    delta = np.full(N, np.nan, dtype=float)
    idx = np.arange(horizon, N - horizon)
    delta[idx] = (future_means[idx + 1] - past_means[idx - horizon]) / past_means[idx - horizon]

    # build nullable Int64 label series
    labels = pd.Series(pd.NA, index=targets.index, dtype="Int64")
    labels[delta < -theta] = 0
    labels[(delta >= -theta) & (delta <= theta)] = 1
    labels[delta > theta] = 2

    return labels.to_frame(name="label")


def normalize_dataset(
        dataset : pd.DataFrame,
        mean_q : float =None,
        mean_p : float =None,
        std_q : float =None,
        std_p : float =None
) -> (pd.DataFrame, dict):
    """
    Assumes that data is formatted wtih columns  like [b0, bq0, b1, bq1, ... ] e.g. alternating between price and quantity
    returns normalized dataset and dictionary of statistics.
    """
    if (mean_q is None) or (std_q is None):
        mean_q = dataset.iloc[:, 1::2].stack().mean()
        std_q = dataset.iloc[:, 1::2].stack().std()

    if (mean_p is None) or (std_p is None):
        mean_p = dataset.iloc[:, 0::2].stack().mean()
        std_p = dataset.iloc[:, 0::2].stack().std()

    price_cols = dataset.columns[0::2]
    size_cols = dataset.columns[1::2]

    for col in size_cols:
        dataset[col] = dataset[col].astype("float64")
        dataset[col] = (dataset[col] - mean_q) / std_q

    for col in price_cols:
        dataset[col] = dataset[col].astype("float64")
        dataset[col] = (dataset[col] - mean_p) / std_p

    if dataset.isnull().values.any():
        raise ValueError("data contains null value")

    statistics = {
        "mean_q" : mean_q,
        "mean_p" : mean_p,
        "std_q" : std_q,
        "std_p" : std_p
    }
    return dataset, statistics