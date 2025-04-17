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
        dataset['target_price'] = (dataset['b1'] + dataset['a1']) / 2

    elif target_type == 'volume_weighted':
        total_value = 0
        total_volume = 0
        # Iterate through levels 1 to num_levels.
        for level in range(1, num_levels + 1):
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
        # Iterate through levels 1 to num_levels.
        for level in range(1, num_levels + 1):
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

