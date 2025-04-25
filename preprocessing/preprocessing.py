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
        base_path : str = 'data/coinbase_btc_usd/coinbase/btc_usd/l2_snapshots/100ms/'
) -> pd.DataFrame:
    """
    :param num_files: number of files you want to load, default behavior loads all files
    :param precision: np.dtype for floating point precision, defaults to np.float32
    :param path: string containing the directory with parquet files you wish to laod
    :return: dataframe with loaded dataset
    """

    dfs = []
    i = 0
    for x in os.listdir(base_path):
        # If needed, remove or modify this limit—here it stops after 10 files.
        if num_files is not None:
            if i > num_files:
                break

        # Build the complete filepath
        file_path = os.path.join(base_path, x)

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
        # Iterate through levels 1 to num_levels.
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
        # Iterate through levels 1 to num_levels.
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


def clean_data(df, strategy='dropna', verbose=True):
    """
    Cleans the dataframe by handling missing values.

    Args:
        df (pd.DataFrame): Input dataframe.
        strategy (str): Method to handle NaNs ('dropna', 'ffill', 'bfill', etc.).
        verbose (bool): Print shapes before and after.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    if verbose:
        print(f"Shape before cleaning ({strategy}): {df.shape}")
        print(f"NaN count before: {df.isna().sum().sum()}")

    if strategy == 'dropna':
        df_cleaned = df.dropna()
    elif strategy == 'ffill':
        df_cleaned = df.ffill()
    elif strategy == 'bfill':
        df_cleaned = df.bfill()
    # Add other strategies like interpolation if needed
    else:
        raise ValueError(f"Unknown cleaning strategy: {strategy}")

    if verbose:
        print(f"Shape after cleaning: {df_cleaned.shape}")
        print(f"NaN count after: {df_cleaned.isna().sum().sum()}")

    return df_cleaned

def calculate_basic_features(df, num_levels=10):
    """
    Calculates basic LOB features like mid-price, spread, volume imbalance.

    Args:
        df (pd.DataFrame): Input dataframe (must contain b1, a1, bq1, aq1 etc.).
        num_levels (int): Number of LOB levels to consider for imbalance calculation.

    Returns:
        pd.DataFrame: Dataframe with added feature columns.
    """
    df_feat = df.copy()
    bid_price_cols = [f'b{i}' for i in range(num_levels)]
    ask_price_cols = [f'a{i}' for i in range(num_levels)]

    # Volume Imbalance (VIP) - based on specified number of levels
    bid_vol_cols = [f'bq{i}' for i in range(num_levels)]
    ask_vol_cols = [f'aq{i}' for i in range(num_levels)]
    df_feat_cols = bid_price_cols + ask_price_cols + bid_vol_cols + ask_vol_cols
    df_feat = df_feat.loc[:, df_feat_cols]
    # Mid-price
    df_feat['mid_price'] = (df_feat['a0'] + df_feat['b0']) / 2.0

    # Spread
    df_feat['spread'] = df_feat['a0'] - df_feat['b0']

    total_bid_vol = df_feat[bid_vol_cols].sum(axis=1)
    total_ask_vol = df_feat[ask_vol_cols].sum(axis=1)

    # Avoid division by zero
    total_vol = total_bid_vol + total_ask_vol
    df_feat[f'vol_imbalance_{num_levels}'] = (total_bid_vol - total_ask_vol) / total_vol.replace(0, np.nan) # Handle cases with zero total volume

    # Add other features if needed (e.g., weighted mid-price, price differences)

    return df_feat

def normalize_features(df_train, df_val, df_test, feature_cols):
    """
    Applies StandardScaler normalization. Fits on training data only.

    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        df_test (pd.DataFrame): Test data.
        feature_cols (list): List of column names to normalize.

    Returns:
        tuple: Normalized (df_train, df_val, df_test), fitted_scaler
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # Fit only on training data
    scaler.fit(df_train[feature_cols])

    # Transform all sets
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    df_train_scaled[feature_cols] = scaler.transform(df_train[feature_cols])
    df_val_scaled[feature_cols] = scaler.transform(df_val[feature_cols])
    df_test_scaled[feature_cols] = scaler.transform(df_test[feature_cols])

    return df_train_scaled, df_val_scaled, df_test_scaled, scaler


# --- Labeling Functions ---

def create_labels(df: pd.DataFrame, price_col='mid_price', method='tlob', k=20, h=10, alpha=0.0002):
    """
    Creates classification labels for price movement prediction based on a specified price column.

    Args:
        df (pd.DataFrame): Dataframe containing the price column.
        price_col (str): The name of the column with the price to use for labeling.
        method (str): Labeling method ('tlob', 'cryptolob').
        k (int): Smoothing window size.
        h (int): Prediction horizon (only used for 'tlob').
        alpha (float | str | None): Threshold or 'auto' for dynamic.

    Returns:
        pd.Series: Labels (0: Down, 1: Stable, 2: Up).
    """
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame.")

    price_series = df[price_col] # Extract the price series
    if method == 'tlob':
        # Pass the extracted Series to _create_labels_tlob
        return _create_labels_tlob(price_series, k, h, alpha)
    elif method == 'cryptolob':
        # _create_labels_cryptolob expects df and price_col, so the call is correct
        # However, for consistency, you might refactor _create_labels_cryptolob
        # to also accept a price_series. Let's keep it as is for now based on your file.
        return _create_labels_cryptolob(df, price_col, k, alpha)
    else:
        raise ValueError(f"Unknown labeling method: {method}")

# --- Main Orchestration Function ---

def preprocess_lob_data(config):
    """
    Loads, preprocesses, adds features, and labels LOB data based on config.
    """
    # 1. Load Data
    df = load_deep_orderbook_dataset(
        base_path=config.data_path or '../data/deep_orderbook/coinbase/', # Adjusted path
        num_files=config.num_files or None,
        precision=config.precision or np.float32,
        verbose=config.verbose or True
    )
    if df is None or df.empty:
         raise ValueError("Data loading failed or returned empty DataFrame.")

    # 2. Clean Data
    df = clean_data(
        df,
        strategy=config.clean_strategy or 'dropna',
        verbose=config.verbose or True
    )
    if df.empty:
        raise ValueError("DataFrame is empty after cleaning.")


    # 3. Construct the SPECIFIC Target Price Series to use for LABELING
    label_price_type = config.label_price_type or 'simple_midpoint'
    label_price_levels = config.label_price_levels or 1
    target_price_df = construct_target_prices( # This adds 'target_price' to df temporarily
        df,
        target_type=label_price_type,
        num_levels=label_price_levels
    )
    # Make sure the target price is added to the main df or extracted correctly
    # Let's assume construct_target_prices now returns just the series/df to merge
    df = df.join(target_price_df) # Join based on index
    labeling_price_column_name = 'target_price' # The column created by construct_target_prices

    if labeling_price_column_name not in df.columns:
        raise ValueError(f"Column '{labeling_price_column_name}' not found after construct_target_prices.")


    # 4. Calculate Other Basic Features (like spread, imbalance, mid_price if different from labeling price)
    # Ensure this doesn't overwrite the price column used for labeling if it's named differently
    df = calculate_basic_features(
        df,
        num_levels=config.num_levels_features or 10
    )

    # 5. Create Labels using the SPECIFIC price column constructed in step 3
    labels = create_labels(
        df, # Pass the full df again, create_labels will extract the series
        price_col=labeling_price_column_name, # Use the specific column name
        method=config.label_method or 'tlob',
        k=config.label_k or 20,
        h=config.label_h or 10,
        alpha=config.label_alpha or 'auto' # Use 'auto' or a specific value
    )
    df = pd.concat([df, labels], axis=1) # Add the 'label' column

    # 6. Drop rows with NaN labels (generated at edges or from target price calc)
    initial_rows = len(df)
    df.dropna(subset=['label'], inplace=True)
    rows_dropped = initial_rows - len(df)

    if config.get('verbose', True):
        print(f"Shape after label creation and NaN drop: {df.shape} ({rows_dropped} rows dropped)")
        if not df.empty:
            print("Label distribution:")
            print(df['label'].value_counts(normalize=True))
        else:
             print("Warning: DataFrame is empty after dropping NaN labels.")


    # 7. Final Column Selection (Optional - keep only necessary features for model)
    # Example: keep original LOB features + calculated features + label
    # feature_cols_to_keep = [...]
    # df = df[feature_cols_to_keep + ['label']]

    # ... (Rest of the function: saving optional, returning df) ...
    # Remove the temporary labeling price column if desired
    if labeling_price_column_name in df.columns and labeling_price_column_name != 'mid_price':
         df = df.drop(columns=[labeling_price_column_name])

    return df

def _calculate_dynamic_alpha(l_values: pd.Series, method='mean_abs_pct_change') -> float:
    """Calculates alpha dynamically."""
    if method == 'mean_abs_pct_change':
        # Calculate alpha as half the mean absolute percentage change
        dynamic_alpha = np.abs(l_values.dropna()).mean() / 2.0
        # Handle potential NaN result if l_values is all NaN
        return dynamic_alpha if pd.notna(dynamic_alpha) else 0.0001 # Default fallback
    # Add 'average_spread' method here if needed later
    else:
        raise ValueError(f"Unknown dynamic alpha method: {method}")
def _create_labels_tlob(prices: pd.Series, k: int, h: int, alpha: float | str | None) -> pd.Series:
    """
    Implements TLOB labeling (Eq 5, 6, 7 from the paper).
    Supports fixed or dynamic alpha.
    """
    # ... (previous calculation of w_minus, w_plus, l) ...
    w_minus = prices.rolling(window=k + 1, min_periods=k + 1).mean()
    w_plus = prices.rolling(window=k + 1, min_periods=k + 1).mean().shift(-h)
    common_idx = w_minus.index.intersection(w_plus.index)
    w_minus_aligned = w_minus.loc[common_idx]
    w_plus_aligned = w_plus.loc[common_idx]
    l = (w_plus_aligned - w_minus_aligned) / w_minus_aligned.replace(0, np.nan)

    # --- Dynamic Alpha Calculation (Optional) ---
    current_alpha = alpha
    if isinstance(alpha, str) and alpha.lower() == 'auto':
        print("Calculating alpha dynamically...")
        # Calculate dynamic alpha based on the 'l' values computed so far
        # Need to calculate 'l' first before determining dynamic alpha
        dynamic_alpha = _calculate_dynamic_alpha(l, method='mean_abs_pct_change')
        print(f"Dynamic alpha = {dynamic_alpha:.6f}")
        current_alpha = dynamic_alpha
    elif alpha is None: # Another way to trigger auto
        print("Calculating alpha dynamically (alpha was None)...")
        dynamic_alpha = _calculate_dynamic_alpha(l, method='mean_abs_pct_change')
        print(f"Dynamic alpha = {dynamic_alpha:.6f}")
        current_alpha = dynamic_alpha
    elif not isinstance(alpha, (float, int)):
         raise ValueError(f"Invalid alpha value: {alpha}. Must be float, int, 'auto', or None.")
    # Ensure alpha is not NaN or zero if calculated dynamically from empty/constant data
    if pd.isna(current_alpha) or current_alpha <= 0:
        print(f"Warning: Invalid alpha {current_alpha}, using default 0.0001")
        current_alpha = 0.0001 # Fallback

    # --- Classification ---
    n = len(prices)
    labels = pd.Series(np.nan, index=prices.index, dtype=np.int8)
    first_valid_idx = k
    last_valid_idx = n - 1 - h

    if first_valid_idx <= last_valid_idx:
        valid_indices = prices.index[first_valid_idx : last_valid_idx + 1]
        l_aligned = l.reindex(valid_indices)

        labels.loc[valid_indices] = 1  # Stable (1)
        labels.loc[valid_indices[l_aligned > current_alpha]] = 2  # Up
        labels.loc[valid_indices[l_aligned < -current_alpha]] = 0 # Down
    else:
         print(f"Warning: Data length ({n}) too short for k={k}, h={h}. No labels generated.")


    return labels.rename('label')

def _create_labels_cryptolob(df, price_col, k, alpha):
    """Implements Crypto-LOB labeling (Eq 1, 2, 3)."""
    prices = df[price_col]
    labels = pd.Series(np.nan, index=df.index) # Initialize with NaN

    # Calculate rolling means for past and future windows efficiently
    # Past window mean (m_minus): mean of p(t-k+1) to p(t) -> use rolling(k).mean()
    m_minus = prices.rolling(window=k, min_periods=k).mean()

    # Future window mean (m_plus): mean of p(t+1) to p(t+k) -> use rolling(k).mean() and shift
    m_plus = prices.rolling(window=k, min_periods=k).mean().shift(-k)

    # Apply classification based on threshold alpha
    # Indices where labels can be calculated: from k-1 to len(df) - k - 1
    valid_idx = df.index[k - 1 : len(df) - k]

    labels.loc[valid_idx] = 1 # Default to stable
    # Note: Crypto-LOB paper uses m_minus > m_plus*(1+alpha) for DOWN (label 2 in their paper, 0 here)
    # and m_minus < m_plus*(1-alpha) for UP (label 1 in their paper, 2 here)
    labels.loc[valid_idx[m_minus.loc[valid_idx] > m_plus.loc[valid_idx] * (1 + alpha)]] = 0 # Down
    labels.loc[valid_idx[m_minus.loc[valid_idx] < m_plus.loc[valid_idx] * (1 - alpha)]] = 2 # Up

    return labels.rename('label')


# --- Main Orchestration Function ---

def preprocess_lob_data(config):
    """
    Loads, preprocesses, adds features, and labels LOB data based on config.

    Args:
        config (dict): Configuration dictionary with keys like:
            'data_path', 'num_files', 'precision', 'clean_strategy',
            'num_levels_features', 'label_method', 'label_k', 'label_h',
            'label_alpha', 'output_path', 'save_intermediate'

    Returns:
        pd.DataFrame: Fully preprocessed dataframe with features and labels.
                      (Or saves to file and returns None if output_path specified)
    """
    # 1. Load Data
    df = load_deep_orderbook_dataset(
        base_path=config.data_path or 'data/coinbase_btc_usd/coinbase/btc_usd/l2_snapshots/100ms/',
        num_files=config.num_files or None,
        precision=config.precision or np.float32,
    )

    # 2. Clean Data
    df = clean_data(
        df,
        strategy=config.clean_strategy or 'dropna',
        verbose=config.verbose or True
    )

    # 3. Calculate Basic Features
    df = calculate_basic_features(
        df,
        num_levels=config.num_levels_features or 10
    )

    # 4. Create Labels
    labels = create_labels(
        df,
        price_col=config.label_price_col or 'mid_price',
        method=config.label_method or 'tlob',
        k=config.label_k or 20,
        h=config.label_h or 10, # Only used if method='tlob'
        alpha=config.label_alpha or 0.0002
    )
    df = pd.concat([df, labels], axis=1)

    # 5. Drop rows with NaN labels (generated at edges)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int) # Convert labels to int

    if config.verbose or True:
        print(f"Shape after label creation and NaN drop: {df.shape}")
        print("Label distribution:")
        print(df['label'].value_counts(normalize=True))

    # Note: Normalization is typically done AFTER splitting train/val/test
    # So, this function returns the unnormalized data with labels.
    # The normalization step should happen in the training script.

    # 6. Save Output (Optional)
    output_path = config.output_path or None
    if output_path:
        if config.verbose or True:
            print(f"Saving preprocessed data to {output_path}...")
        # Consider saving format (parquet, feather, hdf5)
        df.to_parquet(output_path, index=True)
        if config.verbose or True:
            print("Save complete.")
        return None # Indicate data was saved
    else:
        return df

def split_data_chronological(df, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    """
    Splits a DataFrame chronologically based on fractions.

    Args:
        df (pd.DataFrame): Input DataFrame sorted by time.
        train_frac (float): Fraction for training set.
        val_frac (float): Fraction for validation set.
        test_frac (float): Fraction for test set.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    n_total = len(df)
    train_split_idx = int(n_total * train_frac)
    val_split_idx = int(n_total * (train_frac + val_frac))

    train_df = df.iloc[:train_split_idx].copy()
    val_df = df.iloc[train_split_idx:val_split_idx].copy()
    test_df = df.iloc[val_split_idx:].copy()

    print(f"Data Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def normalize_features(df_train, df_val, df_test, feature_cols):
    """
    Applies StandardScaler normalization. Fits on training data only.
    (Keep implementation from previous response)
    """
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols]) # Fit ONLY on train

    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    # Avoid SettingWithCopyWarning
    df_train_scaled.loc[:, feature_cols] = scaler.transform(df_train[feature_cols])
    df_val_scaled.loc[:, feature_cols] = scaler.transform(df_val[feature_cols])
    df_test_scaled.loc[:, feature_cols] = scaler.transform(df_test[feature_cols])

    print("Normalization applied.")
    return df_train_scaled, df_val_scaled, df_test_scaled, scaler