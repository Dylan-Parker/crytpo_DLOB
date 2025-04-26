import os
import time
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

def create_unique_dir(base_dir: str, name_prefix: str) -> str:
    """
    Creates a unique directory within base_dir using a timestamp.

    Args:
        base_dir (str): The parent directory.
        name_prefix (str): A prefix for the directory name.

    Returns:
        str: The path to the created unique directory.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dir_name = f"{name_prefix}_{timestamp}"
    full_path = os.path.join(base_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)
    print(f"Created experiment directory: {full_path}")
    return full_path

def create_dir(base_dir: str, dir_name: str) -> str:
    """
    Creates a unique directory within base_dir using a timestamp.

    Args:
        base_dir (str): The parent directory.
        name_prefix (str): A prefix for the directory name.

    Returns:
        str: The path to the created unique directory.
    """
    dir_name = f"{dir_name}"
    full_path = os.path.join(base_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)
    print(f"Created experiment directory: {full_path}")
    return full_path

def save_config(config_obj: Any, dir_path: str, filename: str = "config.yaml"):
    """
    Saves a configuration object (dict or custom Config class) to YAML.

    Args:
        config_obj (Any): The configuration object (must be YAML-serializable or have a to_dict method).
        dir_path (str): The directory to save the file in.
        filename (str): The name of the configuration file.
    """
    filepath = os.path.join(dir_path, filename)
    try:
        # If it's a custom Config class like the one in trainer.py, convert to dict
        if hasattr(config_obj, '__dict__'):
            # A simple way, might need refinement for nested Config objects
            config_dict = config_obj.__dict__
            # Handle nested Config objects if necessary
            def config_to_dict(cfg):
                d = {}
                for k, v in cfg.__dict__.items():
                    if hasattr(v, '__dict__'):
                        d[k] = config_to_dict(v)
                    else:
                        d[k] = v
                return d
            config_dict = config_to_dict(config_obj)

        elif isinstance(config_obj, dict):
            config_dict = config_obj
        else:
            raise TypeError("config_obj must be a dictionary or have a __dict__ attribute.")

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Configuration saved to {filepath}")
    except Exception as e:
        print(f"Error saving configuration to {filepath}: {e}")


def save_plot(fig: plt.Figure, dir_path: str, filename: str):
    """
    Saves a matplotlib figure.

    Args:
        fig (plt.Figure): The matplotlib figure object.
        dir_path (str): The directory to save the plot in.
        filename (str): The name for the plot file (e.g., 'learning_curve.png').
    """
    filepath = os.path.join(dir_path, filename)
    try:
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")
        plt.close(fig) # Close the figure to free memory
    except Exception as e:
        print(f"Error saving plot to {filepath}: {e}")

def save_df(df, dir_path: str, filetype: str='csv', filename: str = "metrics.csv", index=False):
    """
    Saves a dictionary of metrics to a JSON file.

    Args:
        df: Dataframe
        dir_path (str): The directory to save the metrics file in.
        filename (str): The name for the metrics file.
    """
    filepath = os.path.join(dir_path, filename)

    try:
        if filetype == 'csv':
            df.to_csv(filepath, index=index)
        print(f"Dataframe saved to {filepath}")
    except Exception as e:
        print(f"Error saving Dataframe to {filepath}: {e}")

def set_nested_attr(config_obj: Any, attr_string: str, value: Any):
    """
    Sets a potentially nested attribute in a config object (dict or class instance).

    Example: set_nested_attr(config, 'train.optimizer.lr', 0.001)

    Args:
        config_obj (Any): The configuration object (dict or class instance).
        attr_string (str): Dot-separated string representing the nested attribute.
        value (Any): The value to set.
    """
    keys = attr_string.split('.')
    current_level = config_obj
    for i, key in enumerate(keys):
        if i == len(keys) - 1: # Last key
            if isinstance(current_level, dict):
                current_level[key] = value
            else:
                setattr(current_level, key, value)
        else:
            if isinstance(current_level, dict):
                current_level = current_level.setdefault(key, {}) # Use dict for intermediate levels if needed
            else:
                # If it's an object, assume intermediate attributes exist
                if not hasattr(current_level, key):
                     # Or create a simple object/dict if appropriate for your Config structure
                     setattr(current_level, key, type('Config', (), {})()) # Example: create empty object
                current_level = getattr(current_level, key)


def get_nested_attr(config_obj: Any, attr_string: str) -> Any:
    """
    Gets a potentially nested attribute from a config object (dict or class instance).

    Args:
        config_obj (Any): The configuration object.
        attr_string (str): Dot-separated string representing the nested attribute.

    Returns:
        Any: The value of the attribute.

    Raises:
        AttributeError/KeyError: If the attribute path is invalid.
    """
    keys = attr_string.split('.')
    current_level = config_obj
    for key in keys:
        if isinstance(current_level, dict):
            current_level = current_level[key]
        else:
            current_level = getattr(current_level, key)
    return current_level

def plot_sweep_results(results_df: pd.DataFrame, sweep_param: str, metrics: List[str], save_dir: str):
    """
    Generates plots for hyperparameter sweep results.

    Args:
        results_df (pd.DataFrame): DataFrame containing sweep results (one row per trial).
                                   Must include the sweep_param column and metric columns.
        sweep_param (str): The name of the hyperparameter that was swept.
        metrics (List[str]): A list of metric column names to plot against the sweep_param.
        save_dir (str): The directory to save the plots in.
    """
    if sweep_param not in results_df.columns:
        print(f"Warning: Sweep parameter '{sweep_param}' not found in results DataFrame. Skipping plots.")
        return

    for metric in metrics:
        if metric not in results_df.columns:
            print(f"Warning: Metric '{metric}' not found in results DataFrame. Skipping plot.")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        # Handle numeric vs categorical sweep param for plotting
        try:
            # Attempt numeric plot if possible
            param_values = pd.to_numeric(results_df[sweep_param])
            metric_values = pd.to_numeric(results_df[metric])
            ax.plot(param_values, metric_values, marker='o', linestyle='-')
            ax.set_xlabel(sweep_param)
        except ValueError:
            # Fallback to categorical plot (e.g., bar plot or simple scatter)
            ax.scatter(results_df[sweep_param].astype(str), results_df[metric])
            ax.set_xlabel(f"{sweep_param} (Categorical)")
            plt.xticks(rotation=45, ha='right')


        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {sweep_param}")
        ax.grid(True)
        plt.tight_layout()
        save_plot(fig, save_dir, f"sweep_{metric}_vs_{sweep_param}.png")