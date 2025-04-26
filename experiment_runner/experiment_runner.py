import os
import importlib
import copy
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from numpy import ndarray, dtype

# Assuming utils and other modules are importable
from experiment_runner.experiment_utils import (
    create_unique_dir, create_dir, save_config, save_plot,
    save_df, set_nested_attr,
    get_nested_attr, plot_sweep_results
)
from training.trainer import Trainer, Config # Assuming Config class is used
# Import base model and specific model wrappers as needed
# from models.base_model import BaseModelWrapper # If needed directly

import torch

def run_single_trial(
    trial_config: Config,  # Use Config object or dict
    train_dataset: Any,  # e.g., torch.utils.data.Dataset
    val_dataset: Any,
    test_dataset: Optional[Any],
    device: Any,  # e.g., torch.device
    base_dir: str,
    trial_name: str,
    plot_history: bool = True,
    save_model: bool = False,
    verbose : bool = True,
):
    """
    Runs a single training and evaluation trial.

    Args:
        trial_config (Config): Configuration object for this specific trial.
        train_dataset, Datasets for training
        val_dataset, Datasets for validation,
        test_dataset: Datasets for testing.
        device: The device to run on.
        trial_dir (str): The directory to save results for this trial.
        save_model (bool): Whether to save the trained model state.

    Returns:
        Dict[str, Any]: Dictionary containing key evaluation metrics (e.g., from test set).

    Parameters
    ----------
    trial_dir
    save_model
    device
    trial_config
    train_dataset
    test_dataset
    val_dataset
    """
    trainer = None
    training_history = None
    final_results = None

    create_dir(base_dir, trial_name)
    trial_dir = os.path.join(base_dir, trial_name)

    print(f"\n--- Running Trial: {os.path.basename(trial_dir)} ---")
    print(f"Config: {vars(trial_config)}") # Print relevant part of config

    # 1. Save configuration used for this trial
    save_config(trial_config, trial_dir, "trial_config.yaml")

    # 2. Instantiate Trainer (assuming Trainer takes config and datasets)
    # Note: Adjust if Trainer expects loaders directly
    try:
        trainer = Trainer(
            config=trial_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,  # Pass test dataset if Trainer uses it for final eval
            device=device,
            output_dir=trial_dir  # Pass trial dir for potential internal saving
        )
    except Exception as e:
        print(f"Error instantiating Trainer: {e}")
        return trainer, training_history, final_results


    # 3. Run Training
    try:
        trainer.train() # Assuming train returns history
        print("Training complete.")
    except Exception as e:
        print(f"Error during training: {e}")
        return trainer, training_history, final_results

    # 4. Evaluate on Validation and Test sets
    try:
        test_metrics = {}
        if test_dataset:
            test_loss, test_score = trainer.test()
            print(f"Test Loss: {test_loss}, Test Score: {test_score}")
        else:
            print("No test dataset provided, skipping test evaluation.")
    except Exception as e:
        print(f"Error during Test Evaluation: {e}")
        return trainer, training_history, final_results

    # 5. Log Results
    # Combine metrics for saving
    try:
        training_history = {
            "epochs": np.linspace(1,trainer.n_epochs, trainer.n_epochs),
            "train_loss": trainer.train_loss,
            "train_score": trainer.train_score,
            "val_loss": trainer.val_loss,
            "val_score": trainer.val_score
        }

        final_results = {
            "epochs": trainer.n_epochs,
            "train_loss": float(trainer.train_loss[-1]),
            "train_score": float(trainer.train_score[-1]),  # <-- .item() / float()
            "val_loss": float(trainer.val_loss[-1]),
            "val_score": float(trainer.val_score[-1]),
            "test_loss": float(trainer.test_loss[-1]),
            "test_score": float(trainer.test_score[-1]),
        }

        df_training_history = pd.DataFrame(training_history)
        df_final_results = pd.DataFrame([final_results])
        save_df(df_training_history, trial_dir, filetype='csv', filename="training_history.csv")
        save_df(df_final_results, trial_dir, filetype='csv', filename="final_results.csv")

    except Exception as e:
        print(f"Error saving metrics to {trial_dir}: {e}")
        return trainer, training_history, final_results

    # Plot and save learning curves from history
    try:
         fig_loss, ax_loss = plt.subplots( figsize=(15, 5))
         ax_loss.plot(df_training_history['epochs'], df_training_history['train_loss'], label='Train Loss')
         ax_loss.plot(df_training_history['epochs'], df_training_history['val_loss'], label='Validation Loss')
         ax_loss.axhline(df_final_results['test_loss'].iloc[0], color='red', linestyle='--', label='Test Loss')
         ax_loss.set_title('Loss vs Epoch')
         ax_loss.set_xlabel('Epoch')
         ax_loss.set_ylabel('Loss')
         ax_loss.legend()
         ax_loss.grid(True)
         fig_loss.tight_layout()

         fig_score, ax_score = plt.subplots(figsize=(15, 5))
         ax_score.plot(df_training_history['epochs'], df_training_history['train_score'], label='Train Scores')
         ax_score.plot(df_training_history['epochs'], df_training_history['val_score'], label='Validation Scores')
         ax_score.axhline(df_final_results['test_score'].iloc[0], color='red', linestyle='--', label='Test Score')
         ax_score.set_title(f'F1 Accuracy Score vs Epoch')
         ax_score.set_xlabel('Epoch')
         ax_score.set_ylabel('F1 Accuracy')
         ax_score.legend()
         ax_score.grid(True)
         fig_score.tight_layout()

         save_plot(fig_loss, trial_dir, "learning_curves_loss.png")
         save_plot(fig_score, trial_dir, "learning_curves_score.png")
    except Exception as e:
        print(f"Error Creating Plots: {e}")
        return trainer, training_history, final_results

    if plot_history:
        plt.show()
    else:
        fig_loss.close()
        fig_score.close()

    # Save model if requested
    if save_model:
        trainer.save_model(name="final_model.pt") # Assuming Trainer has save_model method
        print(f"Model saved in {trial_dir}")

    return trainer, training_history, final_results


def run_parameter_sweep(
    base_config,
    sweep_param_name,
    sweep_values,
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Optional[Any],
    device: Any,
    sweep_dir: str,
    save_trial_models: bool = False,
    verbose=True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs multiple trials by sweeping over a single hyperparameter.

    Args:
        base_config (Config): The base configuration object.
        sweep_param_name (str): Dot-separated name of the parameter to sweep (e.g., 'train.learning_rate').
        sweep_values (List[Any]): List of values to try for the sweep parameter.
        train_dataset, val_dataset, test_dataset: Datasets.
        device: Torch device.
        sweep_dir (str): Directory to save all results for this sweep.
        metric_to_optimize (str): Key for the metric in the results dict to determine the best trial.
        save_trial_models (bool): Whether to save the model for each individual trial.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - DataFrame summarizing parameters and results for all trials.
            - Dictionary representing the configuration of the best trial.
    """
    print(f"\n===== Running Sweep for Parameter: {sweep_param_name} =====")
    results_list = []
    best_score = -float('inf') # Assuming higher is better
    best_config_dict = None

    for i, value in enumerate(sweep_values):
        trial_config = copy.deepcopy(base_config)
        try:
            set_nested_attr(trial_config, sweep_param_name, value)
        except (AttributeError, KeyError) as e:
            print(f"Error setting sweep parameter '{sweep_param_name}': {e}. Skipping value {value}.")
            continue

        # Create a subdirectory for this trial within the sweep directory
        trial_name = f"trial_{i:02d}_{sweep_param_name.split('.')[-1]}_{value}"
        # trial_name = f"{sweep_param_name}_{value}"
        # trial_dir = os.path.join(sweep_dir, trial_name)
        # os.makedirs(trial_dir, exist_ok=True)

        # Run the trial
        trainer, training_history, final_results = run_single_trial(trial_config=trial_config,
                                                   train_dataset=train_dataset,
                                                   val_dataset=val_dataset,
                                                   test_dataset=test_dataset,
                                                   device=device,
                                                   base_dir=sweep_dir,
                                                   trial_name=trial_name,
                                                   save_model=save_trial_models,
                                                   verbose=True)

        # Store results
        trial_summary = {sweep_param_name: value}
        # Extract metrics from validation or test based on availability
        trial_summary.update(final_results) # Add all metrics from the chosen source
        results_list.append(trial_summary)

        # Track best score (using validation metric if test not available or errored)
        current_score = final_results['test_score']
        if current_score > best_score:
            best_score = current_score
            best_config_dict = copy.deepcopy(vars(trial_config))  # Store the best config dict

    # Aggregate results
    results_df = pd.DataFrame(results_list)
    save_df(results_df, sweep_dir, filetype='csv', filename="sweep_summary.csv")

    # # Plot results
    metrics_to_plot = [m for m in ['train_loss', 'train_score',
                                   'val_loss', 'val_score',
                                   'test_loss', 'test_score',] if m in results_df.columns]
    plot_sweep_results(results_df, sweep_param_name, metrics_to_plot, sweep_dir)
    #
    print(f"===== Sweep Complete for: {sweep_param_name} =====")
    print(f"Best Score (Test F1 Score): {best_score}")
    print(f"Best Config Params: {best_config_dict}")  # Potentially very long

    return results_df, best_config_dict


def run_multiple_sweeps(
    base_config: Config,
    sweeps_dict: Dict[str, List[Any]],
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Optional[Any],
    device: Any,
    base_experiment_dir: str,
    metric_to_optimize: str = 'test_score',
    save_trial_models: bool = False
):
    """
    Runs multiple single-parameter sweeps sequentially.

    Args:
        base_config (Config): Base configuration.
        model_type (str): Model type.
        sweeps_dict (Dict[str, List[Any]]): Dictionary where keys are sweep parameter names
                                           and values are lists of values to sweep.
        train_dataset, val_dataset, test_dataset: Datasets.
        device: Torch device.
        base_experiment_dir (str): The main directory for this set of sweeps.
        metric_to_optimize (str): Metric to optimize within each sweep.
        save_trial_models (bool): Whether to save models for each trial in the sweeps.
    """
    print(f"\n======= Starting Multiple Sweeps in: {base_experiment_dir} =======")
    all_sweep_summaries = {}
    best_overall_score = -float('inf')
    best_overall_config = None
    best_sweep_name = None

    for sweep_param_name, sweep_values in sweeps_dict.items():
        sweep_name = f"sweep_{sweep_param_name.replace('.', '_')}"
        sweep_dir = os.path.join(base_experiment_dir, sweep_name)
        os.makedirs(sweep_dir, exist_ok=True)

        summary_df, best_trial_config = run_parameter_sweep(
            base_config=base_config,
            sweep_param_name=sweep_param_name,
            sweep_values=sweep_values,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
            sweep_dir=sweep_dir,
            save_trial_models=save_trial_models
        )
        all_sweep_summaries[sweep_param_name] = summary_df

        # Track the best score across all sweeps run so far
        current_best_score = summary_df[metric_to_optimize].max()
        if isinstance(current_best_score, (int, float)) and current_best_score > best_overall_score:
             best_overall_score = current_best_score
             best_overall_config = best_trial_config # Config from the best trial of this sweep
             best_sweep_name = sweep_param_name


    print(f"======= Multiple Sweeps Complete =======")
    print(f"Best Overall Score ({metric_to_optimize}): {best_overall_score} (from sweep '{best_sweep_name}')")
    # Optionally save the overall best config or re-run the best trial

    return all_sweep_summaries, best_overall_config


def orchestrate_experiment(
    experiment_name: str,
    train_dataset: Any,  # Pass datasets directly for now
    val_dataset: Any,
    test_dataset: Optional[Any],
    base_config,
    sweeps_to_run: Optional[Dict[str, List[Any]]] = None, # Dict for sweeps, None for single run
    # Add args for data paths or data loading function if needed
    base_results_dir: str = "results",
    device_str: str = "auto", # "auto", "cuda", "cpu", "mps"
    save_models_in_sweep: bool = False, # Only save best model usually
    save_final_model: bool = True
):
    """
    High-level function to set up and run an experiment (single run or multiple sweeps).
    """
    # 1. Setup Device
    if device_str == "auto":
         device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
         device = torch.device(device_str)
    print(f"Using device: {device}")

    # 3. Create Main Experiment Directory
    exp_dir = create_unique_dir(base_results_dir, experiment_name)

    # 4. Run Experiment(s)
    if sweeps_to_run:
        print("Starting hyperparameter sweeps...")
        _, best_config_dict = run_multiple_sweeps(
            base_config=base_config,
            sweeps_dict=sweeps_to_run,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
            base_experiment_dir=exp_dir,
            metric_to_optimize='test_score',
            save_trial_models=save_models_in_sweep
        )
        # Optionally run the best config found
        # print("\nRunning best configuration found during sweep...")
        # best_config = Config(best_config_dict) # Convert back if needed
        # run_single_trial(...) # Call with best_config and save in a 'best_trial' subdir

    else:
        print("Starting single experiment run...")
        # 1. Run the trial and capture results
        trainer, training_history, final_results = run_single_trial(
            trial_config=base_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
            base_dir=exp_dir,
            trial_name=experiment_name,
            save_model=save_final_model,
            verbose=True
        )

        # 2. If you got back real history & results, persist them
        if training_history is not None and final_results is not None:
            df_training_history = pd.DataFrame(training_history)
            df_final_results = pd.DataFrame([final_results])
            save_df(df_training_history, exp_dir,
                    filetype='csv', filename="training_history.csv")
            save_df(df_final_results, exp_dir,
                    filetype='csv', filename="final_results.csv")
            print("Final Results:", final_results)

    print(f"Experiment '{experiment_name}' finished. Results in: {exp_dir}")