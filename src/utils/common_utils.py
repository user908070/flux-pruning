"""
Utility functions for model introspection and reproducibility.

This module provides functions to retrieve modules from a model by name,
generate model summaries, and set random seeds for reproducibility.
"""

import numpy as np
import random

import torch
from torchinfo import summary


def get_module_by_name(model, name):
    """
    Gets a submodule from the model.

    Parameters
    ----------
        model: torch.nn.Module
            The model from which to retrieve the submodule.
        name: str
            The name of the submodule, specified as a dot-separated path.

    Returns
    -------
        torch.nn.Module: The submodule.

    Raises
    ------
        TypeError: If the model is not an instance of torch.nn.Module.
        ValueError: If the module name path is invalid or module not found.
    """

    # Get module parts
    parts = name.split(".")
    module = model

    # Check if model is a valid torch.nn.Module
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = module._modules.get(part)
            if module is None:
                raise ValueError(f"Module '{part}' not found in the model.")

    return module


def generate_model_summary(model, device="cuda"):
    """
    Generates a summary of the model.

    Parameters
    ----------
        model: torch.nn.Module
            The model to generate the summary for.
        device: str
            The device to run the model on.

    Returns
    -------
        dict: A dictionary containing the model summary.
    """

    # Get summary with a dummy input
    summary = summary(model, (1, 3, 224, 224), device=device, verbose=0)

    # Extract relevant information from the summary
    total_params = summary.total_params
    trainable_params = summary.trainable_params
    gflops = summary.total_mult_adds / 1e9
    forward_backward_pass_size_MB = summary.total_output_bytes / 1024**2
    params_size_MB = summary.total_param_bytes / 1024**2
    estimated_total_memory_MB = (
        summary.total_input +
        summary.total_output_bytes +
        summary.total_param_bytes
    ) / 1024**2

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "gflops": gflops,
        "forward_backward_pass_size_MB": forward_backward_pass_size_MB,
        "params_size_MB": params_size_MB,
        "estimated_total_memory_MB": estimated_total_memory_MB
    }


def set_random_seed(seed):
    """
    Sets random seed for the results reproducibility.
    All random operations (PyTorch, NumPy, Python) will be deterministic and reproducible.

    Parameters
    ----------
        seed: int:
            Random seed value.

    Returns
    -------
        None
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
