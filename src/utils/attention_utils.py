"""
Utility functions for attention mechanism analysis.

These functions help to count attention heads and retrieve named attention layers from a model.
"""

import torch.nn as nn


def count_attention_head_numbers(model):
    """
    Counts the total number of attention heads across all MultiheadAttention layers.

    Parameters
    ----------
        model:
            Model to inspect.

    Returns
    -------
        total_heads: int
            Total number of attention heads in the model.
    """

    total_heads = 0

    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            total_heads += module.num_heads

    return total_heads


def get_named_attention_layers(model):
    """
    Retrieves all attention layers along with their names from the model.

    Parameters
    ----------
        model:
            Model to inspect.

    Returns
    -------
        layers: list
            List of tuples (name, attention_layer) for all MHA layers.
    """

    attention_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            attention_layers.append((name, module))

    return attention_layers
