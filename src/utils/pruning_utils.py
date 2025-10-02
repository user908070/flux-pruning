"""
Utility functions for pruning the model.

These functions help to determine if a layer is prunable, and to freeze or unfreeze model parameters.
"""

import torch.nn as nn


def is_prunable_layer(module, num_classes=10):
    """
    Checks whether the layer is prunable.

    Parameters
    ----------
        module
            Layer to check.
        num_classes: int
            Number of output classes in the model.

    Returns
    -------
        value: bool
            True if the layer can be pruned, False otherwise.
    """

    # Exclude 1x1 conv layers
    if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
        return True

    # Exclude last linear layer to preserve output structure
    elif isinstance(module, nn.Linear) and module.out_features != num_classes:        
        return True

    elif isinstance(module, nn.MultiheadAttention):
        return True

    return False


def freeze_model(model):
    """
    Freezes all parameters in the model (set requires_grad=False).

    Parameters
    ----------
        model: 
            Model to freeze.

    Returns
    -------
        None
    """

    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """
    Unfreezes all parameters in the model (set requires_grad=True).

    Parameters
    ----------
        model: 
            Model to unfreeze.

    Returns
    -------
        None
    """

    for param in model.parameters():
        param.requires_grad = True
