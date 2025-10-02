"""
Utility functions for metrics evaluation.

These functions help to evaluate the top-k accuracy of a model on a test dataset.
"""

import time

import torch


def evaluate_topk_accuracy(model, test_loader, device, k=5):
    """
    Evaluates the Top-k accuracy of the model on the given test dataset.

    Parameters
    ----------
        model:
            Model to evaluate.
        test_loader: torch.utils.data.DataLoader
            DataLoader for the test set.
        device: str
            Device to use (cpu / cuda).
        k: int
            Number of top predictions to consider. Default is 5.

    Returns
    -------
        accuracy: float
            Top-k accuracy as a ratio between 0 and 1.
    """

    # model to evaluation mode
    model.eval()


    right_answer = 0
    total_answers = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.topk(k, 1, True, True)
            right_answer += (predicted == y.view(-1, 1)).sum().item()
            total_answers += y.size(0)

    return right_answer / total_answers
