"""
Single model-dataset combination pruning.

The module implements a function to prune a single model-dataset pair using the IFAP pruning algorithm.
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ifap_pruning.ifap_pruning import ifap_pruning_algorithm
from models.models import load_model_with_dataset
from utils.common_utils import set_random_seed


def process_pruning(model_dict, dataset):
    """
    Prunes a single model-dataset pair.

    Parameters
    ----------
        model_dict: dict
            Python dictionary with model name and weight paths.
        dataset: str
            The name of the dataset to use.

    Returns
    -------
        None
    """

    # Define the device
    device = torch.device("cuda")

    # Set random seed for reproducibility
    set_random_seed(seed=12345)

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Extract model name
    model_name = model_dict["name"]

    # Extract model path
    model_path = model_dict["paths"][dataset]

    # Load the pretrained model and the dataset
    pretrained_model, num_classes, train_set, test_set = load_model_with_dataset(
        model_path, model_name, dataset
    )

    # Create data loaders for training and testing
    train_dataloader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=100, shuffle=False)

    # Log initial model statistics
    total_params = sum(p.numel() for p in pretrained_model.parameters())
    writer.add_scalar(
        f'{model_name}_{dataset}/Initial_Parameters', total_params, 0)

    # Process model pruning
    pruned_model = ifap_pruning_algorithm(
        save_model_path=model_path,
        pretrained_model=pretrained_model,
        test_dataloader=test_dataloader,
        train_dataloader=train_dataloader,
        num_classes=num_classes,
        max_metric_degradation_th=0.05,
        pruning_iterations=30,
        pruning_percentage=0.10,
        device=device,
        writer=writer
    )

    # Log model statistics after pruning
    total_params_after = sum(p.numel() for p in pruned_model.parameters())
    writer.add_scalar(f'{model_name}_{dataset}/Final_Parameters', total_params_after, 1)
    writer.add_scalar(
        f'{model_name}_{dataset}/Parameters_Reduction',
        (total_params - total_params_after) / total_params * 100,
        1
    )

    # Save the pruned model
    pruned_model_path = model_path.replace(".pth", f"{dataset}_pruned.pth")
    torch.save(pruned_model, pruned_model_path)

    # Close the TensorBoard writer
    writer.close()
