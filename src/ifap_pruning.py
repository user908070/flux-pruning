"""
IFAP Implementation.

The module implements the IFAP (Iterative Flow-Aware Pruning) algorithm for pruning neural networks.
"""

import copy
import logging
import os
from tqdm import tqdm

import torch
from torch import nn
import torch_pruning as tp

from models.distillation import distill_model
from ifap_pruning.compute_flow import compute_neurons_flow_from_loader
from utils.common_utils import (
    get_module_by_name,
    generate_model_summary,
)
from utils.pruning_utils import is_prunable_layer

def ifap_pruning_algorithm(
    save_model_path,
    pretrained_model,
    test_dataloader,
    train_dataloader,
    num_classes,
    performance_metric_foo,
    max_metric_degradation_th,
    pruning_iterations,
    pruning_percentage,
    pruning_alpha,
    device,
    save_checkpoints,
    writer,
):
    """
    IFAP Pruning Algorithm.

    This function implements the IFAP pruning algorithm to iteratively prune a neural network model
    while maintaining its performance through knowledge distillation.

    Parameters
    ----------
        save_model_path: str
            Path to save the pruned model.
        pretrained_model: torch.nn.Module
            The pretrained model to be pruned.
        test_dataloader: torch.utils.data.DataLoader
            DataLoader for the test dataset.
        train_dataloader: torch.utils.data.DataLoader
            DataLoader for the training dataset.
        num_classes: int
            Number of classes in the dataset.
        performance_metric_foo: callable
            Function to evaluate model performance.
        max_metric_degradation_th: float
            Maximum allowed degradation in performance metric.
        pruning_iterations: int
            Number of pruning iterations to perform.
        pruning_percentage: float
            Base percentage of neurons to prune in each iteration.
        pruning_alpha: float
            Exponent for scaling the pruning percentage.
        device: str
            Device to run the model on.
        save_checkpoints: bool
            Whether to save checkpoints of the model during pruning.
        writer: torch.utils.tensorboard.SummaryWriter
            TensorBoard writer for logging metrics.

    Returns
    -------
        torch.nn.Module:
            The pruned model with the best performance.
    
    Raises
    ------
        ValueError:
            If the model is not a valid torch.nn.Module.
        RuntimeError:
            If the pruning process fails due to insufficient resources.
    """

    # Create directory for saving the model if it does not exist
    dir_name = os.path.dirname(save_model_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Pretrained model to device
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()

    # Evaluate init model metrics
    logging.debug("Evaluating initial model performance...")
    init_acc_top1 = performance_metric_foo(pretrained_model, test_dataloader, device, k=1)
    init_acc_top5 = performance_metric_foo(pretrained_model, test_dataloader, device, k=5)

    # Log init metrics
    if writer is not None:
        writer.add_scalar('Accuracy/Initial_Top1', init_acc_top1, 0)
        writer.add_scalar('Accuracy/Initial_Top5', init_acc_top5, 0)

    logging.debug("Initial TOP-1 accuracy: %s%%", init_acc_top1)
    logging.debug("Initial TOP-5 accuracy: %s%%", init_acc_top5)

    # Get initial model statistics
    initial_model_summary = generate_model_summary(pretrained_model, device=device)
    logging.debug("Initial model summary: %s", initial_model_summary)

    # teacher model for distillation
    teacher_model = copy.deepcopy(pretrained_model)

    best_model = copy.deepcopy(pretrained_model)
    best_accuracy = init_acc_top1

    for t in range(pruning_iterations):
        # Current model summary
        model_summary = generate_model_summary(pretrained_model, device=device)
        logging.debug("Model summary, step %s: %s", model_summary, t)

        if writer is not None:
            writer.add_scalar('Model/Parameters', model_summary['total_params'], t)
            writer.add_scalar('Model/FLOPs', model_summary['gflops'], t)

        layers_to_prune = [
            (name, m)
            for name, m in pretrained_model.named_modules()
            if is_prunable_layer(m, num_classes)
        ]

        if not layers_to_prune:
            logging.info("No layers to prune. Stopping pruning.")
            break

        # Compute flow importance of neurons
        neurons_flow_importance = compute_neurons_flow_from_loader(
            pretrained_model,
            test_dataloader,
            layers_to_prune,
            max_batches=100,
            device=device,
            writer=writer,
            iteration=t,
        )
        neurons_flow_importance.sort(key=lambda x: x[0])

        # Calculate pruning percentage for this iteration
        pruning_percentage = 100*pruning_percentage*(1+t/pruning_iterations)**pruning_alpha
        num_to_prune = int(pruning_percentage*len(neurons_flow_importance))
        
        # Log pruning statistics
        if writer is not None:
            writer.add_scalar('Pruning/Percentage', pruning_percentage, t)
            writer.add_scalar('Pruning/Neurons_Removed', num_to_prune, t)

        logging.debug(
            "Pruning %s neurons; total prunable elements: %s",
            num_to_prune,
            len(neurons_flow_importance),
        )

        if num_to_prune == 0:
            logging.debug("No more filters to prune.")
            break

        # Select neurons to prune
        prune_targets = neurons_flow_importance[:num_to_prune]

        to_prune_dict = {}
        # Group neurons by layer
        for _, name, neuron_idx in tqdm(prune_targets):
            if name not in to_prune_dict:
                to_prune_dict[name] = [neuron_idx]
            else:
                to_prune_dict[name].append(neuron_idx)

        # Prune the model
        for name, idxs in to_prune_dict.items():
            # Dependency graph of the model
            dependency_graph = tp.DependencyGraph().build_dependency(
                pretrained_model,
                example_inputs=dummy_input
            )
            module = get_module_by_name(pretrained_model, name)

            # Create pruning group based on layer type
            if isinstance(module, nn.Conv2d):
                pruning_group = dependency_graph.get_pruning_group(
                    module,
                    tp.prune_conv_out_channels,
                    idxs=idxs
                )
            elif isinstance(module, nn.Linear):
                pruning_group = dependency_graph.get_pruning_group(
                    module,
                    tp.prune_linear_out_channels,
                    idxs=idxs
                )
            elif isinstance(module, nn.MultiheadAttention):
                # MHA
                pruning_group = dependency_graph.get_pruning_group(
                    module,
                    tp.prune_multihead_attention_out_channels,
                    idxs=idxs
                )
            else:
                logging.warning("Layer %s, skipping pruning.", name)
                continue

            # Execute pruning if valid
            if dependency_graph.check_pruning_group(pruning_group):
                pruning_group.prune()
                torch.cuda.empty_cache()

        # Apply knowledge distillation to recover performance
        distill_model(
            pretrained_model,
            teacher_model,
            dummy_input,
            train_dataloader,
            test_dataloader,
            performance_metric_foo,
            init_acc_top1,
            max_metric_degradation_th,
            device,
            writer=writer,
        )
        teacher_model = copy.deepcopy(pretrained_model)

        # Evaluate current model performance
        cur_acc_top1 = performance_metric_foo(pretrained_model, test_dataloader, device, k=1)
        cur_acc_top5 = performance_metric_foo(pretrained_model, test_dataloader, device, k=5)

        # Log current performance metrics
        if writer is not None:
            writer.add_scalar('Accuracy/Current_Top1', cur_acc_top1, t)
            writer.add_scalar('Accuracy/Current_Top5', cur_acc_top5, t)
            writer.add_scalar('Accuracy/Degradation', init_acc_top1 - cur_acc_top1, t)

        logging.debug("Current Top-1 Accuracy: %s%%", cur_acc_top1)
        logging.debug("Current Top-5 Accuracy: %s%%", cur_acc_top5)

        # Check if performance degradation exceeds threshold
        if (
            init_acc_top1 - cur_acc_top1
            > max_metric_degradation_th
        ):
            logging.info("Accuracy degradation threshold exceeded. Stopping pruning.")
            break

        # Update best model if current performance is better
        if cur_acc_top1 > best_accuracy:
            best_accuracy = cur_acc_top1
            best_model = copy.deepcopy(pretrained_model)
            if writer is not None:
                writer.add_scalar('Accuracy/Best', best_accuracy, t)

    return best_model
