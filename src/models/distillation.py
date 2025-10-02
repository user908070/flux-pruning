"""
Distillation Module

The module implements a feature distillation process between a teacher and a student model.
"""

import logging
from tqdm import tqdm

from torch import nn
import torch
import torch.nn.functional as F


class FeatureDistiller:
    """
    Helper class for distillation.
    """

    def __init__(self, student_model, teacher_model, layers_to_distill):
        """
        Initialize the feature distiller.
        """

        self.student_model = student_model
        self.teacher_model = teacher_model

        self.layers_to_distill = layers_to_distill

        self.alignment_convolutions = nn.ModuleDict()

        self.student_features = {}
        self.teacher_features = {}

        self._register_forward_hooks(
            self.student_model, self.student_features, "student")
        self._register_forward_hooks(
            self.teacher_model, self.teacher_features, "teacher")

    def _register_forward_hooks(self, model, feature_dict, model_type):
        """
        Registers forward hooks to capture features from specified layers.

        Parameters
        ----------
            model:
                Model to register hooks for.
            feature_dict: dicr
                Dictionary to store captured features.
            model_type: str
                Model type: 'student'/ 'teacher'

        Returns
        -------
            None
        """

        for name, module in model.named_modules():
            if name in self.layers_to_distill:
                module.register_forward_hook(
                    self._get_forward_hook(name, feature_dict))

    def _remove_forward_hooks(self, model):
        """
        Removes all forward hooks from the model.

        Parameters
        ----------
            model:
                Model to remove hooks from

        Returns
        -------
            None
        """

        for module in model.modules():
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks.clear()

    def _get_forward_hook(self, name, feat_dict):
        """
        Creates a hook function to capture layer outputs.

        Parameters
        ----------
            name: str
                Name of the layer.
            feat_dict: dict
                Dictionary to store the captured features.

        Returns
        -------
            function:
                Hook function that captures and stores layer outputs
        """

        def hook(module, input, output):
            feat_dict[name] = output

        return hook

    def _init_align_convolutions(self):
        """
        Initialize alignment convolutions for feature dimension matching.
        Creates 1x1 convolutions to match feature dimensions between teacher and student
        models when they don't match exactly.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        for name in self.layers_to_distill:
            s_feat = self.student_features.get(name)
            t_feat = self.teacher_features.get(name)

            in_channels = s_feat.shape[1]
            out_channels = t_feat.shape[1]

            if in_channels != out_channels:
                key = name.replace(".", "__")
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                self.alignment_convolutions[key] = conv.to(s_feat.device)

    def compute_feature_loss(self):
        """
        Computes feature distillation loss between teacher and student models.

        Parameters
        ----------
            None

        Returns
        -------
            loss: float
                Total feature matching loss across all specified layers.
        """

        loss = 0.0
        for name in self.layers_to_distill:
            f_s = self.student_features.get(name)
            f_t = self.teacher_features.get(name)

            if f_s is None or f_t is None:
                continue

            key = name.replace(".", "__")
            if key in self.alignment_convolutions:
                f_s = self.alignment_convolutions[key](f_s)

            if f_s.shape != f_t.shape:
                continue

            # L1 loss between features
            loss += F.l1_loss(f_s, f_t)

        return loss


def distill_model(
    pretrained_model,
    teacher_model,
    dummy_input,
    train_loader,
    test_loader,
    performance_metric_function,
    init_acc_top1,
    max_metric_degradation_th,
    device,
    writer=None,
):
    """
    Performs knowledge distillation between teacher and student models.

    Parameters
    ----------
        pretrained_model:
            Student model to be trained
        teacher_model:
            Teacher model providing knowledge
        dummy_input:
            Sample input for feature extraction
        train_loader:
            DataLoader for training
        test_loader:
            DataLoader for evaluation
        performance_metric_function:
            Function to evaluate model performance
        initial_accuracy_top1:
            Initial top-1 accuracy
        max_performance_metric_degradation_th:
            Maximum allowed accuracy degradation
        device:
            Device to run computations on
        writer:
            TensorBoard writer for logging

    Returns
    -------
        None
    """

    # Select ReLU layers for feature distillation
    layers_for_distillation = [
        name
        for name, module in pretrained_model.named_modules()
        if isinstance(module, nn.ReLU)
    ]

    # Select first, middle, and last ReLU layers for efficiency
    layers_for_distillation = [
        layers_for_distillation[0],
        layers_for_distillation[len(layers_for_distillation) // 2],
        layers_for_distillation[-1],
    ]

    distiller = FeatureDistiller(
        pretrained_model, teacher_model, layers_to_distill=layers_for_distillation
    )

    # Extract initial features with dummy input
    with torch.no_grad():
        _ = teacher_model(dummy_input)
        _ = pretrained_model(dummy_input)

    distiller._init_align_convolutions()

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        list(pretrained_model.parameters()) +
        list(distiller.alignment_convolutions.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.90, patience=20
    )

    # Initialize loss functions
    ce_loss_fn = nn.CrossEntropyLoss()

    # Initialize progress tracking
    progress_bar = tqdm()
    prev_loss = 0.0
    running_loss = 0.0
    it = 0

    for epoch in range(100):
        teacher_model.eval()
        pretrained_model.train()

        # Training iterations
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Extract teacher features
            with torch.no_grad():
                _ = teacher_model(x)

            # Forward pass through student model
            student_outputs = pretrained_model(x)

            # Compute combined loss
            ce_loss = ce_loss_fn(student_outputs, y)
            feat_loss = distiller.compute_feature_loss()
            total_loss = ce_loss + 0.5 * feat_loss
            total_loss.backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()

            # Track loss statistics
            running_loss += total_loss.item()
            avg_loss = running_loss / (it + 1)
            it += 1
            loss_difference = abs(avg_loss - prev_loss)

            # Log metrics to TensorBoard
            if writer is not None:
                writer.add_scalar('Distillation/Total_Loss',
                                  total_loss.item(), it)
                writer.add_scalar('Distillation/CE_Loss', ce_loss.item(), it)
                writer.add_scalar('Distillation/Feature_Loss',
                                  feat_loss.item(), it)
                writer.add_scalar('Distillation/Learning_Rate',
                                  optimizer.param_groups[0]['lr'], it)

            # Check for convergence
            if (i > 100 or epoch > 0) and loss_difference < 1e-4:
                distiller._remove_forward_hooks(pretrained_model)
                logging.debug(
                    "Loss converged, stopping. %f, %f, %f",
                    loss_difference,
                    avg_loss,
                    prev_loss,
                )
                break

            prev_loss = avg_loss

            # Update progress information
            logging_text = f"Epoch: {epoch + 1} - Iter: {i + 1} - Loss: {avg_loss:.7f} - Loss Difference: {loss_difference:.7f} - lr: {optimizer.param_groups[0]['lr']:.7f}"
            logging.debug(logging_text)
            progress_bar.set_description(logging_text)
            progress_bar.update(1)

            # Update learning rate
            scheduler.step(avg_loss)

        # Evaluate current performance
        current_accuracy_top1 = performance_metric_function(
            pretrained_model, test_loader, device, k=1
        )

        # Log epoch metrics
        if writer is not None:
            writer.add_scalar('Distillation/Accuracy',
                              current_accuracy_top1, epoch)
            writer.add_scalar('Distillation/Avg_Loss', avg_loss, epoch)

        logging.debug(f"Epoch {epoch + 1} completed. Avg loss: {avg_loss:.6f}")
        logging.debug(f"Current TOP-1 accuracy: {current_accuracy_top1}%")

        # Check if performance degradation is within acceptable range
        if init_acc_top1 - current_accuracy_top1 <= max_metric_degradation_th:
            distiller._remove_forward_hooks(pretrained_model)
            break

        # Check for convergence
        if loss_difference < 1e-4:
            break
