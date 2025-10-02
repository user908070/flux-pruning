"""
Compute flow importance of neurons in a neural network model.
"""

import torch


def compute_neurons_flow_from_loader(
    model, data_loader, layers_to_prune, max_batches, device, writer, iteration
):
    """
    Compute neuron flow importance from data loader.
    
    Parameters:
    -----------
        model:
            The neural network model to analyze
        data_loader: torch.utils.data.DataLoader
            DataLoader providing input samples
        layers_to_prune: list of tuples
            List of (name, module) tuples for layers that can be pruned
        max_batches: int
            Maximum number of batches to process for importance calculation
        device: str
            Device to run computations on ('cuda' or 'cpu')
        writer: torch.utils.tensorboard.SummaryWriter
            TensorBoard writer for logging importance scores
        iteration: int
            Current pruning iteration for logging

    Returns
    -------
        l: list
            List of (importance_score, layer_name, neuron_index) tuples
    """

    # Ensure model is in evaluation mode
    model.eval()

    # Initialize accumulator for importance scores
    importance_accumulator = {name: [] for name, _ in layers_to_prune}

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            x = x.to(device)

            hooks = []

            def make_hook(name, module):
                def save_divergence(module, input, output):
                    output = output.detach()
                    output_activated = torch.relu(output)
                    weights = module.weight.detach()

                    if len(output_activated.shape) == 4:
                        b, c, h, w = output_activated.shape

                        # Calculate activation norm across spatial dimensions
                        div = output_activated.view(b, c, -1)
                        div = torch.norm(div, dim=2).mean(dim=0)

                        # Calculate weight norm
                        weights = weights.view(c, -1)
                        weights = torch.norm(weights, dim=1)
                    else:
                        b, n = output_activated.shape

                        # Calculate activation norm
                        div = output_activated.view(b, n)
                        div = torch.norm(div, dim=0).mean(dim=0)

                        # Calculate weight norm
                        weights = torch.norm(weights, dim=1)

                    # Combine activation and weight importance
                    div = div * weights

                    importance_accumulator[name].append(div.cpu())

                return save_divergence

            # Register hooks for all prunable layers
            for name, module in layers_to_prune:
                hooks.append(module.register_forward_hook(make_hook(name, module)))

            # Forward pass to compute importance scores
            _ = model(x)

            # Remove hooks after computation
            for h in hooks:
                h.remove()

    # Compute average importance scores across batches
    result = []
    for name, list_of_divs in importance_accumulator.items():
        if list_of_divs:
            stacked = torch.stack(list_of_divs)
            avg = stacked.mean(dim=0)
            
            # Log neuron importance to TensorBoard
            if writer is not None:
                writer.add_histogram(f'Neuron_Importance/{name}', avg, iteration)
            
            # Store importance scores with layer and neuron information
            for i, score in enumerate(avg):
                result.append((score.item(), name, i))

    return result
