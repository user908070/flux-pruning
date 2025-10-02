"""
IFAP Pruning Pipeline
The module implements a pipeline for compressing neural network models using IFAP pruning.
"""

import logging
import os
import yaml

from ifap_pruning.process_pruning import process_pruning

# Logging to a file
logging.basicConfig(
    filename="logs/ifap_pruning.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
    datefmt="%d.%m.%Y %H:%M",
)


def main():
    """
    Runs the IFAP pruning pipeline.

    Iterates over all model-dataset combinations defined in the configuration 
    files, loading the models and datasets, and applying the IFAP pruning 
    algorithm.
    """

    # Define configs root directory
    configs_root = os.path.join(os.path.dirname(__file__), 'configs')

    # Load weights_paths config from yaml file
    with open(os.path.join(configs_root, 'model_configs.yaml'), 'r') as f:
        weights_paths = yaml.safe_load(f)["weights_paths"]

    # Load datasets config from yaml file
    with open(os.path.join(configs_root, 'datasets.yaml'), 'r') as f:
        datasets = yaml.safe_load(f)["datasets"]

    # Iterate over all combinations of models and datasets
    logging.debug("Starting pruning for all model-dataset combinations")

    for weights_path in weights_paths:
        for dataset in datasets:
            process_pruning(weights_path, dataset)


if __name__ == "__main__":
    main()
