"""
Dataset module.

The module implements functions for handling datasets, including downloading and transforming images.
"""

import os

from torchvision import transforms

from kaggle.api.kaggle_api_extended import KaggleApi


def get_transform(model_name, dataset_name):
    """
    Gets the image transformation pipeline for the given model and dataset.
    
    Parameters
    ----------
        model_name: str
            Name of the model architecture.
        dataset_name: str
            Name of the dataset.
    
    Returns
    -------
        transform: torchvision.transforms.Compose
            The composed image transformation pipeline.
    """

    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]

    # Map of model names to their required input sizes
    dataset_size_mapping = {
        "EfficientNetB4": 380,
        "EfficientNetV2S": 384,
        "ViTBasePatch16": 224,
        "MobileNetV3L": 224,
        "DenseNet121": 224,
        "ConvNextSmall": 224,
        "InceptionV3": 299,
        "VGG19BN": 224,
        "ShuffleNetV2x2_0": 224,
        "ResNet50": 224,
    }

    # Define the transform
    if dataset_name != "ImageNet":
        # Get the input size for the model
        size = dataset_size_mapping[model_name]

        # gray datasets
        if dataset_name in ["Fashion_MNIST", "FER2013"]:
            transform = transforms.Compose(
                [
                    transforms.Resize((size, size)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_mean, std=_std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_mean, std=_std),
                ]
            )

    else:
        t = [
            transforms.ToTensor(),
            transforms.Normalize(mean=_mean, std=_std)
        ]

        if model_name == "InceptionV3":
            # 299 for InceptionV3
            transform = transforms.Compose([transforms.Resize((299, 299))] + t)
        else:
            # 224 for all other ImageNet models
            transform = transforms.Compose([transforms.Resize((224, 224))] + t)

    return transform


def download_FER2013_dataset(target_directory="./data/FER2013"):
    """
    Downloads the FER2013 dataset from Kaggle.
    
    Parameters
    ----------
        target_directory: str
            The directory where the dataset will be downloaded and extracted.

    Returns
    -------
        None

    Raises
    ------
        kaggle.api.KaggleApiError: If there is an error during the download or extraction process.
    """

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Create the target directory if it does not exist
    os.makedirs(target_directory, exist_ok=True)

    # Download and unzip the FER2013 dataset
    api.dataset_download_files("msambare/fer2013", path=target_directory, unzip=True)
