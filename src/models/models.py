"""
Models module.

The module implements functions for working with models and datasets.
"""

import os

import timm
import torch
from torchvision import datasets
from torchvision.datasets.folder import ImageFolder

from models.dataset import download_FER2013_dataset, get_transform


def load_model_with_dataset(model_path, model_name, dataset_name):
    """
    Loads a model and its corresponding dataset based on the provided model name and dataset name.

    Parameters
    ----------
        model_path: str
            Path to the model weights.
        model_name: str
            Name of the model architecture.
        dataset_name: str
            Name of the dataset.

    Returns
    -------
        model: torch.nn.Module
            The loaded model.
        num_classes: int
            Number of classes in the dataset.
        train_dataset: torchvision.datasets.Dataset
            The training dataset.
        test_dataset: torchvision.datasets.Dataset
            The testing dataset.

    Raises
    ------
        ValueError: If the provided dataset name is not supported.
        ValueError: If the provided model name is not supported.
    """

    weights_only = True

    try:
        # Load the model as a checkpoint
        ckpt = torch.load(model_path, map_location="cpu")

    except Exception as e:
        # Load the model as a state dict
        weights_only = False
        model = torch.load(
            model_path, weights_only=weights_only, map_location="cpu")

    # Get image transformation pipeline for the model and dataset
    transform = get_transform(model_name, dataset_name)

    train_root = "./data/train"
    test_root = "./data/test"

    # Load the dataset
    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root=train_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=test_root, train=False, download=True, transform=transform
        )
        num_classes = 10

    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root=train_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root=test_root, train=False, download=True, transform=transform
        )
        num_classes = 100

    elif dataset_name == "Flowers102":
        train_dataset = datasets.Flowers102(
            root=train_root, split="train", download=True, transform=transform
        )
        test_dataset = datasets.Flowers102(
            root=test_root, split="test", download=True, transform=transform
        )
        num_classes = 102

    elif dataset_name == "StanfordCars":
        train_dataset = datasets.StanfordCars(
            root=train_root, split="train", download=True, transform=transform
        )
        test_dataset = datasets.StanfordCars(
            root=test_root, split="test", download=True, transform=transform
        )
        num_classes = 196

    elif dataset_name == "ImageNet":
        train_dataset = datasets.ImageNet(
            root=train_root, split="train", download=True, transform=transform
        )
        test_dataset = datasets.ImageNet(
            root=test_root, split="val", download=True, transform=transform
        )
        num_classes = 1000

    elif dataset_name == "iNaturalist":
        train_dataset = datasets.INaturalist(
            train_root, version="2021_train", download=True, transform=transform
        )
        test_dataset = datasets.INaturalist(
            test_root, version="2021_valid", download=True, transform=transform
        )
        num_classes = 8142

    elif dataset_name == "Food101":
        train_dataset = datasets.Food101(
            train_root, split="train", download=True, transform=transform
        )
        test_dataset = datasets.Food101(
            test_root, split="test", download=True, transform=transform
        )
        num_classes = 101

    elif dataset_name == "Oxford_IIIT_Pet":
        train_dataset = datasets.OxfordIIITPet(
            train_root, split="trainval", download=True, transform=transform
        )
        test_dataset = datasets.OxfordIIITPet(
            test_root, split="test", download=True, transform=transform
        )
        num_classes = 37

    elif dataset_name == "Fashion_MNIST":
        train_dataset = datasets.FashionMNIST(
            train_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            test_root, train=False, download=True, transform=transform
        )
        num_classes = 10

    elif dataset_name == "FER2013":
        dataset_path = "./data/FER2013"

        # Download and prepare the FER2013 dataset
        download_FER2013_dataset(dataset_path)

        train_dataset = ImageFolder(
            root=os.path.join(dataset_path, "train"), transform=transform
        )
        test_dataset = ImageFolder(
            root=os.path.join(dataset_path, "test"), transform=transform
        )
        num_classes = 7

    else:
        raise ValueError("Invalid dataset name.")

    # Define the model
    if not weights_only:
        return ckpt, num_classes, train_dataset, test_dataset

    if model_name == "ResNet50":
        model = timm.create_model(
            "resnet50", pretrained=False, num_classes=num_classes
        )
        state_dict = {
            k.replace("module.", ""): v for k, v in ckpt["model"].items()
        }

    elif model_name == "DenseNet121":
        model = timm.create_model(
            "densenet121", pretrained=False, num_classes=num_classes
        )
        state_dict = {
            k.replace("module.", ""): v for k, v in ckpt["model"].items()
        }

    elif model_name == "EfficientNetB4":
        model = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=num_classes
        )
        state_dict = {
            k.replace("module.", ""): v for k, v in ckpt["model"].items()
        }

    elif model_name == "ViTBasePatch16":
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=num_classes
        )

    elif model_name == "MobileNetV3L":
        model = timm.create_model(
            "mobilenetv3_large_100", pretrained=False, num_classes=num_classes
        )
        state_dict = {
            k.replace("module.", ""): v for k, v in ckpt["model"].items()
        }

    elif model_name == "ConvNextSmall":
        model = timm.create_model(
            "convnext_small", pretrained=False, num_classes=num_classes
        )
        state_dict = {
            k.replace("module.", ""): v for k, v in ckpt["model"].items()
        }

    elif model_name == "InceptionV3":
        model = timm.create_model(
            "inception_v3", pretrained=False, num_classes=num_classes
        )
        state_dict = {
            k.replace("module.", ""): v for k, v in ckpt["model"].items()
        }

    elif model_name == "EfficientNetV2S":
        model = timm.create_model(
            "efficientnetv2_s", pretrained=False, num_classes=num_classes
        )
        state_dict = {
            k.replace("module.", ""): v for k, v in ckpt["model"].items()
        }

    elif model_name == "VGG19BN":
        model = timm.create_model(
            "vgg19_bn", pretrained=False, num_classes=num_classes
        )
        state_dict = {
            k.replace("module.", ""): v for k, v in ckpt["model"].items()
        }

    elif model_name == "ShuffleNetV2x2_0":
        model = timm.create_model(
            "shufflenet_v2_x2_0", pretrained=False, num_classes=num_classes
        )
        state_dict = {
            k.replace("module.", ""): v for k, v in ckpt["model"].items()
        }

    else:
        raise ValueError(f"Invalid model_name: {model_name}")

    # Load weights into the model
    model.load_state_dict(state_dict)

    return model, num_classes, train_dataset, test_dataset
