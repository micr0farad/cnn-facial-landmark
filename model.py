"""This module provides the function to build the network."""
import torch
from torch.nn import Sequential


def build_landmark_model(output_size) -> Sequential:
    """Build the convolutional network model with Keras Functional API.

    Args:
        output_size: the number of output node, usually equals to the number of marks times 2 (in 2d space).

    Returns:
        a model
    """
    return torch.nn.Sequential(
        # |== Layer 1 ==|
        torch.nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3),
        ),
        torch.nn.ReLU(),
        torch.nn.LazyBatchNorm2d(),
        torch.nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        ),
        # |== Layer 2 ==|
        torch.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            stride=(1, 1),
            padding=(0, 0),
            kernel_size=(3, 3),
        ),
        torch.nn.ReLU(),
        torch.nn.LazyBatchNorm2d(),
        torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            stride=(1, 1),
            padding=(0, 0),
            kernel_size=(3, 3),
        ),
        torch.nn.ReLU(),
        torch.nn.LazyBatchNorm2d(),
        torch.nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        ),
        # |== Layer 3 ==|
        torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            stride=(1, 1),
            padding=(0, 0),
            kernel_size=(3, 3),
        ),
        torch.nn.ReLU(),
        torch.nn.LazyBatchNorm2d(),
        torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            stride=(1, 1),
            padding=(0, 0),
            kernel_size=(3, 3),
        ),
        torch.nn.ReLU(),
        torch.nn.LazyBatchNorm2d(),
        torch.nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        ),
        # |== Layer 4 ==|
        torch.nn.Conv2d(
            in_channels=64,
            out_channels=128,
            stride=(1, 1),
            padding=(0, 0),
            kernel_size=(3, 3),
        ),
        torch.nn.ReLU(),
        torch.nn.LazyBatchNorm2d(),
        torch.nn.Conv2d(
            in_channels=128,
            out_channels=128,
            stride=(1, 1),
            padding=(0, 0),
            kernel_size=(3, 3),
        ),
        torch.nn.ReLU(),
        torch.nn.LazyBatchNorm2d(),
        torch.nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(0, 0)
        ),
        # |== Layer 5 ==|
        torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            stride=(1, 1),
            padding=(0, 0),
            kernel_size=(3, 3),
        ),
        torch.nn.ReLU(),
        torch.nn.LazyBatchNorm2d(),
        # |== Layer 6 ==|
        torch.nn.Flatten(),
        torch.nn.LazyLinear(out_features=1024),
        torch.nn.ReLU(),
        torch.nn.LazyBatchNorm1d(),
        torch.nn.LazyLinear(out_features=output_size)
    )
