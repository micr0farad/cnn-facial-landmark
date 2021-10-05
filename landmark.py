"""
Convolutional Neural Network for facial landmarks detection.
"""
import argparse
from typing import Iterator

import cv2
import matplotlib.pyplot as plt
import numpy
import torch
from torch import optim
from torch.utils.data import DataLoader

from IterableTFRecordDataset import IterableTFRecordDataset
from model import get_landmark_model

IMAGE = "image/encoded"
MARKS = "label/marks"
INPUT_SHAPE = (128, 128, 3)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_record', default='train.record', type=str, help='Training record file')
    parser.add_argument('--train_index', default='train.index', type=str, help='Training record index file')
    parser.add_argument('--epochs', default=1, type=int, help='epochs for training')
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
    return parser.parse_args()


def decode_image(features: dict):
    features[IMAGE] = numpy.reshape(cv2.imdecode(features[IMAGE], -1), INPUT_SHAPE[::-1])
    return features


def get_data_loader(dataset: torch.utils.data.Dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size)


def get_loss_function() -> torch.nn.Module:
    return torch.nn.MSELoss()


def get_optimizer(parameters: Iterator[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
    return optim.Adam(params=parameters, lr=lr)


def split_dataset(dataset, validation_size):
    return torch.utils.data.random_split(dataset, (len(dataset) - validation_size, validation_size))


def main(train_record: str, train_index: str, batch_size: int, lr: float, output_size: int, validation_size: int):
    train_set, val_set = split_dataset(
        IterableTFRecordDataset(
            train_record,
            train_index,
            description={IMAGE: "byte", MARKS: "byte"},
            shuffle_queue_size=1024,
            transform=decode_image
        ),
        validation_size
    )
    model: torch.nn.Module = get_landmark_model(output_size=output_size)
    criterion: torch.nn.Module = get_loss_function()
    optimizer: torch.optim.Optimizer = get_optimizer(model.parameters(), lr)
    train_data_loader = get_data_loader(train_set, batch_size)
    val_data_loader = get_data_loader(val_set, batch_size)
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0

        model.train()
        for data in iter(train_data_loader):
            optimizer.zero_grad()
            outputs = model.forward(data[IMAGE].float().cuda())
            targets = data[MARKS].float().cuda()
            loss = criterion.forward(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (outputs == targets).sum().item()

        with torch.no_grad():
            model.eval()
            for data in iter(val_data_loader):
                optimizer.zero_grad()
                outputs = model.forward(data[IMAGE].float().cuda())
                targets = data[MARKS].float().cuda()
                loss = criterion.forward(outputs, targets)
                val_loss += loss.item()
                val_acc += (outputs == targets).sum().item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"""Epoch {epoch}:
            Train: loss = {train_loss} acc = {train_acc}
            Validation: loss = {val_loss} acc = {val_acc}
            """
        )

    plt.plot(numpy.array(train_losses))
    plt.plot(numpy.array(val_losses))
    plt.show()


if __name__ == '__main__':
    OUTPUT_SIZE = 1583
    LEARNING_RATE = 0.001
    VALIDATION_SIZE = 500
    args = get_args()
    main(
        args.train_record,
        args.train_index,
        args.batch_size,
        LEARNING_RATE,
        OUTPUT_SIZE,
        VALIDATION_SIZE
    )
