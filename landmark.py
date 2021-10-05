"""
Convolutional Neural Network for facial landmarks detection.
"""
import argparse

import cv2
import numpy
import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch import optim
from torch.utils.data import DataLoader

from model import build_landmark_model
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--train_record', default='train.record', type=str, help='Training record file')
parser.add_argument('--train_index', default='train.index', type=str, help='Training record index file')
parser.add_argument('--val_record', default='validation.record', type=str, help='validation record file')
parser.add_argument('--epochs', default=1, type=int, help='epochs for training')
parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
args = parser.parse_args()

image = "image/encoded"
marks = "label/marks"


def decode_image(features):
    features[image] = numpy.reshape(cv2.imdecode(features[image], -1), (3, 128, 128))
    return features


if __name__ == '__main__':
    dataset = TFRecordDataset(
        args.train_record,
        args.train_index,
        description={image: "byte", marks: "byte"},
        shuffle_queue_size=1024,
        transform=decode_image
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size)
    model: torch.nn.Sequential = build_landmark_model(output_size=1583)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    model.to(torch.device('cuda'))
    loss_vals = []
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(iter(data_loader), 0):
            optimizer.zero_grad()
            loss = criterion(model.forward(data[image].float().cuda()), data[marks].float().cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[{epoch + 1}] loss: {running_loss}")
        loss_vals.append(running_loss)
    plt.plot(numpy.array(loss_vals))
    plt.show()
