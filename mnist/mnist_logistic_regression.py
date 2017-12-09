# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
from __future__ import print_function
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=100, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=100, shuffle=True)


def build_set(train=True):
    # Train
    data_set = []
    labels_set = []

    if train:
        loader = train_loader
    else:
        loader = test_loader

    for data, label in loader:
        data_set.append(data.numpy()[:, 0, :, :].reshape(100, -1))
        labels_set.append(label.numpy())

    X = np.array(data_set)
    X = X.reshape(-1, 784)
    y = np.array(labels_set)
    y = y.reshape(-1, 1)
    return X[::10], y[::10]


def train():
    logistic_reg = LogisticRegression()
    X, y = build_set(train=True)
    logistic_reg.fit(X=X, y=y)

    # Save the model
    with open(save_path, 'wb') as f:
        pickle.dump(logistic_reg, f)


def test():
    with open(save_path, 'rb') as f:
        model = pickle.load(f)
    X, y = build_set(train=False)
    print(model.score(X, y))


def load_model():
    with open(save_path, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    # train()
    build_set(train=True)
    test()
    print('Model trained and saved at {}'.format(save_path))
