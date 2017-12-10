# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import os
import sys
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
# noinspection PyPackageRequirements
from mnist_basenet_torch import BaseNet
from random import choice
import pandas as pd
from mnist_logreg_torch import LogReg


import matplotlib.pyplot as plt

# Add src path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fgsm import get_label, run_targeted_attack, run_non_targeted_attack, image_to_tensor, tensor_to_image, \
    draw_result, \
    random_noise, \
    get_label

if __name__ == '__main__':
    # Parameters
    kwargs = {
        'eps': 40 / 225.,
        'n_iterations': 50,
        'step_size': 0.05
    }

    # Load our trained model
    model = BaseNet()
    model.load_state_dict(torch.load('saved_models/mnist_basenet_training_normalized.pt'))

    # MNIST data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)

    # Accuracy

    # Run non targeted
    n_images = 100
    img, label = next(iter(test_loader))
    count = 0
    iter_count = 0
    images_per_label = dict()
    count_per_label = dict()
    for i in range(10):
        images_per_label[i] = 0
        count_per_label[i] = 0

    for img, label in test_loader:
        iter_count += 1
        if iter_count > n_images:
            break
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        adversarial_label = get_label(adv_img, model)
        predicted_label = get_label(img, model)

        images_per_label[label.numpy()[0]] += 1

        if adversarial_label == predicted_label:
            count += 1
            count_per_label[label.numpy()[0]] += 1

    print("Mean accuracy noisy: ", 1.0 * count / n_images)
    print("Image count per label", images_per_label)
    print("Accurate classification per label", count_per_label)

    accuracy_per_label = dict()
    for i in range(10):
        accuracy_per_label[i] = 1.0 * count_per_label[i] / images_per_label[i]
    print("Accuracy per label", accuracy_per_label)

    # Run targeted
    n_images = 100
    img, label = next(iter(test_loader))

    transfer_from_ = [next(iter(test_loader)) for _ in range(1000)]
    transfer_from = dict()
    for i in range(10):
        transfer_from[i] = [x for x in transfer_from_ if x[1].numpy()[0] == i][:n_images]

    results = pd.DataFrame(np.zeros((10, 10)))

    for target_label in range(10):
        for label_from in transfer_from:
            print("Label to {}, Label from {}".format(target_label, label_from))
            if target_label != label_from:
                for img, _ in transfer_from[label_from]:
                    # Model prediction on the noisy image
                    adv_img, noise_from, losses = run_targeted_attack(image=img, label=target_label, model=model, **kwargs)
                    adv_label = get_label(adv_img, model)

                    results.ix[target_label, label_from] += 1.0 * (target_label == adv_label)

    results = results / n_images
    results.to_csv('out/accuracy_targeted_attack.csv')

    model = LogReg()
    model.load_state_dict(torch.load('saved_models/mnist_logreg_pytorch.pt'))

    # Accuracy

    # Run non targeted
    n_images = 500
    img, label = next(iter(test_loader))
    count = 0
    iter_count = 0
    images_per_label = dict()
    count_per_label = dict()
    for i in range(10):
        images_per_label[i] = 0
        count_per_label[i] = 0

    for img, label in test_loader:
        iter_count += 1
        if iter_count > n_images:
            break
        img = img.view(-1, 784)
        predicted_label = get_label(img, model)
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        adv_img = adv_img.view(-1, 784)
        adversarial_label = get_label(adv_img, model)

        images_per_label[label.numpy()[0]] += 1

        if adversarial_label == predicted_label:
            count += 1
            count_per_label[label.numpy()[0]] += 1

    print("Mean accuracy noisy: ", 1.0 * count / n_images)
    print("Image count per label", images_per_label)
    print("Accurate classification per label", count_per_label)

    accuracy_per_label = dict()
    for i in range(10):
        accuracy_per_label[i] = 1.0 * count_per_label[i] / images_per_label[i]
    print("Accuracy per label", accuracy_per_label)

    # Run targeted
    n_images = 100
    img, label = next(iter(test_loader))

    transfer_from_ = [next(iter(test_loader)) for _ in range(10000)]
    transfer_from = dict()
    for i in range(10):
        transfer_from[i] = [x for x in transfer_from_ if x[1].numpy()[0] == i][:n_images]

    results = pd.DataFrame(np.zeros((10, 10)))

    for target_label in range(10):
        for label_from in transfer_from:
            count = 0
            print("Label to {}, Label from {}".format(target_label, label_from))
            if target_label != label_from:
                for img, _ in transfer_from[label_from]:
                    count += 1
                    # Model prediction on the noisy image
                    img = img.view(-1, 784)
                    adv_img, noise_from, losses = run_targeted_attack(image=img, label=target_label, model=model,
                                                                      **kwargs)
                    adv_img = adv_img.view(-1, 784)
                    adv_label = get_label(adv_img, model)

                    results.ix[target_label, label_from] += 1.0 * (target_label == adv_label)
                results.ix[target_label, label_from] *= 1 / count

    results.to_csv('out/lr_accuracy_targeted_attack.csv')

