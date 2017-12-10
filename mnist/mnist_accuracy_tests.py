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
from mnist_torch import Net
from random import choice

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
        'step_size': 0.1
    }
    # Load our trained model
    model = Net()
    model.load_state_dict(torch.load('mnist_pytorch_R_normalized.pt'))

    # MNIST data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)

    # Accuracy

    # Run non targeted
    # n_images = 10000
    # img, label = next(iter(test_loader))
    # count = 0
    # iter_count = 0
    # images_per_label = dict()
    # count_per_label = dict()
    # for i in range(10):
    #     images_per_label[i] = 0
    #     count_per_label[i] = 0
    #
    # for img, label in test_loader:
    #     iter_count += 1
    #     if iter_count > n_images:
    #         break
    #     adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
    #     adversarial_label = get_label(adv_img, model)
    #     predicted_label = get_label(img, model)
    #
    #     images_per_label[label.numpy()[0]] += 1
    #
    #     if adversarial_label == predicted_label:
    #         count += 1
    #         count_per_label[label.numpy()[0]] += 1
    #
    # print("Mean accuracy noisy: ", 1.0 * count / n_images)
    # print("Image count per label", images_per_label)
    # print("Accurate classification per label", count_per_label)
    #
    # accuracy_per_label = dict()
    # for i in range(10):
    #     accuracy_per_label[i] = 1.0 * count_per_label[i] / images_per_label[i]
    # print("Accuracy per label", accuracy_per_label)


    # Run targeted
    n_images = 10000
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

        # Pick a target label
        possible_labels = [x for x in range(10) if x != label.numpy()[0]]
        target_label = choice(possible_labels)

        adv_img, noise, losses = run_targeted_attack(image=img, label=target_label, model=model, **kwargs)
        adversarial_label = get_label(adv_img, model)

        images_per_label[label.numpy()[0]] += 1

        if adversarial_label == target_label:
            count += 1
            count_per_label[label.numpy()[0]] += 1

    print("Mean accuracy noisy: ", 1.0 * count / n_images)
    print("Image count per label", images_per_label)
    print("Accurate classification per label", count_per_label)

    accuracy_per_label = dict()
    for i in range(10):
        accuracy_per_label[i] = 1.0 * count_per_label[i] / images_per_label[i]
    print("Accuracy per label", accuracy_per_label)