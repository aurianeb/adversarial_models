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
# from mnist_logreg_torch import LogReg
from random import choice
import pandas as pd

import matplotlib.pyplot as plt

# Add src path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fgsm import get_label, run_targeted_attack, run_non_targeted_attack, image_to_tensor, tensor_to_image, \
    draw_result, \
    random_noise


if __name__ == '__main__':
    # Parameters
    kwargs = {
        'eps': 100 / 225.,
        'n_iterations': 50,
        'step_size': 0.1
    }
    # Load our trained models
    model = BaseNet()
    model.load_state_dict(torch.load('saved_models/mnist_basenet_training_normalized.pt'))

    # log_model = LogReg()
    # log_model.load_state_dict(torch.load('saved_models/mnist_logreg_pytorch.pt'))

    # MNIST data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)

    n_images = 10

    # Run non targeted
    img, label = next(iter(test_loader))
    transfer_to_ = [next(iter(test_loader)) for _ in range(1000)]
    transfer_to = dict()
    for i in range(10):
        transfer_to[i] = [x for x in transfer_to_ if x[1].numpy()[0] == i][:n_images]
    transfer_from_ = [next(iter(test_loader)) for _ in range(1000)]
    transfer_from = dict()
    for i in range(10):
        transfer_from[i] = [x for x in transfer_from_ if x[1].numpy()[0] == i][:n_images]

    results = pd.DataFrame(np.zeros((10, 10)))

    for label_to in transfer_to:
        for label_from in transfer_from:
            print("Label to {}, Label from {}".format(label_to, label_from))
            for img_to, _ in transfer_to[label_to]:

                # Model prediction on the image
                model_label = get_label(img, model)

                for img_from, _ in transfer_from[label_from]:

                    # Model prediction on the noisy image
                    adv_img, noise_from, losses = run_non_targeted_attack(image=img_from, model=model, **kwargs)
                    adv_label = get_label(img_to + noise_from, model)

                    results.ix[label_to, label_from] += 1.0 * (model_label != adv_label)

    results = results / n_images
    results.to_csv('non_targeted_transfer_image_noise_eps_{}_alpha.csv'.format(
                      kwargs['eps'], kwargs['alpha']))


    # Run targeted
    img, label = next(iter(test_loader))
    transfer_to_ = [next(iter(test_loader)) for _ in range(1000)]
    transfer_to = dict()
    for i in range(10):
        transfer_to[i] = [x for x in transfer_to_ if x[1].numpy()[0] == i][:n_images]
    transfer_from_ = [next(iter(test_loader)) for _ in range(1000)]
    transfer_from = dict()
    for i in range(10):
        transfer_from[i] = [x for x in transfer_from_ if x[1].numpy()[0] == i][:n_images]

    results = pd.DataFrame(np.zeros((10, 10)))

    for label_to in transfer_to: # Image that receives the noise
        different_labels = [k for k in range(10) if k != label_to]
        for target_label in different_labels:
            print("Label to {}, Label from {}".format(label_to, target_label))
            for img_to, _ in transfer_to[label_to]:
                # Model prediction on the image
                model_label = get_label(img, model)

                # Picks a different image with the same label
                img_from, _ = choice(transfer_from[label_to])

                # Model prediction on the noisy image
                adv_img, noise_from, losses = run_targeted_attack(image=img_from, label=target_label, model=model, **kwargs)
                adv_label = get_label(img_to + noise_from, model)

                results.ix[label_to, target_label] += 1.0 * (target_label == adv_label)

    results = results / n_images
    results.to_csv('targeted_transfer_image_noise_eps_{}_alpha.csv'.format(
                  kwargs['eps'], kwargs['step_size']))


# Comparaison with random noise
    img, label = next(iter(test_loader))
    different_label = 0
    fooled = 0
    iter_count = 0
    for img, label in test_loader:
        iter_count += 1
        print(iter_count)
        if iter_count > n_images:
            break

        # Model prediction on the image
        model_label = get_label(img, model)

        # Model prediction on the noisy image
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        adversarial_label = get_label(adv_img, model)

        # Model prediction with random noise
        img = img.view(-1, 784)
        adv_img, noise = random_noise(img, model, kwargs['eps'])
        adv_img = adv_img.view(1, 1, 28, 28)
        random_adversarial_label = get_label(adv_img, model)

        if adversarial_label != model_label:
            different_label += 1

        if model_label != random_adversarial_label:
            fooled += 1

    print(different_label, fooled, iter_count)