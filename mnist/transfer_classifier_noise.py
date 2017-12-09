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
from mnist_logreg_torch import LogReg
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
        'n_iterations': 1000,
        'step_size': 0.1
    }
    # Load our trained models
    model = Net()
    model.load_state_dict(torch.load('mnist_pytorch_R_normalized.pt'))

    log_model = LogReg()
    log_model.load_state_dict(torch.load('mnist_logreg_pytorch.pt'))

    # MNIST data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)

    n_images = 50
    #
    # # Run non targeted
    # img, label = next(iter(test_loader))
    #
    # log_to_cnn = 0
    # cnn_to_log = 0
    # log_fooled = 0
    # cnn_fooled = 0
    # iter_count = 0
    # str_values = ['model_label', 'cnn_adv_label', 'log_adv_label', 'log_to_cnn_label', 'cnn_to_log_label']
    # results = pd.DataFrame(np.zeros((len(str_values), len(str_values))))
    # for img, label in test_loader:
    #     iter_count += 1
    #     print(iter_count)
    #     if iter_count > n_images:
    #         break
    #
    #     # Model prediction on the image
    #     model_label = get_label(img, model)
    #
    #     # Model prediction on the noisy image
    #     cnn_adv_img, cnn_noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
    #     cnn_adv_label = get_label(cnn_adv_img, model)
    #
    #     # Model prediction with Log Reg noise
    #     img = img.view(-1, 784)
    #     log_adv_img, log_noise, log_losses = run_non_targeted_attack(image=img, model=log_model, **kwargs)
    #     log_adv_label = get_label(log_adv_img, log_model)
    #
    #     # Transfer log noise to cnn
    #     log_adv_img = log_adv_img.view(1, 1, 28, 28)
    #     log_to_cnn_label = get_label(log_adv_img, model)
    #
    #     if cnn_adv_label == log_to_cnn_label:
    #         log_to_cnn += 1
    #
    #     if model_label != log_to_cnn_label:
    #         log_fooled += 1
    #
    #     # Transfer cnn noise to log
    #     cnn_adv_img = cnn_adv_img.view(-1, 784)
    #     cnn_to_log_label = get_label(cnn_adv_img, log_model)
    #
    #     if log_adv_label == cnn_to_log_label:
    #         cnn_to_log += 1
    #
    #     if model_label != cnn_to_log_label:
    #         cnn_fooled += 1
    #
    #     values = [model_label, cnn_adv_label, log_adv_label, log_to_cnn_label, cnn_to_log_label]
    #     for i, value1 in enumerate(values):
    #         for j, value2 in enumerate(values):
    #             results.ix[i, j] += 1.0 * (value1 == value2)
    #
    # results.columns = str_values
    # results.index = str_values
    # results = results / n_images
    # results.to_csv('non_targeted_transfer_noise.csv')
    #
    # print("CNN gives same answer with Log Noise:", 1.0 * log_to_cnn / iter_count)
    # print("CNN fooled with Log Noise:", 1.0 * log_fooled / iter_count)
    # print("Log gives same answer with CNN Noise:", 1.0 * cnn_to_log / iter_count)
    # print("Log fooled with CNN Noise:", 1.0 * cnn_fooled / iter_count)
    #
    #
    # # Run targeted
    # img, label = next(iter(test_loader))
    # log_to_cnn = 0
    # cnn_to_log = 0
    # log_fooled = 0
    # cnn_fooled = 0
    # iter_count = 0
    # str_values = ['target_label', 'cnn_adv_label', 'log_adv_label', 'log_to_cnn_label', 'cnn_to_log_label']
    # results = pd.DataFrame(np.zeros((len(str_values), len(str_values))))
    # for img, label in test_loader:
    #     iter_count += 1
    #     print(iter_count)
    #     if iter_count > n_images:
    #         break
    #
    #     # Model prediction on the image
    #     model_label = get_label(img, model)
    #
    #     # Pick a target label
    #     possible_labels = [x for x in range(10) if x != model_label]
    #     target_label = choice(possible_labels)
    #
    #     # Model prediction on the noisy image
    #     cnn_adv_img, cnn_noise, losses = run_targeted_attack(image=img, label=target_label, model=model, **kwargs)
    #     cnn_adv_label = get_label(cnn_adv_img, model)
    #
    #     # Model prediction with Log Reg noise
    #     img = img.view(-1, 784)
    #     log_adv_img, log_noise, log_losses = run_targeted_attack(image=img, label=target_label,model=log_model, **kwargs)
    #     log_adv_label = get_label(log_adv_img, log_model)
    #
    #     # Transfer log noise to cnn
    #     log_adv_img = log_adv_img.view(1, 1, 28, 28)
    #     log_to_cnn_label = get_label(log_adv_img, model)
    #
    #     if cnn_adv_label == log_to_cnn_label:
    #         log_to_cnn += 1
    #
    #     if target_label == log_to_cnn_label:
    #         log_fooled += 1
    #
    #     # Transfer cnn noise to log
    #     cnn_adv_img = cnn_adv_img.view(-1, 784)
    #     cnn_to_log_label = get_label(cnn_adv_img, log_model)
    #
    #     if log_adv_label == cnn_to_log_label:
    #         cnn_to_log += 1
    #
    #     if target_label == cnn_to_log_label:
    #         cnn_fooled += 1
    #
    #     values = [target_label, cnn_adv_label, log_adv_label, log_to_cnn_label, cnn_to_log_label]
    #     for i, value1 in enumerate(values):
    #         for j, value2 in enumerate(values):
    #             results.ix[i, j] += 1.0 * (value1 == value2)
    #
    # results.columns = str_values
    # results.index = str_values
    # results = results / n_images
    # results.to_csv('targeted_transfer_noise.csv')
    #
    # print("CNN gives same answer with Log Noise:", 1.0 * log_to_cnn / iter_count)
    # print("CNN fooled with Log Noise:", 1.0 * log_fooled / iter_count)
    # print("Log gives same answer with CNN Noise:", 1.0 * cnn_to_log / iter_count)
    # print("Log fooled with CNN Noise:", 1.0 * cnn_fooled / iter_count)


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