# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import os
import sys
import torch
from torchvision import datasets, transforms
import numpy as np
# noinspection PyUnresolvedReferences
from mnist_basenet_torch import BaseNet

import matplotlib.pyplot as plt

# Add src path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fgsm import plot_heatmap_results, get_label, run_targeted_attack, run_non_targeted_attack, image_to_tensor, \
    tensor_to_image, \
    draw_result, \
    random_noise, \
    run_non_targeted_attack_v2


def plot_bunch_result():
    for img, label in test_loader:
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=model)
        plt.savefig('out/non_targeted/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
        plt.close()


if __name__ == '__main__':
    # Parameters
    kwargs = {
        'eps': 2 / 225.,
        'n_iterations': 40,
        'step_size': 0.1
    }
    # Load our trained model
    model = BaseNet()
    model.load_state_dict(torch.load('saved_models/mnist_basenet_training_normalized.pt'))

    # MNIST data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)

    # Experiments
    plot_bunch_result()

    print('done')
