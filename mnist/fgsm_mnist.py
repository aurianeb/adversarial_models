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
# noinspection PyUnresolvedReferences
from mnist_basenet_torch import BaseNet

import matplotlib.pyplot as plt
import seaborn as sns

# Add src path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fgsm import plot_heatmap_results, get_label, run_targeted_attack, run_non_targeted_attack, image_to_tensor, \
    tensor_to_image, \
    draw_result, \
    random_noise

if __name__ == '__main__':
    # Parameters
    # kwargs = {
    #     'eps': 100 / 225.,
    #     'n_iterations': 1000,
    #     'step_size': 0.1
    # }
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

    eps_range = [5 * i / 200 for i in range(1, 11)]
    step_size_range = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    res = np.zeros(shape=(len(eps_range), len(step_size_range)))
    n_images = 20

    for i, eps in enumerate(eps_range):
        for j, step_size in enumerate(step_size_range):
            kwargs = {
                'eps': eps,
                'n_iterations': 100,
                'step_size': step_size
            }
            hparam = 'eps={},step_size={},n_iterations={}'.format(kwargs['eps'],
                                                                  kwargs['step_size'],
                                                                  kwargs['n_iterations'])
            print(hparam)

            # Run non targeted
            count = 0
            iter_count = 0
            for img, label in test_loader:
                iter_count += 1
                if iter_count > n_images:
                    break
                adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
                adversarial_label = get_label(adv_img, model)
                predicted_label = get_label(img, model)
                if adversarial_label != predicted_label:
                    # We fooled the classifier
                    count += 1
            res[i, j] = count
    print(res)
    perc_res = res / n_images
    fig = plot_heatmap_results(res=perc_res,
                               x_range=step_size_range,
                               y_range=eps_range,
                               x_label='step size',
                               y_label='max noise')
    plt.savefig('out//mnist_successful_adv_images_using_not_norm_images_with_norm_model.png')

    # Accuracy

    # Run non targeted
    # n_images = 50
    # img, label = next(iter(test_loader))
    # count = 0
    # iter_count = 0
    # for img, label in test_loader:
    #     iter_count += 1
    #     if iter_count > n_images:
    #         break
    #     adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
    #     adversarial_label = get_label(adv_img, model)
    #     predicted_label = get_label(img, model)
    #     if adversarial_label == predicted_label:
    #         count += 1
    #
    # print("Mean accuracy: ", 1.0 * count / n_images)
