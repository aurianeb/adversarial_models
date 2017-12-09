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

import matplotlib.pyplot as plt

# Add src path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fgsm import get_label, run_targeted_attack, run_non_targeted_attack, image_to_tensor, tensor_to_image, \
    draw_result, \
    random_noise, \
    run_non_targeted_attack_v2

if __name__ == '__main__':
    # Parameters
    kwargs = {
        'eps': 100 / 225.,
        'n_iterations': 100,
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
    #
    # final_losses = []
    # eps_range = np.arange(1, 100, 10) / 225.
    # step_size_range = [0.1, 0.05, 0.001, 0.0005, 0.00001]
    # res = np.zeros(shape=(len(eps_range), len(step_size_range)))
    # n_images = 10
    #
    # for i, eps in enumerate(eps_range):
    #     # eps_losses = []
    #     for j, step_size in enumerate(step_size_range):
    #         kwargs = {
    #             'eps': eps,
    #             'n_iterations': 100,
    #             'step_size': step_size
    #         }
    #         hparam = 'eps={},step_size={},n_iterations=100'.format(eps, step_size)
    #         print(hparam)
    #
    #         # Run non targeted
    #         img, label = next(iter(test_loader))
    #         count = 0
    #         iter_count = 0
    #         for img, label in test_loader:
    #             iter_count += 1
    #             if iter_count > n_images:
    #                 break
    #             adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
    #             adversarial_label = get_label(adv_img, model)
    #             predicted_label = get_label(img, model)
    #             if adversarial_label != predicted_label:
    #                 count += 1
    #
    #                 # fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=model)
    #                 # plt.savefig('out/{}.png'.format(hparam))
    #
    #                 # Save associated loss
    #                 # plt.plot(losses)
    #                 # plt.title('Cross Entropy Loss\norig_label={},adversarial_label={}'.format(orig_label, adversarial_label))
    #                 # plt.ylabel('loss')
    #                 # plt.xlabel('n_iterations')
    #                 # plt.savefig('out/loss_{}.png'.format(hparam))
    #                 # plt.close()
    #         res[i, j] = count
    #
    #         final_loss = losses[-1]
    #     final_losses.append(final_loss)
    #
    # # Losses after convergence
    # for i, eps in enumerate(eps_range):
    #     plt.plot(final_losses[i], step_size_range, label=eps)
    # plt.title('Cross Entropy')
    # plt.ylabel('loss')
    # plt.xlabel('step_size')
    # plt.legend()
    # plt.savefig('out/loss_function_of_step_{}.png'.format(eps))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_title('Percentage of successful adversarial images')
    # plt.imshow(res / n_images, cmap='hot')
    #
    # # ax.set_xticks(step_size_range)
    # # plt.xticks([itstep_size_range])
    # # plt.gca().set_xticks(step_size_range)
    # ax.set_xlabel('step size')
    #
    # # ax.set_yticks(eps_range)
    # ax.set_ylabel('max noise')
    #
    # plt.colorbar()
    # plt.savefig('successful_adv_images.png')

    eps_range = np.arange(1, 100, 10) / 225.
    step_size_range = [0.1, 0.05, 0.001, 0.0005, 0.00001]

    # Run non targeted v2
    img, label = next(iter(test_loader))
    n_images = 10
    avg_losses = [0 for i in range(kwargs['n_iterations'])]
    avg_losses_v2 = [0 for i in range(kwargs['n_iterations'])]
    iter_count = 0
    for img, label in test_loader:
        iter_count += 1
        if iter_count > n_images:
            break
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        adversarial_label = get_label(adv_img, model)
        adv_img_v2, noise_v2, losses_v2 = run_non_targeted_attack_v2(image=img, model=model, **kwargs)
        adversarial_label_v2 = get_label(adv_img_v2, model)

        avg_losses = [sum(x) for x in zip(avg_losses, losses)]
        avg_losses_v2 = [sum(x) for x in zip(avg_losses_v2, losses_v2)]

    plt.plot([i for i in range(kwargs['n_iterations'])], avg_losses, label='v1')
    plt.plot([i for i in range(kwargs['n_iterations'])], avg_losses_v2, label='v2')
    plt.title('Losses of v1 and v2 vs number of iterations')
    plt.xlabel('n_iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('out/targeted_v1_v2.png')