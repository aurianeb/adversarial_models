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
from mnist_logistic_regresssion import load_model
from mnist_torch import Net


import matplotlib.pyplot as plt

# Add src path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fgsm import get_label, run_targeted_attack, run_non_targeted_attack, image_to_tensor, tensor_to_image, \
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
    model = Net()
    model.load_state_dict(torch.load('mnist_pytorch_R_normalized.pt'))

    logistic_model = load_model()

    # MNIST data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)

    final_losses = []
    eps_range = np.arange(1, 100, 10) / 225.
    step_size_range = [0.1, 0.05, 0.001, 0.0005, 0.00001]
    res = np.zeros(shape=(len(eps_range), len(step_size_range)))
    n_images = 100

    kwargs = {
        'eps': eps,
        'n_iterations': 100,
        'step_size': step_size
    }
    hparam = 'eps={},step_size={},n_iterations=100'.format(eps, step_size)
    print(hparam)

    # Run non targeted
    img, label = next(iter(test_loader))
    count = 0
    iter_count = 0
    for img, label in test_loader:
        iter_count += 1
        if iter_count > n_images:
            break
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        log_adv_img, log_noise, log_losses = run_non_targeted_attack(image=img, model=logistic_model, **kwargs)
        adversarial_label = get_label(adv_img, model)
        log_adversarial_label = model.predict(tensor_to_image(log_adv_img))
        if adversarial_label != log_adversarial_label:
            count += 1
    print(count, iter_count)

    # final_loss = losses[-1]
            # eps_losses.append(final_loss)

    # Losses after convergence
    for i, eps in enumerate(eps_range):
        plt.plot(final_losses[i], step_size_range, label=eps)
    plt.title('Cross Entropy')
    plt.ylabel('loss')
    plt.xlabel('step_size')
    plt.legend()
    plt.savefig('out/loss_function_of_step_{}.png'.format(eps))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Percentage of successful adversarial images')
    plt.imshow(res / n_images, cmap='hot')

    # ax.set_xticks(step_size_range)
    # plt.xticks([itstep_size_range])
    # plt.gca().set_xticks(step_size_range)
    ax.set_xlabel('step size')

    # ax.set_yticks(eps_range)
    ax.set_ylabel('max noise')

    plt.colorbar()
    plt.savefig('successful_adv_images.png')

    #
    # results = {'count': 0, 'non_target': 0, 'random': 0}
    # max_iter = 100
    # it = 0

    # Accuracy non targeted vs random
    # for img, label in test_loader:
    #     print(it)
    #     x = Variable(img, requires_grad=True)
    #     y_model = model(x)
    #     # Non targeted
    #     adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
    #     x_adv = Variable(adv_img, requires_grad=True)
    #     y_non_target = model(x_adv)
    #     results['non_target'] += y_model == y_non_target
    #     # Random
    #     adv_img, noise = random_noise(img, model, kwargs['eps'])
    #     x_rand = Variable(adv_img, requires_grad=True)
    #     y_random = model(x_rand)
    #     results['random'] += y_model == y_random
    #     it += 1
    #     if it > max_iter:
    #         break

    img, label = next(iter(test_loader))
    count = 0
    iter_count = 0
    for img, label in test_loader:
        iter_count += 1
        if iter_count > 10: break
        # Non targeted
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        adversarial_label = get_label(adv_img, model)
        # Random
        adv_img, noise = random_noise(img, model, kwargs['eps'])
        predicted_label = get_label(img, model)
        if adversarial_label != predicted_label:
            count += 1

    res[i, j] = count

    # final_loss = losses[-1]
    # eps_losses.append(final_loss)
