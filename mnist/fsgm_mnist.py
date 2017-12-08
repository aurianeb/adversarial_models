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
from imagenet.fgsm_imagenet import run_targeted_attack, draw_result, run_non_targeted_attack
# noinspection PyPackageRequirements
from mnist.mnist_torch import Net

import matplotlib.pyplot as plt

# Add src path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fgsm import run_targeted_attack, run_non_targeted_attack, image_to_tensor, tensor_to_image, draw_result


if __name__ == '__main__':
    # Parameters
    kwargs = {
        'eps': 100 / 225.,
        'n_iterations': 1000,
        'step_size': 0.01
    }
    # Load our trained model
    model = Net()
    model.load_state_dict(torch.load('mnist_pytorch.pt'))

    # MNIST data loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)

    # Get one MNIST image
    data, target = next(iter(test_loader))
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    pred = output.data.max(1)[1]

    img, label = next(iter(test_loader))
    adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
    fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=model)
    plt.show()

    plt.plot(losses)
    plt.title('Cross Entropy Loss\norig_label={},adversarial_label={}'.format(orig_label, adversarial_label))
    plt.ylabel('loss')
    plt.xlabel('n_iterations')
    plt.show()
