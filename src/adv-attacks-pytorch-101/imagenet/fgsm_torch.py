# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import logging

import matplotlib
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision.models.inception import inception_v3

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def load_image(img_path):
    img = trans(Image.open(img_path).convert('RGB'))
    return img


def get_class(img):
    x = Variable(img, volatile=True)
    cls = model(x).data.max(1)[1].cpu().numpy()[0]
    return classes[cls]


def non_targeted_attack(image, model):
    """
    Performs a non-targeted attack vs a given model
    Args:
        model ():
        image ():

    Returns:

    """
    image = image
    label = torch.zeros(1, 1)

    x, y = Variable(image, requires_grad=True), Variable(label)

    for step in range(steps):
        # Reset the gradients
        zero_gradients(x)
        # Forward
        out = model(x)
        y.data = out.data.max(1)[1]
        _loss = loss(out, y)
        logging.warning('{}: {}'.format(step, _loss.data))
        # Backprop
        _loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)
        step_adv = x.data + normed_grad
        adv = step_adv - image
        adv = torch.clamp(adv, -eps, eps)
        result = image + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result.cpu(), adv.cpu()


def targeted_attack(img, label):
    img = img
    label = torch.Tensor([label]).long()

    x, y = Variable(img, requires_grad=True), Variable(label)
    for step in range(steps):
        zero_gradients(x)
        out = model(x)
        _loss = loss(out, y)
        _loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)
        step_adv = x.data - normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps, eps)
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result.cpu(), adv.cpu()


def draw_result(img, noise, adv_img):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    orig_class, attack_class = get_class(img), get_class(adv_img)
    ax[0].imshow(reverse_trans(img[0]))
    ax[0].set_title('Original image: {}'.format(orig_class.split(',')[0]))
    ax[1].imshow(noise[0].cpu().numpy().transpose(1, 2, 0))
    ax[1].set_title('Attacking noise')
    ax[2].imshow(reverse_trans(adv_img[0]))
    ax[2].set_title('Adversarial example: {}'.format(attack_class))
    for i in range(3):
        ax[i].set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    with open('classes.txt') as f:
        classes = eval(f.read())
    # Turns to tensor and ravels
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda t: t.unsqueeze(0))])
    # Reverse to PIL image
    reverse_trans = lambda x: np.asarray(transforms.ToPILImage()(x))

    eps = 2 * 8 / 225.
    steps = 40
    norm = float('inf')
    step_alpha = 0.01

    model = inception_v3(pretrained=True, transform_input=True)
    loss = nn.CrossEntropyLoss()
    model.eval()

    img = load_image('input.png')

    # Non-targeted
    adv_img, noise = non_targeted_attack(image=img, model=model)

    # Targeted
    adv_img, noise = targeted_attack(img, 859)
    draw_result(img, noise, adv_img)
