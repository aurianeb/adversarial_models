# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import logging

import matplotlib
import torch
import torchvision.transforms as transforms
import torchvision.datasets

from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision.models.inception import inception_v3
from torchvision.models.squeezenet import squeezenet1_1

from PIL import Image

from sklearn.linear_model import LogisticRegression

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def load_image(img_path):
    img = trans(Image.open(img_path).convert('RGB'))
    return img


def get_class(image, model):
    x = Variable(image, volatile=True)
    cls = model(x).data.max(1)[1].numpy()[0]
    return classes.get(cls) or cls


def non_targeted_attack(alpha, image, model):
    """
    Performs a non-targeted attack vs a given model_
    Args:
        alpha ():
        model ():
        image ():

    Returns:

    """
    label = torch.zeros(1, 1)

    x, y = Variable(image, requires_grad=True), Variable(label)

    for step in range(steps):
        # Reset the gradients
        zero_gradients(x)
        # Forward
        out = model(x)
        y.data = out.data.max(1)[1]
        _loss = loss(out, y)
        logging.warning('{}: {}'.format(step, _loss.data[0]))
        # Backprop
        _loss.backward()
        # Fixed norm steps not to stay trapped around a local minima
        normed_grad = alpha * torch.sign(x.grad.data)
        step_adv = x.data + normed_grad
        adv = step_adv - image
        adv = torch.clamp(adv, -eps, eps)
        result = image + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result, adv


def targeted_attack(alpha, image, label):
    """

    Args:
        alpha ():
        image ():
        label ():

    Returns:

    """
    label = torch.Tensor([label]).long()
    _losses = []
    x, y = Variable(image, requires_grad=True), Variable(label)
    for step in range(steps):
        zero_gradients(x)
        out = inception_model(x)
        _loss = loss(out, y)
        _loss.backward()
        _losses.append(_loss.data[0])
        # Why normalizing here?
        normed_grad = alpha * torch.sign(x.grad.data)
        # normed_grad = step_alpha * x.grad.data
        step_adv = x.data - normed_grad
        adv = step_adv - image
        adv = torch.clamp(adv, -eps, eps)
        result = image + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result, adv, _losses


def draw_result(img, noise, adv_img):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    orig_class, attack_class = get_class(img), get_class(adv_img)
    ax[0].imshow(reverse_trans(img[0]))
    ax[0].set_title('Original image: {}'.format(orig_class.split(',')[0]))
    ax[1].imshow(noise[0].numpy().transpose(1, 2, 0))
    ax[1].set_title('Attacking noise')
    ax[2].imshow(reverse_trans(adv_img[0]))
    ax[2].set_title('Adversarial example: {}'.format(attack_class))
    for i in range(3):
        ax[i].set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load the ImageNet classes
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
    alpha = 0.01

    inception_model = inception_v3(pretrained=True, transform_input=True)
    squeezenet_model = squeezenet1_1(pretrained=True)
    loss = nn.CrossEntropyLoss()
    # Instantiate the model in some way
    inception_model.eval()
    squeezenet_model.eval()

    img = load_image('images/ac2a8a4777dbe746.png')
    img = load_image('sport_car.png')

    get_class(img, inception_model)
    get_class(img, squeezenet_model)
    # Non-targeted
    # adv_img, noise = non_targeted_attack(image=img, model_=model_)

    # Targeted
    adv_img, noise, losses = targeted_attack(alpha=alpha, image=img, label=823)
    draw_result(img, noise, adv_img)

    # Add the noise to another image
    innocent_img = load_image('sport_car.png')
    corrupted_img = innocent_img + noise
    print('innocent_img = {}'.format(get_class(innocent_img)))
    print('corrupted_img = {}'.format(get_class(innocent_img)))
    # Ne fonctionne pas, inception finalement assez robuste meme a un fort noise
    draw_result(innocent_img, noise, corrupted_img)

    # alphas_ = [0.1, 0.05, 0.01, 0.005, 0.001]
    # all_losses = []
    # for alpha_ in alphas_:
    #     print('alpha = {}'.format(alpha))
    #     adv_img, noise, losses = targeted_attack(alpha=alpha_, image=img, label=823)
    #     all_losses.append(losses)
    #
    # plt.figure()
    # for i, loss in enumerate(all_losses):
    #     plt.plot(loss, label='alpha = {}'.format(alphas_[i]))
    # plt.xlabel('steps')
    # plt.legend()
    # plt.show()

    print('over')
    # Print loss en fonction de alpha step
    # non-normalizer le gradient nous empeche de sortir du minimum apparemment
