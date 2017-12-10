# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import os

import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision.models.inception import inception_v3

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Global
# Load the ImageNet classes
with open('../imagenet/labels.txt') as f:
    labels = eval(f.read())


def random_noise(image, model, eps):
    # Create PyTorch tensor variables
    x = Variable(image, requires_grad=True)

    noise = torch.Tensor(x.data.shape).uniform_(0, eps)
    adversarial_image = x.data + noise
    adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)
    x.data = adversarial_image

    return adversarial_image, noise


def get_label(image, model):
    """
    Returns label for the image as predicted by the model
    Args:
        image ():

    Returns:
        label of the image
    """
    x = Variable(image, volatile=True)
    label = model(x).data.max(1)[1].numpy()[0]
    # We have string labels for ImageNet
    if isinstance(model, torchvision.models.inception.Inception3):
        label_string = labels.get(label)
        return label_string
    return label


def run_non_targeted_attack(step_size, image, model, n_iterations, eps, loss=nn.CrossEntropyLoss()):
    """
    Performs a non-targeted attack against a given model
    Args:
        eps ():
        n_iterations ():
        model ():
        loss ():
        step_size (): gradient ascent step size
        image (): model to fool

    Returns:
        adversarial_image, attacking_noise
    """
    # Here we do not care about the value of the target label
    label = torch.zeros(1, 1)
    # Record our loss values
    losses = []
    # Create PyTorch tensor variables
    x, y = Variable(image, requires_grad=True), Variable(label)
    # Perform our gradient ascent
    for _ in range(n_iterations):
        # Reset the gradients
        zero_gradients(x)
        # Forward propagation
        out = model(x)
        # Our prediction
        y.data = out.data.max(1)[1]
        # Compute our loss
        loss_tensor = loss(out, y)
        # Record our loss
        losses.append(loss_tensor.data[0])
        # Back propagation
        loss_tensor.backward()
        # Fixed norm n_iterations not to stay trapped around a local minima
        normed_grad = step_size * torch.sign(x.grad.data)
        # Perform our gradient ascent step
        step_adv = x.data + normed_grad
        # Compute our adversarial noise
        attacking_noise = step_adv - image
        # Clamp our adversarial noise
        attacking_noise = torch.clamp(attacking_noise, -eps, eps)
        # Compute our adversarial image
        adversarial_image = image + attacking_noise
        # Normalize it to feed it to inception
        adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)
        x.data = adversarial_image
    return adversarial_image, attacking_noise, losses


# noinspection PyUnboundLocalVariable
def run_targeted_attack(image, label, model, step_size, eps, n_iterations, loss=nn.CrossEntropyLoss()):
    """
    Performs a targeted attack against a given model

    Args:
        step_size (): gradient descent step size
        image (): image to attack
        label (): target model

    Returns:
        adversarial_image, attacking_noise
    """
    # PyTorch tensor for target label
    if not isinstance(label, torch.LongTensor):
        label = torch.Tensor([label]).long()
    # Record our loss values
    losses = []
    # Create PyTorch tensor variables
    x, y = Variable(image, requires_grad=True), Variable(label)
    # Perform our gradient descent
    for _ in range(n_iterations):
        # Reset the gradients
        zero_gradients(x)
        # Forward propagation
        out = model(x)
        # Compute our loss
        loss_tensor = loss(out, y)
        # Record our loss
        losses.append(loss_tensor.data[0])
        # Back propagation
        loss_tensor.backward()
        # Fixed norm n_iterations not to stay trapped around a local minima
        normed_grad = step_size * torch.sign(x.grad.data)
        # Perform our gradient descent step
        step_adv = x.data - normed_grad
        # Compute our adversarial noise
        attacking_noise = step_adv - image
        # Clamp our adversarial noise
        attacking_noise = torch.clamp(attacking_noise, -eps, eps)
        # Compute our adversarial image
        adversarial_image = image + attacking_noise
        # Normalize it to feed it to inception
        adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)
        x.data = adversarial_image
    return adversarial_image, attacking_noise, losses


def draw_result(image, attacking_noise, adversarial_image, model):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    # Get labels
    orig_label = get_label(image, model)
    adversarial_label = get_label(adversarial_image, model)

    # Truncate long string labels
    if isinstance(orig_label, str):
        orig_label = orig_label.split(',')[0]

    if isinstance(adversarial_label, str):
        adversarial_label = adversarial_label.split(',')[0]

    ax[0].set_title('Original image: {}'.format(orig_label))
    # Original image
    ax[0].imshow(tensor_to_image(image[0]))

    noise_trans = attacking_noise[0].numpy().transpose(1, 2, 0)
    normalized_noise = (noise_trans - noise_trans.min()) / (noise_trans.max() - noise_trans.min())
    if normalized_noise.shape[2] == 1:
        normalized_noise = normalized_noise[:, :, 0]
    ax[1].imshow(normalized_noise)
    ax[1].set_title('Attacking noise')

    ax[2].imshow(tensor_to_image(adversarial_image[0]))
    ax[2].set_title('Adversarial image: {}'.format(adversarial_label))

    for i in range(3):
        ax[i].set_axis_off()

    plt.tight_layout()
    return fig, orig_label, adversarial_label


def image_to_tensor(image):
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Lambda(lambda t: t.unsqueeze(0))])(image)


def tensor_to_image(tensor):
    return np.asarray(transforms.ToPILImage()(tensor))


def plot_heatmap_results(res, x_range, y_range, x_label, y_label):
    x_range = ['{:.2E}'.format(val) for val in x_range]
    y_range = ['{:.1E}'.format(val) for val in y_range]

    fig = plt.figure(figsize=(18, 9))
    ax = sns.heatmap(res,
                     annot=True,
                     xticklabels=x_range,
                     yticklabels=y_range)
    ax.set_title('Percentage of successful adversarial images')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return fig
