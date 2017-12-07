# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import os
import logging

import torch
import torchvision.transforms as transforms

from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision.models.inception import inception_v3

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob


def load_image(img_path):
    """
    Loads a PNG image and convert it to a PyTorch tensor image
    Args:
        img_path ():

    Returns:
        PyTorch tensor image
    """
    image = image_to_tensor(Image.open(img_path).convert('RGB'))
    return image


def load_images(input_dir):
    """
    Generates images
    Args:
        input_dir ():

    Returns:

    """
    for image_path in glob.glob(os.path.join(input_dir, '*.png')):
        yield load_image(image_path)


def get_inception_label(image):
    """
    Returns ImageNet label for the image as predicted by the inception model
    Args:
        image ():

    Returns:
        label of the image
    """
    x = Variable(image, volatile=True)
    label = inception_model(x).data.max(1)[1].numpy()[0]
    # Get label name
    label_name = labels[label]
    return label_name.replace(' ', '')


def run_non_targeted_attack(step_size, image):
    """
    Performs a non-targeted attack against a given model
    Args:
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
        out = inception_model(x)
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
def run_targeted_attack(step_size, image, label, model):
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


def draw_result(image, attacking_noise, adversarial_image):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    # Get labels
    orig_label = get_inception_label(image)
    adversarial_label = get_inception_label(adversarial_image)

    ax[0].imshow(tensor_to_image(image[0]))
    ax[0].set_title('Original image: {}'.format(orig_label.split(',')[0]))

    noise_trans = attacking_noise[0].numpy().transpose(1, 2, 0)
    normalized_noise = (noise_trans - noise_trans.min()) / (noise_trans.max() - noise_trans.min())
    ax[1].imshow(normalized_noise)
    ax[1].set_title('Attacking attacking_noise')

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


if __name__ == '__main__':
    # Load the ImageNet classes
    with open('labels.txt') as f:
        labels = eval(f.read())

    # Parameters
    eps = 2 * 8 / 225.
    n_iterations = 40
    step_size = 0.01

    # Model
    inception_model = inception_v3(pretrained=True, transform_input=True)
    loss = nn.CrossEntropyLoss()
    # Instantiate the model
    inception_model.eval()

    # Load our images
    img = load_image('images/sport_car.png')
    images_gen = load_images('images')

    #  for img in images_gen:
    #     # Non-targeted
    #     adv_img, noise, losses = run_non_targeted_attack(step_size=step_size, image=img)
    #     fig, orig_label, adversarial_label = draw_result(img, noise, adv_img)
    #     plt.savefig('out/non_targeted/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    #     plt.close(fig)
    #
    #     plt.plot(losses)
    #     plt.title('Cross Entropy Loss\norig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    #     plt.ylabel('loss')
    #     plt.xlabel('n_iterations')
    #     plt.savefig(
    #         'out/non_targeted/loss_orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    #     plt.close()
    #
    #     # Targeted
    #     adv_img, noise, losses = run_targeted_attack(step_size=step_size, image=img, label=823, model=inception_model)
    #     fig, orig_label, adversarial_label = draw_result(img, noise, adv_img)
    #     plt.savefig('out/targeted/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    #     plt.close(fig)
    #
    #     plt.plot(losses)
    #     plt.title('Cross Entropy Loss\norig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    #     plt.ylabel('loss')
    #     plt.xlabel('n_iterations')
    #     plt.savefig('out/targeted/loss_orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    #     plt.close()

    # Test transferrable noise
    # Add the noise to another image in a targeted attack
    images_gen = load_images('images')
    img = next(images_gen)
    adv_img, noise, losses = run_targeted_attack(step_size=step_size, image=img, label=834, model=inception_model)
    fig, orig_label, adversarial_label = draw_result(img, noise, adv_img)
    plt.savefig('out/innocent/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    plt.close(fig)

    innocent_img = next(images_gen)
    corrupted_img = innocent_img + noise
    fig, orig_label, adversarial_label = draw_result(innocent_img, noise, corrupted_img)
    plt.savefig('out/innocent/innocent_orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    plt.close(fig)

    # Plots loss evolution
    # step_sizes_ = [0.1, 0.05, 0.01, 0.005, 0.001]
    # all_losses = []
    # for step_size_ in step_sizes_:
    #     print('step_size = {}'.format(step_size_))
    #     adv_img, noise, losses = run_targeted_attack(step_size=step_size_, image=img, label=823, model=inception_model)
    #     all_losses.append(losses)
    #
    # plt.figure()
    # for i, loss in enumerate(all_losses):
    #     plt.plot(loss, label='step_size = {}'.format(step_sizes_[i]))
    # plt.title('Cross entropy loss and gradient descent step')
    # plt.ylabel('loss')
    # plt.xlabel('n_iterations')
    # plt.legend()
    # plt.savefig('out/loss_with_different_alphas.png')
    # plt.close()

    # Print loss en fonction de step_size step
    # non-normalizer le gradient nous empeche de sortir du minimum apparemment
