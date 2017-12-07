# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import logging
import matplotlib
matplotlib.use('TkAgg')
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
import os


def load_image(img_path):
    """
    Loads image and turns it into a PyTorch Tensor of one dimension
    Args:
        img_path ():

    Returns:
        image as PyTorch tensor
    """
    image = image_to_tensor(Image.open(img_path).convert('RGB'))
    return image


def load_images(input_dir):
    """
    Read png images from input directory in batches.
    Args:
      input_dir: input directory
    Yields:
      image: image
    """
    for image_path in glob.glob(os.path.join(input_dir, '*.png')):
        yield load_image(image_path)


def get_image_class(image, model):
    """
    Get image class as predicted by the model
    Args:
        image ():
        model ():

    Returns:
        Image class

    """
    # Create PyTorch tensor
    x = Variable(image, volatile=True)
    # Prediction
    label = model(x).data.max(1)[1].numpy()[0]
    label_name = labels[label]
    return label_name


def run_non_targeted_attack(alpha, image, model, n_iterations, eps):
    """
    Performs a non-targeted attack vs a given model_
    Args:
        eps (): noise cap
        n_iterations (): number of iterations of the gradient ascent
        alpha (): step size of the gradient ascent
        model (): model to attack
        image (): image on which we perform the attack

    Returns:
        transformed image and noise
    """
    # Here we do not care about the label, initialize it to null tensor
    label = torch.zeros(1, 1)
    # Record our losses
    losses = []
    # x will store the tensor image, y the tensor label
    x, y = Variable(image, requires_grad=True), Variable(label)

    # Perform gradient ascent
    for step in range(n_iterations):
        # Reset the gradients
        zero_gradients(x)
        # Forward propagation
        out = model(x)
        # Get predicted label
        y.data = out.data.max(1)[1]
        # Compute loss
        loss_tensor = loss(out, y)
        # Back propagation
        loss_tensor.backward()
        # Store running loss value
        losses.append(loss_tensor.data[0])
        # Compute normed gradient
        normed_grad = alpha * torch.sign(x.grad.data)
        # Update adversarial image
        step_adversarial = x.data + normed_grad
        # Compute adversarial noise added
        adversarial_noise = step_adversarial - image
        # Clamp our noise
        adversarial_noise = torch.clamp(adversarial_noise, -eps, eps)
        # Compute result with clamped noise
        result = image + adversarial_noise
        # Reclamp to make sure we stay in ImageNet format
        result = torch.clamp(result, 0.0, 1.0)
        # Update image value
        x.data = result

    return result, adversarial_noise, losses


def run_targeted_attack(alpha, image, label, model, n_iterations, eps):
    """
    Performs a targeted attack vs a given model_
    Args:
        eps (): noise cap
        label (): target label
        n_iterations (): number of iterations of the gradient ascent
        alpha (): step size of the gradient ascent
        model (): model to attack
        image (): image on which we perform the attack

    Returns:
        transformed image and noise
    """
    # Initialize our Tensor target label
    label = torch.Tensor([label]).long()
    # Record our losses
    losses = []
    # x will store the tensor image, y the tensor label
    x, y = Variable(image, requires_grad=True), Variable(label)

    # Perform gradient descent
    for step in range(n_iterations):
        # Reset the gradients
        zero_gradients(x)
        # Forward propagation
        out = model(x)
        # Compute loss
        loss_tensor = loss(out, y)
        # Back propagation
        loss_tensor.backward()
        # Store running loss value
        losses.append(loss_tensor.data[0])
        # Compute normed gradient
        normed_grad = alpha * torch.sign(x.grad.data)
        # Update adversarial image
        step_adversarial = x.data - normed_grad
        # Compute adversarial noise added
        adversarial_noise = step_adversarial - image
        # Clamp our noise
        adversarial_noise = torch.clamp(adversarial_noise, -eps, eps)
        # Compute result with clamped noise
        result = image + adversarial_noise
        # Reclamp to make sure we stay in ImageNet format
        result = torch.clamp(result, 0.0, 1.0)
        # Update image value
        x.data = result
    return result, adversarial_noise, losses


def non_targeted_attack(img):
    img = img
    label = torch.zeros(1, 1)

    x, y = Variable(img, requires_grad=True), Variable(label)
    for step in range(n_iterations_):
        zero_gradients(x)
        out = inception_model(x)
        y.data = out.data.max(1)[1]
        _loss = loss(out, y)
        _loss.backward()
        normed_grad = step * torch.sign(x.grad.data)
        step_adv = x.data + normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps_, eps_)
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result, adv


def draw_result(original_image, adversarial_noise, adversarial_image, model):
    """
    Plots original image, adversarial adversarial_noise and adversarial image for given model
    Args:
        original_image ():
        adversarial_noise ():
        adversarial_image ():
        model ():

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    # Get classes
    orig_class = get_image_class(original_image, model=model)
    attack_class = get_image_class(adversarial_image, model=model)

    ax[0].imshow(tensor_to_image(original_image[0]))
    ax[0].set_title('Original image: {}'.format(orig_class.split(',')[0]))

    ax[1].imshow(adversarial_noise[0].numpy().transpose(1, 2, 0))
    ax[1].set_title('Attacking adversarial_noise')

    ax[2].imshow(tensor_to_image(adversarial_image[0]))
    ax[2].set_title('Adversarial example: {}'.format(attack_class))
    for i in range(3):
        ax[i].set_axis_off()
    plt.tight_layout()
    # plt.savefig('original={}_adversarial={}.png'.format(orig_class, attack_class))
    plt.show()


if __name__ == '__main__':
    # Load the ImageNet labels
    with open('classes.txt') as f:
        labels = eval(f.read())

    # Auxiliary functions
    # Define PyTorch transform pipeline
    image_to_tensor = transforms.Compose([transforms.ToTensor(),
                                          transforms.Lambda(lambda t: t.unsqueeze(0))])
    # Reverse to PIL image
    tensor_to_image = lambda x: np.asarray(transforms.ToPILImage()(x))

    # Parameters
    eps_ = 2 * 8 / 225.
    n_iterations_ = 30
    norm = float('inf')
    alpha_ = 0.01

    # Initialize model and loss
    inception_model = inception_v3(pretrained=True, transform_input=True)
    loss = nn.CrossEntropyLoss()

    # Instantiate the model in some way
    inception_model.eval()

    # img = load_image('images/ac2a8a4777dbe746.png')
    img = load_image('images/sport_car.png')
    get_image_class(img, inception_model)

    # Non-targeted
    # adv_img, noise, losses = run_non_targeted_attack(image=img, alpha=alpha_, model=inception_model,
    #                                                  eps=eps_, n_iterations=n_iterations_)
    # draw_result(img, noise, adv_img, model=inception_model)

    adv_img_old, noise_old = non_targeted_attack(img=img)
    draw_result(img, noise_old, adv_img_old, model=inception_model)

    # Targeted
    # adv_img, noise, losses = run_targeted_attack(alpha=alpha_, image=img, label=831, model=inception_model,
    #                                              n_iterations=n_iterations_, eps=eps_)
    # draw_result(img, noise, adv_img, model=inception_model)

    # # Add the noise to another image
    # innocent_img = load_image('sport_car.png')
    # corrupted_img = innocent_img + noise
    # print('innocent_img = {}'.format(get_class(innocent_img)))
    # print('corrupted_img = {}'.format(get_class(innocent_img)))
    # # Ne fonctionne pas, inception finalement assez robuste meme a un fort noise
    # draw_result(innocent_img, noise, corrupted_img)
    #
    # alphas_ = [0.1, 0.05, 0.005, 0.001]
    # all_losses = []
    # for alpha_ in alphas_:
    #     print('alpha = {}'.format(alpha_))
    #     adv_img, noise, losses = run_targeted_attack(alpha=alpha_, image=img, label=823, model=inception_model)
    #     all_losses.append(losses)

    # plt.figure()
    # for i, loss in enumerate(all_losses):
    #     plt.plot(loss, label='alpha = {}'.format(alphas_[i]))
    # plt.xlabel('steps')
    # plt.legend()
    # plt.show()

    # print('over')
    # # Print loss en fonction de alpha step
    # # non-normalizer le gradient nous empeche de sortir du minimum apparemment
