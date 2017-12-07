# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import os

from torch import nn
from torchvision.models.inception import inception_v3
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import glob

from src.fgsm import run_targeted_attack, run_non_targeted_attack, image_to_tensor, tensor_to_image, draw_result


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


if __name__ == '__main__':
    # Parameters
    kwargs = {
        'eps': 2 * 8 / 225.,
        'n_iterations': 40,
        'step_size': 0.01
    }

    # Model
    inception_model = inception_v3(pretrained=True, transform_input=True)
    # Instantiate the model
    inception_model.eval()

    # Load our images
    img = load_image('images/sport_car.png')
    images_gen = load_images('images')

    # for img in images_gen:
    #     # Non-targeted
    #     adv_img, noise, losses = run_non_targeted_attack(image=img, model=inception_model, **kwargs)
    #     fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=inception_model)
    #     plt.show()
    #     plt.savefig('out/non_targeted/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    #     plt.close(fig)

    #     plt.plot(losses)
    #     plt.title('Cross Entropy Loss\norig_label={},adversarial_label={}'.format(orig_label, adversarial_label))
    #     plt.ylabel('loss')
    #     plt.xlabel('n_iterations')
    #     plt.savefig(
    #         'out/non_targeted/loss_orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    #     plt.close()
    #
    #     # Targeted
    #     adv_img, noise, losses = run_targeted_attack(image=img, label=823, model=inception_model, **kwargs)
    #     fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=inception_model)
    #     plt.savefig('out/targeted/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    #     plt.close(fig)
    #
    #     plt.plot(losses)
    #     plt.title('Cross Entropy Loss\norig_label={},adversarial_label={}'.format(orig_label, adversarial_label))
    #     plt.ylabel('loss')
    #     plt.xlabel('n_iterations')
    #     plt.savefig('out/targeted/loss_orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    #     plt.close()

    # Test transferrable noise
    # Add the noise to another image in a targeted attack
    images_gen = load_images('images')
    img = next(images_gen)
    adv_img, noise, losses = run_targeted_attack(image=img, label=834, model=inception_model, **kwargs)
    fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=inception_model)
    plt.savefig('out/innocent/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    plt.close(fig)

    innocent_img = next(images_gen)
    corrupted_img = innocent_img + noise
    fig, orig_label, adversarial_label = draw_result(innocent_img, noise, corrupted_img, model=inception_model)
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
