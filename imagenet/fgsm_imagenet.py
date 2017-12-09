# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import os
import sys
from torch import nn
from torchvision.models.inception import inception_v3
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import glob
import numpy as np

# Add src path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fgsm import run_targeted_attack, run_non_targeted_attack, image_to_tensor, tensor_to_image, draw_result, \
    random_noise, get_label, plot_heatmap_results


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


def plot_bunch_results():
    # Load our images
    images_gen = load_images('images')
    for img in images_gen:
        # Non-targeted
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=model)
        plt.show()
        plt.savefig('out/non_targeted/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
        plt.close(fig)

        plt.plot(losses)
        plt.title('Cross Entropy Loss\norig_label={},adversarial_label={}'.format(orig_label, adversarial_label))
        plt.ylabel('loss')
        plt.xlabel('n_iterations')
        plt.savefig(
            'out/non_targeted/loss_orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
        plt.close()

        # Targeted
        adv_img, noise, losses = run_targeted_attack(image=img, label=823, model=model, **kwargs)
        fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=model)
        plt.savefig('out/targeted/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
        plt.close(fig)

        plt.plot(losses)
        plt.title('Cross Entropy Loss\norig_label={},adversarial_label={}'.format(orig_label, adversarial_label))
        plt.ylabel('loss')
        plt.xlabel('n_iterations')
        plt.savefig('out/targeted/loss_orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
        plt.close()


def plot_transferrable_noise():
    # Test transferrable noise
    # Add the noise to another image in a targeted attack
    images_gen = load_images('images')
    img = next(images_gen)
    adv_img, noise, losses = run_targeted_attack(image=img, label=834, model=model, **kwargs)
    fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=model)
    plt.savefig('out/innocent/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    plt.close(fig)

    innocent_img = next(images_gen)
    corrupted_img = innocent_img + noise
    fig, orig_label, adversarial_label = draw_result(innocent_img, noise, corrupted_img, model=model)
    plt.savefig('out/innocent/innocent_orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    plt.close(fig)


def plot_step_size():
    step_sizes_ = [0.1, 0.05, 0.01, 0.005, 0.001]
    all_losses = []
    for step_size_ in step_sizes_:
        print('step_size = {}'.format(step_size_))
        adv_img, noise, losses = run_targeted_attack(step_size=step_size_, image=img, label=823, model=model)
        all_losses.append(losses)

    plt.figure()
    for i, loss in enumerate(all_losses):
        plt.plot(loss, label='step_size = {}'.format(step_sizes_[i]))
    plt.title('Cross entropy loss and gradient descent step')
    plt.ylabel('loss')
    plt.xlabel('n_iterations')
    plt.legend()
    plt.savefig('out/loss_with_different_alphas.png')
    plt.close()


if __name__ == '__main__':
    # Parameters
    kwargs = {
        'eps': 2 * 8 / 225.,
        'n_iterations': 40,
        'step_size': 0.01
    }

    # Model
    model = inception_v3(pretrained=True, transform_input=True)
    # Instantiate the model
    model.eval()

    images_gen = load_images('images')

    eps_range = [5 * i / 200 for i in range(1, 11)]
    step_size_range = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    res = np.zeros(shape=(len(eps_range), len(step_size_range)))
    n_images = 20

    for i, eps in enumerate(eps_range):
        for j, step_size in enumerate(step_size_range):
            kwargs = {
                'eps': eps,
                'n_iterations': 40,
                'step_size': step_size
            }
            hparam = 'eps={},step_size={},n_iterations={}'.format(kwargs['eps'],
                                                                  kwargs['step_size'],
                                                                  kwargs['n_iterations'])
            print(hparam)

            # Run non targeted
            count = 0
            iter_count = 0
            for img in images_gen:
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
    plt.savefig('out/imagenet_successful_adv_images_using_not_norm_images_with_norm_model.png')
