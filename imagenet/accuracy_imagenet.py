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

# Add src path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fgsm import run_targeted_attack, run_non_targeted_attack, image_to_tensor, \
    tensor_to_image, draw_result, random_noise, get_label


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
    model = inception_v3(pretrained=True, transform_input=True)
    # Instantiate the model
    model.eval()

    # Load our images
    img = load_image('images/sport_car.png')
    images_gen = load_images('images')

    n_images = 10

    # Run non targeted
    count = 0

    for it in range(n_images):
        print("Non targeted iteration {}".format(it))
        img = next(images_gen)
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        adversarial_label = get_label(adv_img, model)
        predicted_label = get_label(img, model)

        if adversarial_label == predicted_label:
            count += 1

    print("Mean accuracy non targeted: ", 1.0 * count / n_images)

    # Run targeted
    count = 0

    for it in range(n_images):
        print("Targeted iteration {}".format(it))
        img = next(images_gen)
        label = get_label(img, model)

        # Pick a target label
        target_label = 823

        adv_img, noise, losses = run_targeted_attack(image=img, label=target_label, model=model, **kwargs)
        adversarial_label = get_label(adv_img, model)

        if adversarial_label == 'stethoscope':
            count += 1

    print("Mean accuracy targeted: ", 1.0 * count / n_images)

    # Comparaison with random noise
    different_label = 0
    fooled = 0

    for it in range(n_images):
        print("Random iteration {}".format(it))
        img = next(images_gen)
        # Model prediction on the image
        model_label = get_label(img, model)

        # Model prediction on the noisy image
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        adversarial_label = get_label(adv_img, model)

        # Model prediction with random noise
        adv_img, noise = random_noise(img, model, kwargs['eps'])
        random_adversarial_label = get_label(adv_img, model)

        if adversarial_label != model_label:
            different_label += 1

        if model_label != random_adversarial_label:
            fooled += 1

    print(different_label, fooled)