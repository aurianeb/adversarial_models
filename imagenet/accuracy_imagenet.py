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
from src.fgsm import run_targeted_attack, run_non_targeted_attack, image_to_tensor, tensor_to_image, draw_result, random_noise


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

    # Run non targeted
    n_images = 10
    count = 0
    iter_count = 0
    images_per_label = dict()
    count_per_label = dict()
    for i in range(10):
        images_per_label[i] = 0
        count_per_label[i] = 0

    for img in images_gen:
        iter_count += 1
        if iter_count > n_images:
            break
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=inception_model, **kwargs)
        adversarial_label = get_label(adv_img, inception_model)
        predicted_label = get_label(img, inception_model)

        images_per_label[label.numpy()[0]] += 1

        if adversarial_label == predicted_label:
            count += 1
            count_per_label[label.numpy()[0]] += 1

    print("Mean accuracy noisy: ", 1.0 * count / n_images)
    print("Image count per label", images_per_label)
    print("Accurate classification per label", count_per_label)

    accuracy_per_label = dict()
    for i in range(10):
        accuracy_per_label[i] = 1.0 * count_per_label[i] / images_per_label[i]
    print("Accuracy per label", accuracy_per_label)


    # Run targeted
    n_images = 10
    count = 0
    iter_count = 0
    images_per_label = dict()
    count_per_label = dict()
    for i in range(10):
        images_per_label[i] = 0
        count_per_label[i] = 0

    for img in images_gen:
        iter_count += 1
        if iter_count > n_images:
            break

        label = get_label(img, inception_model)

        # Pick a target label
        possible_labels = [x for x in range(10) if x != label.numpy()[0]]
        target_label = choice(possible_labels)

        adv_img, noise, losses = run_targeted_attack(image=img, label=target_label, model=inception_model, **kwargs)
        adversarial_label = get_label(adv_img, inception_model)

        images_per_label[label.numpy()[0]] += 1

        if adversarial_label == target_label:
            count += 1
            count_per_label[label.numpy()[0]] += 1

    print("Mean accuracy noisy: ", 1.0 * count / n_images)
    print("Image count per label", images_per_label)
    print("Accurate classification per label", count_per_label)

    accuracy_per_label = dict()
    for i in range(10):
        accuracy_per_label[i] = 1.0 * count_per_label[i] / images_per_label[i]
    print("Accuracy per label", accuracy_per_label)