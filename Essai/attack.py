import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image



def draw_result(img, noise, adv_img):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(img)
    ax[0].set_title('Original image: {}'.format(orig_class.split(',')[0]))
    ax[1].imshow(noise)
    ax[1].set_title('Attacking noise')
    ax[2].imshow(adv_img)
    ax[2].set_title('Adversarial example: {}'.format(attack_class))
    for i in range(3):
        ax[i].set_axis_off()
    plt.tight_layout()
    plt.show()

# Attack methods

def calc_output_weighted_weights(output, w):
    for c in range(len(output)):
        if c == 0:
            weighted_weights = output[c] * w[c]
        else:
            weighted_weights += output[c] * w[c]
    return weighted_weights


def targeted_gradient(foolingtarget, output, w):
    ww = calc_output_weighted_weights(output, w)
    for k in range(len(output)):
        if k == 0:
            gradient = foolingtarget[k] * (w[k] - ww)
        else:
            gradient += foolingtarget[k] * (w[k] - ww)
    return gradient


def non_targeted_gradient(target, output, w):
    ww = calc_output_weighted_weights(output, w)
    for k in range(len(target)):
        if k == 0:
            gradient = (1 - target[k]) * (w[k] - ww)
        else:
            gradient += (1 - target[k]) * (w[k] - ww)
    return gradient


def non_targeted_sign_gradient(target, output, w):
    gradient = non_targeted_gradient(target, output, w)
    return np.sign(gradient)


# Attack class

class Attack:
    def __init__(self, model):

        self.fooling_targets = None
        self.model = model

    def prepare(self, X_train, y_train, X_test, y_test):
        self.images = X_test
        self.classes = y_train
        self.true_targets = y_test
        self.num_samples = X_test.shape[0]
        self.train(X_train, y_train)
        print("Model training finished.")
        self.test(X_test, y_test)
        print("Model testing finished. Initial accuracy score: " + str(self.initial_score))

    def set_fooling_targets(self, fooling_targets):
        self.fooling_targets = fooling_targets

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.weights = self.model.coef_
        self.num_classes = self.weights.shape[0]

    def test(self, X_test, y_test):
        self.preds = self.model.predict(X_test)
        self.preds_proba = self.model.predict_proba(X_test)
        self.initial_score = accuracy_score(y_test, self.preds)

    def create_one_hot_targets(self, targets):
        self.one_hot_targets = np.zeros(self.preds_proba.shape)
        for n in range(targets.shape[0]):
            self.one_hot_targets[n, targets[n]] = 1

    def attack(self, attackmethod, epsilon):
        perturbed_images, highest_epsilon = self.perturb_images(epsilon, attackmethod)
        perturbed_preds = self.model.predict(perturbed_images)
        score = accuracy_score(self.true_targets, perturbed_preds)
        return perturbed_images, perturbed_preds, score, highest_epsilon

    def attack_class(self, target_class, attackmethod, epsilon):
        indexes = [ind for ind in range(self.num_samples) if self.true_targets[ind] == target_class]
        images = self.images[indexes]
        labels = self.true_targets[indexes]
        perturbed_images, mean_perturbation = self.perturb_class(target_class, epsilon, attackmethod)
        perturbed_preds = self.model.predict(perturbed_images)
        score = accuracy_score(labels, perturbed_preds)

        draw_result(self.images[indexes][0], mean_perturbation, perturbed_images[0])

        return perturbed_images, perturbed_preds, score

    def perturb_images(self, epsilon, gradient_method):
        perturbed = np.zeros(self.images.shape)
        max_perturbations = []
        for n in range(self.images.shape[0]):
            perturbation = self.get_perturbation(epsilon, gradient_method, self.one_hot_targets[n], self.preds_proba[n])
            perturbed[n] = self.images[n] + perturbation
            max_perturbations.append(np.max(perturbation))
        highest_epsilon = np.max(np.array(max_perturbations))
        return perturbed, highest_epsilon

    def perturb_class(self, target_class, epsilon, gradient_method):
        indexes = [ind for ind in range(self.num_samples)
                   if self.true_targets[ind] == target_class]
        perturbed = np.zeros((len(indexes), self.images.shape[1]))
        max_perturbations = []
        mean_perturbation = np.zeros(self.images[0].shape)
        for n in indexes:
            perturbation = self.get_perturbation(epsilon,
                                                 gradient_method,
                                                 self.one_hot_targets[n],
                                                 self.preds_proba[n])
            mean_perturbation += perturbation

        mean_perturbation *= 1 / len(indexes)

        for n in range(len(indexes)):
            perturbed[n] = self.images[indexes[n]] + mean_perturbation
        highest_epsilon = np.max(np.array(mean_perturbation))
        return perturbed, mean_perturbation

    def get_perturbation(self, epsilon, gradient_method, target, pred_proba):
        gradient = gradient_method(target, pred_proba, self.weights)
        inf_norm = np.max(gradient)
        perturbation = epsilon / inf_norm * gradient
        return perturbation

    def attack_to_max_epsilon(self, attackmethod, max_epsilon):
        self.max_epsilon = max_epsilon
        self.scores = []
        self.epsilons = []
        self.perturbed_images_per_epsilon = []
        self.perturbed_outputs_per_epsilon = []
        for epsilon in range(0, self.max_epsilon):
            perturbed_images, perturbed_preds, score, highest_epsilon = self.attack(attackmethod, epsilon)
            self.epsilons.append(highest_epsilon)
            self.scores.append(score)
            self.perturbed_images_per_epsilon.append(perturbed_images)
            self.perturbed_outputs_per_epsilon.append(perturbed_preds)
