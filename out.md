---
abstract: |
    A growing ensemble of tasks is being delegated to machine learning
    algorithms. Among them, classification of images is a major subset,
    especially with the recent developments of self-driving technology.
    Recent studies by Google Brain have shown that any machine learning
    classifier can be tricked to give incorrect predictions. Indeed, it is
    possible to trick them to get results very different from what is
    expected, and sometimes, pretty much any result you want. All the code
    used in this project lives in this public GitHub repository:
    <https://github.com/louis-r/adversarial_models>. TensorFlow
    [@tensorflow2015-whitepaper] was used.
author:
- 'Louis Rémus, Auriane Blarre and Romain Kakko-Chiloff'
bibliography:
- 'references.bib'
date: Fall 2017
title: 'Adversarial Attacks: Fooling Neural Nets'
---

Introduction
============

Adversarial attacks
-------------------

Fast Gradient Sign Method
=========================

Overview
--------

Algorithm
---------

### Non-Targeted Attacks

### Targeted Attacks

ImageNet
========

Presentation
------------

Inception Model
---------------

Results
-------

The Figure \[fig:losses\_alphas\] shows the evolution of cross entropy
loss with regards to the number of gradient descent iterations for
different gradient descent steps size.

![Cross-entropy loss evolution with different
alphas[]{data-label="fig:losses_alphas"}](figures/loss_with_different_alphas.png){width="70.00000%"}

![Caption[]{data-label="fig:pandas"}](figures/orig_label=giantpanda,panda,pandabear,coonbear,Ailuropodamelanoleuca,adversarial_label=suit,suitofclothes.png){width="\textwidth"}

MNIST dataset
=============

Presentation
------------

Our Model
---------

Results
-------

Where we are
============

-   Several adversarial attacks exist. We ran Fast Gradient Sign Method
    in on Inception. This method works well. Some findings:

    -   Targeted and non-targeted work

    -   Noise learned on one image is not transferrable to another image

    -   Gradient step plays a crucial role: if too important, we will
        basically do shit, cross the linear border back and forth, if
        too small, we will remain in the local minima. Graph

    -   Resistance to regularization

    -   One method to select noise strength is to pick the smallest at
        which we misclassify (obtain a convergence for our loss)

Where we go
===========

-   Test if noise learned by FGSM for Inception also work for other
    classifiers.

-   Evolution of noise with alpha step

-   Train some model to learn how to make noise and fool the classifier?
