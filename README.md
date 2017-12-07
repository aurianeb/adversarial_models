[![Build Status](https://travis-ci.org/louis-r/adversarial_models.svg?branch=master)](https://travis-ci.org/louis-r/adversarial_models)
# Adversarial Models
Research project on adversarial networks

# Models
## Neural Net
## Tree
## Logistic Regression

# Adversarial Algorithms
# FGSM: Fast Gradient Sign Method
# Change entropy criterion in trees

# Defenses
## Adversarial training
This is a brute force solution where we simply generate a lot of adversarial examples and explicitly train the model not to be fooled by each of them. An open-source implementation of adversarial training is available in the cleverhans library and its use illustrated in the following tutorial.
## Defensive distillation
This is a strategy where we train the model to output probabilities of different classes, rather than hard decisions about which class to output. The probabilities are supplied by an earlier model, trained on the same task using hard class labels. This creates a model whose surface is smoothed in the directions an adversary will typically try to exploit, making it difficult for them to discover adversarial input tweaks that lead to incorrect categorization. (Distillation was originally introduced in Distilling the Knowledge in a Neural Network as a technique for model compression, where a small model is trained to imitate a large one, in order to obtain computational savings.)

Source: https://blog.openai.com/adversarial-example-research/
