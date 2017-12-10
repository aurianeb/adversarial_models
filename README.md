---
abstract: |
    A growing ensemble of tasks is being delegated to machine learning
    algorithms. Among them, classification of images is a major subset,
    especially with the recent developments of self-driving technology.
    Recent studies by Google Brain (see
    [@Adversarial_examples_in_the_physical_world]) have shown that any
    machine learning classifier can be tricked to give incorrect
    predictions. In this project, we successfully conducted targeted and non
    targeted attacks on a pre-trained ImageNet Inception classifier and on a
    custom neural classifier trained on the MNIST dataset, tuning attack
    hyperparamaters to achieve the most efficient attacks. The success rate
    of our attacks calls, in our opinion, for a need to move past
    gradient-based optimization. We also proved that the adversarial noise
    learned on one model is transferable to another model, which means that
    using ensemble or bagging methods as a defense against adversarial
    examples will not work. We also proved that the adversarial noise
    learned had a very particular structure related to the attacked image,
    and could not be successfully applied to other images. Lastly, we
    experimented several customized attacks on the ImageNet dataset. All the
    code used in this project lives in this public GitHub repository:
    <https://github.com/louis-r/adversarial_models>.
author:
- 'Louis Rémus, Auriane Blarre'
bibliography:
- 'references.bib'
date: Fall 2017
title: 'Adversarial Attacks: Fooling Neural Nets'
---

Introduction
============

Over the past few years, recent progress in machine learning and deep
neural networks have enabled researchers to solve several crucial
complex problems such as image and text classification.

However, it has been also showed that machine learning models are often
vulnerable to adversarial manipulation of their input intended to cause
incorrect classification (Dalvi et al., 2004 in [@proceedings]). In
particular, neural networks and many other categories of machine
learning models are highly vulnerable to attacks based on small
modifications of the input to the model at test time (Goodfellow et al.,
2014 in [@harnessing]).

General Paradigm
----------------

Let us say there is a classifier $ C $ and input sample $ X $ which we
call a clean example. Let us assume that sample $ X $ is correctly
classified by the classifier, i.e. $ C(X) = y_{true} $. We can construct
an adversarial example $ A $ which is perceptually indistinguishable
from $ X $ but is classified incorrectly, i.e. $ C(A) \neq y_{true} $.
These adversarial examples are misclassified far more often than
examples that have been perturbed by random noise, even if the magnitude
of the noise is much larger than the magnitude of the adversarial noise,
see (Szegedy et al., 2013 in [@DBLP:journals/corr/SzegedyZSBEGF13]).

Adversarial Attacks
-------------------

The challenge is to make to design an adversarial noise so subtle that a
human observer does not even notice the modification, while the
classifier makes a mistake.

Note that the classifier is left untouched, which corresponds to the
real setting, where we rarely if ever have access to the classifier
design.

Types of Adversarial Attacks
----------------------------

We distinguish are two types of adversarial attacks : the non-targeted
attacks and the targeted attacks.

Non-targeted Attacks :

:   The non-targeted attacks goal is to modify slightly the input (so
    the image) so that the unknown classifier misclassifies the input.
    In this case, we do not choose the classification of the fooled
    input.

Targeted Attacks :

:   Here, we choose the class of the fooled input. Indeed, given an
    image in class $ A $ that we want to misclassify in class $ B $, we
    compute an adversarial noise to modify this image to have in
    classified in $ B $ by the unknown classifier.

Methods of Generating Adversarial Images
========================================

Notations
---------

We use the following notations:

-   $ X $, an image, which is either a 3D tensor (width x height x
    depth) for ImageNet, either a 2D tensor (width x height) for MNIST.
    In our setting, the values of the pixels are in the range
    $[0, 255]$.

-   $ y_{true} $, true class for the image $ X $.

-   $ CE(X, y_{true}) $, cross-entropy cost function of the neural
    network for image $ X $ and class $ y_{true} $.

-   $ \epsilon $, upper bound on the infinity norm of the noise added.

-   $ Clamp(X, \epsilon_{min}, \epsilon_{max}) $, clamps all elements in
    $ X $ into the range $ [\epsilon_{min}, \epsilon_{max}] $

Fast Gradient Sign Method
-------------------------

This simple method first introduced by Goodfellow et al. in
[@harnessing] in 2014 essentialy relies on a modification of the image
based on a normalized gradient update.

$$\label{eq:fgsm}
    X_{adversarial} = X + \alpha \times \text{sign}(\Delta_{X}(CE(X, y_{true}))$$

where $ \alpha $ is a hyperparameter step to be chosen. Note that we
only use the sign of the gradient of the cross entropy cost here, for
reasons we will explain later in this report. The Fast Gradient Sign
Method (FGSM) of Equation \[eq:fgsm\] can be improved by performing
updates and clamp values after each step to make sure that they remain
$ \epsilon $-close to the original image:

$$\label{eq:bim}
    \begin{split}
        X^{adversarial}_{0} &= X \\
        X^{adversarial}_{N} &= X^{adversarial}_{N-1} + \alpha \times \text{sign}(\Delta_{X}(CE(X^{adversarial}_{N-1}, y_{true}))
    \end{split}$$

Note that the method presented in Equation \[eq:bim\] is often referred
to as the Basic Iterative Method (BIM).

Algorithm
---------

The algorithm takes an image as an input and returns a noise that, when
added to the image, will change the prediction of the classifier. The
noise is almost imperceptible by the naked eye as each pixel is inferior
to an arbitrarily chosen hyperparameter $ \epsilon_{max} $.

### Non-Targeted Attacks

The goal of non-targeted attacks is to trick the classifier into not
predicting the label originally predicted for the image. This is
achieved by performing a gradient ascent on the label predicted by the
model. The algorithm is presented in Algorithm \[algo:non\_targeted\].

**End**

The function starts by initializing the computing the label of the image
as computed by the model $y_{true}$ line 2.\
X is the current noisy image. It is initialized with the image.\
Then at every step of the iteration between 1 and $n_{max}$, the
function:

-   Computes a distribution of probability on the labels for the current
    noisy image X: $out$.

-   Computes the loss between this distribution of probability and the
    label $y_{true}$ according to the Cross Entropy loss function.

-   The gradient is normalized with the $sign$ function. This operation
    will prevent the gradient ascent from being stuck around a local
    minimum. Indeed, if the point X is very close to the local minimum
    of $ X \rightarrow \Delta_{X} Loss(X, y_{true}) $ then the norm of
    the loss will be very small. This is useful in the gradient descent
    setting where we aim at a local minimum but in the non targeted
    attack, we perform a gradient ascent and our goal is to land far
    from the local minimum. As our model is pretrained, we are already
    close to a local minimum, therefore
    $X_{next} \leftarrow X + \Delta_{X} Loss(y_{true}, out)$ would add
    only a very small value to our current value and the convergence
    would take a lot of iterations. Conversely, when X is “far” from the
    local minimum taking steps of size $ \alpha \times sign(loss) $ is
    not a problem.

-   We are performing a gradient ascent so the loss is added to the
    current image line 8, and not substracted as in a gradient descent.

-   The noise on the original image is computed as the difference
    between the total noise after this step
    ($X_{next} \leftarrow X + loss$ line 8) and the original image (line
    9).

-   To remain indistinguishable by the human eye, the noise is clamped
    between values $-\epsilon_{max}$ and $\epsilon_{max}$.

-   Actualize the value of X as the sum of the original image and the
    noise.

### Targeted Attacks

The goal of non-targeted attacks is to trick the classifier into
predicting a target label for the image. This is achieved by performing
a gradient descent on the input image and the target label. The
algorithm is presented in Algorithm \[algo:targeted\].

**End**

The algorithm is very similar to the non targeted attack. One difference
is that we do not normalize the loss by using the $sign$ function since
we are performing a gradient descent and are trying to get closer to a
local minimum. Moreover, we substract the step from the current noisy
image on line 6 in the regular gradient descent fashion.

Experiments
===========

We conducted our adversarial attacks on two datasets: the ImageNet
dataset and the MNIST dataset.

ImageNet
--------

Note that given the size of the ImageNet dataset, we only used a subset
of 1000 images from this dataset. The data is available as part of a
Kaggle contest at this address
<https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack>.
Each image has dimension $[299, 299, 3]$.

### Inception Model

To save time, we decided to use a pretrained ImageNet Inception V3
model, available in .

### Model Structure

![Inception v3 model
structure[]{data-label="fig:inception_v3"}](figures/inception_v3.png){width="\textwidth"}

MNIST
-----

We also conducted adversarial attacks on the MNIST dataset to compare
the performance of our algorithms on a dataset with a smaller dimension
since each MNIST image has dimension $[28, 28]$. We will see that this
difference in dimensionality plays a crucial role in adversarial
attacks. The data was obtained from API.

### BaseNet Model

We built and trained a custom model on the MNIST dataset. Our training
set has 60000 images and our testing set has 10000 images. Note that, as
illustrated by the training logs in the Appendix, our model gives us 98%
accuracy on a test batch of 1000 images for 10 epochs and a learning
rate of 0.01.

### Logistic Regression Model

We also trained another classifier so as to see whether adversarial
noise learned on MNIST by our was transferable to another classifier. We
picked a Logistic Regression model mainly because it could be written as
a Neural Net model with one layer, which enabled us to use the same
pipeline of the model. As illustrated in the training log, our model
gives us 92% accuracy on a test batch of 1000 images for 10 epochs and a
learning rate of 0.01.

Results
=======

ImageNet
--------

### Non Targeted Attacks

The following images presents the results of adversarial attacks
performed on the Inception model with the following hyperparameters.

$$\begin{split}
    \epsilon = \frac{16}{225} = 0.07 \\
    n_{iterations} = 40 \\
    \alpha = 0.01
\end{split}$$

As $ n_{iterations} > 1 $, we are performing a BIM attack. We also
display the losses associated with these attacks

![Non targeted BIM attack on beetle:
loss[]{data-label="fig:loss_beetle_non_targeted"}](figures/non_targeted/loss_orig_label=long-hornedbeetle,longicorn,longicornbeetle,adversarial_label=starfish,seastar){width="40.00000%"}

![Non targeted BIM attack on cannon:
loss[]{data-label="fig:loss_cannon_non_targeted"}](figures/non_targeted/loss_orig_label=cannon,adversarial_label=bassinet.png){width="40.00000%"}

![Non targeted BIM attack on ostrich:
loss[]{data-label="fig:loss_ostrich_non_targeted"}](figures/non_targeted/loss_orig_label=ostrich,Struthiocamelus,adversarial_label=dhole,Cuonalpinus.png){width="40.00000%"}

From these sample images, we can make very interesting observations.

-   We observe that the adversarial noise added is not random but has a
    structure that visually seems to match the image structure. From
    this, we can guess that the adversarial noise computed for one image
    will not work as well on another image, which is confirmed by our
    experimental results.

-   It was surprisingly easy to fool the Inception model, despite its
    very high level of accuracy. In the Figures
    \[fig:beetle\_non\_targeted\], \[fig:cannon\_non\_targeted\] and
    \[fig:ostrich\_non\_targeted\] above, we performed a BIM attack with
    $ n_{iterations} = 40 $ but we observed later that even a FGSM (ie a
    BIM with $ n_{iterations} = 1 $) was sufficient to fool the
    Inception model! Thus, we can infer that it is very easy for an
    adversarial algorithm to cross the border of the decision function.

Let us illustrate how well a FGSM attack work on the Inception model:
the following image is a non targeted adversarial attack performed on
the Inception model with the following hyperparamaters.

$$\begin{split}
    \epsilon = \frac{16}{225} = 0.07 \\
    n_{iterations} = 1 \\
    \alpha = 0.01
\end{split}$$

Thus, we only do one iteration and perform a FGSM attack here. Even with
one iteration, we observe that this is enough to fool the Inception
model.

Here, and contrary to the previous figures, the noise is
indistinguishable. Note something very interesting: the Inception model
is indeed wrong but not that much, since the pattern on the ground of
the cannon indeed looks like a mosquito net. If we increase the value of
$ n_{iterations} $, the Inception model will make a very different
prediction and will be completely incorrect. This phenomenon can be
observed on the pattern of the losses displayed in Figures
\[fig:loss\_beetle\_non\_targeted\], \[fig:loss\_cannon\_non\_targeted\]
and \[fig:loss\_ostrich\_non\_targeted\]: the loss increases with
$ n_{iterations} $, meaning that we can farther and farther away from
the original label (during our gradient ascent), even though the label
was already incorrect after the first step.

The following image (taken from Goodfellow CS231n 2017 lecture) shows
how the FGSM crosses the classification boundaries.

![Adversarial
map[]{data-label="fig:adv_map"}](figures/maps_adversarial.png){width="60.00000%"}

Our results tell us that for the Inception model, the classes are very
close to each other, since for most of them we can go to another
classification zone in just one step with a reasonably small noise
bound.

### Targeted Attacks

We then perform targeted attacks against the same images, aiming for the
label stethoscope. Note that we reach the target label in all the cases
and we could have reached it with smaller $ n_{iterations} $ and thus
smaller distortion of the original image.

![Targeted BIM attack on beetle:
loss[]{data-label="fig:loss_beetle_targeted"}](figures/targeted/loss_orig_label=long-hornedbeetle,longicorn,longicornbeetle,adversarial_label=stethoscope.png){width="40.00000%"}

![Targeted BIM attack on cannon:
loss[]{data-label="fig:loss_cannon_targeted"}](figures/targeted/loss_orig_label=cannon,adversarial_label=stethoscope.png){width="40.00000%"}

![Targeted BIM attack on ostrich:
loss[]{data-label="fig:loss_ostrich_targeted"}](figures/targeted/loss_orig_label=bobsled,bobsleigh,bob,adversarial_label=stethoscope.png){width="40.00000%"}

### Overall Performance

We performed a grid search over two hyperparameters, $ \epsilon $ and
$ \alpha $. For each couple $ (\epsilon,\alpha) $, we compute the
percentage of successful attacks on 20 randomly selected images, with
$ n_{iterations} = 1 $, therefore with a FGSM attack. The attack here is
non targeted.

![ImageNet: % of successful adversarial images on attacks:
loss[]{data-label="fig:imagenet_succ_attacks"}](imagenet_successful_adv_images.png){width="\textwidth"}

Considering Figure \[fig:imagenet\_succ\_attacks\], it is safe to say
that the Inception model is quite easy to attack! On some combinations,
we obtain 100 % of successful attacks with only 1 iteration.

### Baseline Comparison: Random Noise

We measured the accuracy of the non targeted and the targeted attacks on
the Inception model on a random subset of 10 images. As we consider non
targeted and targeted attacks, we define one metric per attack type:

$$\text{NonTargetedAttackAccuracy} = \frac{\# \{C(X) \neq y_{true}\}}{n_{images}}$$

$$\text{TargetedAttackAccuracy} = \frac{\# \{C(X) = y_{target}\}}{n_{images}}$$

where $ C $ if the classifier, $ X $ is the image, $ y_{true} $ the true
label of $ X $, $ y_{target} $ the target label for the targeted attack.

The parameters used for the experiments are:

$$\begin{split}
    \epsilon = \frac{16}{225} = 0.07 \\
    n_{iterations} = 40 \\
    \alpha = 0.01
\end{split}$$

Note that as $ n_{iterations} > 1 $, we are performing a BIM attack.

For both algorithms, the attack accuracy was $100\%$. To put those
results in perspective, we led the same tests with this time a randomly
generated noise. We used the same upper bound
$\epsilon = \frac{16}{225} $ to cap the noise on each pixel. On a set of
10 images, the random noised fooled the model $ 0\% $ of the time when
the non targeted attack fooled the model $ 100\% $ of the time.

Therefore, the BIM method is a drastic improvement over the baseline
method.

Increasing the level of noise for the baseline did not improve the
baseline performance, therefore we understand that it is the *structure*
of the noise that matters: a smaller, but well-built noise for a non
targeted attack has a better performance that a bigger random noise
attack.

### Non targeted attack with update proportional to gradient

We previously explained that in the algorithm of the non targeted
attack, the gradient is normalized so that the ascent does not get stuck
around a local minimum. We verified that assumption by running the non
targeted algorithm (Algorithm \[algo:non\_targeted\]) with and without
the $sign$ normalization.

![Loss in logarithmic scale computed at each iteration of the gradient
ascent with and without the
normalization[]{data-label="fig:normalization"}](figures/log_not_normalized_loss.png){width="50.00000%"}

Erratum: note that in the Figure \[fig:normalization\], Fast Gradient
Descent actually stands for Basic Iterative Method (BIM), and Gradient
Descent for Gradient Ascent (if we do not normalized in Algorithm
\[algo:non\_targeted\], we are effectively performing gradient ascent).

As shown in Figure \[fig:normalization\], the loss of the Gradient
Ascent is increasing over the iterations at a rate much slower than that
of the BIM attack: the sign normalization is crucial.

### Convergence with step size $\alpha$

The following figures shows the speed of divergence (gradient ascent) of
different alphas.

![Cross entropy loss w.r.t iteration for different step sizes for the
non targeted
attack[]{data-label="fig:ce_alpha_non_targeted"}](figures/non_targeted/loss_with_different_alphas_non_targeted.png){width="50.00000%"}

![Cross entropy loss w.r.t iteration for different step sizes for the
targeted
attack[]{data-label="fig:ce_alpha_targeted"}](figures/loss_with_different_alphas.png){width="50.00000%"}

-   When the step size is too large ($ \alpha = 0.1, \alpha = 0.05 $),
    the loss oscillates in both Figures \[fig:ce\_alpha\_non\_targeted\]
    and \[fig:ce\_alpha\_targeted\]. It seems we get out of the
    classification region of the neural net, and then step back in: this
    is an usual problem with gradient descent based methods. We jump
    around the local minimum but do not get any closer.

-   When the steps are smaller
    ($ \alpha = 0.01, \alpha = 0.005, \alpha = 0.001 $) however, the non
    targeted attack in Figure \[fig:ce\_alpha\_non\_targeted\] converges
    towards a bigger loss, while the targeted attack in Figure
    \[fig:ce\_alpha\_targeted\] converges. Let’s notice that the
    smallest step size ($ \alpha = 0.001 $) converges at a slower rate
    than for instance $ \alpha = 0.005 $ but asymptotically achieves
    greater loss. This is because the algorithm with $ \alpha = 0.001 $
    takes smaller steps so cannot converge as fast as with steps
    $ \alpha = 0.005 $ but is more precise in the sense it can get
    closer to the local minimum.

MNIST
-----

### Non Targeted Attacks

The MNIST dataset was much harder to fool. For the ImageNet/Inception
model, it seems that the high dimensionality of the images
($[299, 299, 3]$ vs $[28, 28]$) makes it easier to fool than the MNIST
dataset.

We used our with the following hyperparameters in a non targeted attack.

$$\begin{split}
    \epsilon = \frac{2}{225} = 0.008 \\
    n_{iterations} = 40 \\
    \alpha = 0.1
\end{split}$$

We see that with carefully chosen hyperparameters, we are able to
successfully conduct a adversarial attack on the MNIST dataset

### Study of accuracy

#### Convolutional neural network

Here is a table summarizing the accuracy of the non targeted and
targeted algorithm ran on the 10000 images of the testing MNIST data
set. We use the same NonTargetedAttackAccuracy and
TargetedAttackAccuracy as with the ImageNet dataset:

$$\text{NonTargetedAttackAccuracy} = \frac{\# \{C(X) \neq y_{true}\}}{n_{images}}$$

$$\text{TargetedAttackAccuracy} = \frac{\# \{C(X) = y_{target}\}}{n_{images}}$$

where $ C $ if the classifier, $ X $ is the image, $ y_{true} $ the true
label of $ X $, $ y_{target} $ the target label for the targeted attack.

The parameters used for the experiments are:

$$\begin{split}
    \epsilon = \frac{40}{225} \\
    n_{iterations} = 100 \\
    \alpha = 0.05
\end{split}$$

Note that as $ n_{iterations} > 1 $, we are performing a BIM attack.

We computed the NonTargetedAttackAccuracy for each class in Table
\[table:acc\_non\_targeted\].

   Global   Label 0   Label 1   Label 2   Label 3   Label 4   Label 5   Label 6   Label 7   Label 8   Label 9
  -------- --------- --------- --------- --------- --------- --------- --------- --------- --------- ---------
    0.30     0.10      0.29      0.29      0.22      0.67      0.20      0.29      0.39      0.33      0.50

  : Accuracy of the non targeted
  algorithm[]{data-label="table:acc_non_targeted"}

We computed the TargetedAttackAccuracy for every combination of labels
in Table \[table:acc\_targeted\]: how easy is it to trick the classifier
into predicting the label “To” (columns) for the images of the class
“From” (rows)? (Adversarial noise added to the class “From” (rows).

   From/To   Label 0   Label 1    Label 2   Label 3   Label 4   Label 5    Label 6   Label 7   Label 8   Label 9
  --------- --------- ---------- --------- --------- --------- ---------- --------- --------- --------- ----------
   Label 0             **0.03**    0.18      0.06      0.07       0.07      0.25      0.13      0.09       0.08
   Label 1    0.10                 0.18      0.17      0.25       0.12      0.25      0.24      0.29       0.26
   Label 2    0.42       0.69                0.32      0.41       0.10      0.59      0.39      0.54       0.31
   Label 3    0.25       0.59      0.35                0.25     **0.65**    0.18      0.44      0.49       0.54
   Label 4    0.00       0.06      0.03      0.02                 0.08      0.28      0.12      0.14       0.68
   Label 5    0.34       0.22      0.05      0.28      0.26                 0.38      0.19      0.59       0.44
   Label 6    0.17       0.10      0.05      0.03      0.20       0.23                0.01      0.21       0.04
   Label 7    0.08       0.43      0.16      0.07      0.34       0.14      0.03                0.17     **0.73**
   Label 8    0.28       0.58      0.28      0.22      0.40       0.45      0.47      0.28                 0.57
   Label 9    0.03       0.08      0.07      0.08      0.48       0.16      0.08      0.29      0.14    

  : Accuracy of a targeted targeted attack label by
  label[]{data-label="table:acc_targeted"}

We observe that some combinations reach greater accuracy than other. For
instance, it is easy to fool a classifier into predicting a 9 from the
picture of a 7 (accuracy $ 73 \% $) or a 5 from a 3 (accuracy
$ 65 \% $). This makes sense since the handwritting of those figures are
similar. On the other hand, it is hard to make the classifier predict a
1 from the picture of a 0 (accuracy $ 3 \% $).\
For both algorithms, the performance is well below what we got on the
ImageNet dataset. It is more difficult to fool a classifier on this
dataset as there are less labels (10 vs 1000 on ImageNet), and the
dimensionality is lower.

#### Logistic Regression

With the same definitions of accuracy, we get the following results for
the logistic regression:

   Global   Label 0   Label 1   Label 2   Label 3   Label 4   Label 5   Label 6   Label 7   Label 8   Label 9
  -------- --------- --------- --------- --------- --------- --------- --------- --------- --------- ---------
    0.64     0.54      0.78      0.62      0.61      0.65      0.75      0.65      0.64      0.52      0.65

  : Accuracy of the non targeted algorithm for the logistic regression

   From/To   Label 0   Label 1   Label 2   Label 3   Label 4   Label 5   Label 6   Label 7   Label 8   Label 9
  --------- --------- --------- --------- --------- --------- --------- --------- --------- --------- ---------
   Label 0     0.0      0.43      0.80      0.84      0.92      0.99      0.78      0.90      0.66      0.86
   Label 1    0.01       0.0      0.56      0.52      0.27      0.45      0.27      0.45      0.46      0.40
   Label 2    0.74       1.0       0.0      0.89      0.98      0.90       1.0      0.91       1.0      0.95
   Label 3    0.79       1.0      0.99       0.0      0.96       1.0      0.95       1.0      0.99       1.0
   Label 4     0.3      0.84      0.66      0.69       0.0      0.92      0.94      0.97      0.82       1.0
   Label 5    0.78      0.93      0.67       0.9      0.91       0.0      0.81      0.98      0.76      0.90
   Label 6    0.70       1.0      0.95      0.53       1.0      0.93       0.0      0.77      0.84      0.97
   Label 7    0.33      0.92      0.48      0.54      0.99      0.61      0.42       0.0      0.37      0.99
   Label 8     0.9       1.0       1.0       1.0       1.0       1.0       1.0       1.0       0.0       1.0
   Label 9    0.61       1.0      0.82      0.93       1.0      0.99      0.89       1.0      0.96       0.0

  : Accuracy of a targeted targeted attack label by label for the
  logistic regression

The accuracy of the attack is much higher. It is easier to fool the
logistic regression than the BIM. This makes sense since this model is
simpler than BIM and does not take into account the proximity of pixels
in a convoluted way.

### Baseline Comparison: Random Noise

To measure the accuracy of the non targeted attack, we compared its
results to those of the baseline case. For a sample of images, we
generated a random noise within the bounds $[-\epsilon, \epsilon]$. Then
we compared the accuracy of the classifier on the randomly noised images
to the accuracy of the non targeted attack. The non targeted model does
not bring any improvement over the baseline case: on a sample of 50
images, the image with random noise fooled the classifier 7 times and so
did the image with the non targeted noise.

### Modification of the gradient ascent step

For the gradient ascent to be effective, the algorithm needs to take
large steps near the local minimum (where the norm of the gradient is
small) and can take small steps when far away from the local minimum
(where the norm of the gradient is large). For that reason, we thought
it would be a good idea to update the gradient ascent with the inverse
of the gradient.\
More specifically, step 7 of Algorithm \[algo:non\_targeted\] becomes:
$ loss \leftarrow \alpha \times \text{pinv}(CE(X, y_{true})) $ where
$ \text{pinv}(CE(X, y_{true})) $ is the pseudo inverse of the loss
matrix.\
The results follow our intuition: the convergence of this new method is
faster in the very first steps but asymptotically, it yields a loss
almost half that of our previous method.

![Cross-entropy loss evolution of the two versions of the non targeted
attack. v1 is the method presented in algorithm 1, v2 is the method with
the pseudo
inverse[]{data-label="fig:targeted_v1_v2"}](figures/non_targeted/targeted_v1_v2.png){width="50.00000%"}

### Results for different step sizes $\alpha$ and upper noise bounds $\epsilon$

We use the same definition of accuracy: NonTargetedAttackAccuracy and
TargetedAttackAccuracy. The following figures were computed with the
model.

![Accuracy of the non targeted attack as a function of the upper bound
on the noise, for different step
sizes[]{data-label="fig:non_targeted_accuracy_and_maximum_noise"}](figures/non_targeted/Accuracy_and_Maximum_noise.png){width="50.00000%"}

From Figure \[fig:non\_targeted\_accuracy\_and\_maximum\_noise\], we see
that both the upper bound on the noise and the step size limit the
performance. For the non targeted attack the larger the noise and the
larger the step size, the greater the loss. We will work with a step
size $ \alpha = 0.1 $ and an upper bound on the noise
$ \epsilon = 0.3 $. Indeed, we want to keep the upper bound on the noise
as small as possible to keep the noise invisible. These observations are
also valid for the targeted attack, as proved in Figure
\[fig:targeted\_accuracy\_and\_maximum\_noise\] below.

![Accuracy of the targeted attack as a function of the upper bound on
the noise, for different step
sizes[]{data-label="fig:targeted_accuracy_and_maximum_noise"}](figures/targeted/Accuracy_and_Maximum_noise.png){width="50.00000%"}

### Image Noise Transfer

We have tried to fool the classifier by adding the noise generated by
BIM for one image on an another images. In the table below, find the
accuracy of the non targeted attack when adding the noise of an image of
the label Noise (columns) on an image of the label Image (row).

   Image/Noise   Label 0   Label 1   Label 2   Label 3   Label 4   Label 5   Label 6   Label 7   Label 8   Label 9
  ------------- --------- --------- --------- --------- --------- --------- --------- --------- --------- ---------
     Label 0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
     Label 1       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
     Label 2       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
     Label 3       0.0       0.0       0.0       0.0       0.0      0.01       0.0       0.0       0.0       0.0
     Label 4       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
     Label 5       0.0       0.0       0.0      0.01       0.0       0.0       0.0       0.0       0.0       0.0
     Label 6       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
     Label 7      0.01       0.0      0.01       0.0       0.0       0.0       0.0       0.0       0.0       0.0
     Label 8       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
     Label 9       0.0       0.0       0.0       0.0      0.01       0.0       0.0       0.0       0.0       0.0

  : Accuracy of non targeted attack with noise from another image

We observe that the accuracy is almost always 0: the noise is not
transferable from one image to the other. This is coherent with our
above observations: the noise has a structure very specific to the
image.

### Model Noise Transfer

Is it possible to use the adversarial noise learned for one model to
trick an other model? We generated adversarial noise for the classifier
and the on a sample of 50 images. For each tuple (image, adversarial
noise, adversarial noise), we cross-classified: the image with the
adversarial noise was classified with the classifier, the image with the
adversarial noise was classified with the classifier. We used the best
hyperparameters of Figures
\[fig:non\_targeted\_accuracy\_and\_maximum\_noise\] and
\[fig:targeted\_accuracy\_and\_maximum\_noise\].

-   For the targeted attack:

    -   For an adversarial noise, we achieved an
        NonTargetedAttackAccuracy of 87 % for .

    -   For an adversarial noise, we achieved an
        NonTargetedAttackAccuracy of 80 % for .

-   For the non targeted attack:

    -   For an adversarial noise, we achieved an TargetedAttackAccuracy
        of 85 % for .

    -   For an adversarial noise, we achieved an TargetedAttackAccuracy
        of 77 % for .

Therefore, for both models, the adversarial noise learned is
transferable to the other model.

In both the targeted and non targeted attack, we notice that the label
predicted by both and are very often the same with or adversarial noise.
This means that the two noises share a similar structure, which is why
they are transferable.

Conclusion
==========

In this project, we explored the possibility of creating adversarial
images for two famous datasets: ImageNet and MNIST. We illustrated how
successful adversarial attacks were, leading us to think that the
classification boundary has a simple structure. We also illustrated how
the adversarial noise learned on one classifier, our was transferable to
another classifier, our , and conversely, which shows that the
adversarial noise generalizes well and has a common structure among
relatively similar classifiers. We also demonstrated that the structure
of the noise was crucial: targeted and non targeted were much more
effective than random noise attacks.

Remarks
=======

-   It seems that every algorithm which is easy to optimize is easy to
    perturb: do we need to move past gradient-based optimization to
    overcome adversarial examples?

-   A smaller, but well-built noise for a non targeted attack has a
    better performance that a bigger random noise attack.

-   As we saw that adversarial noise was transferable, **we cannot use
    ensembles or bagging methods as a defense against adversarial
    attacks**.

-   One must consider the existence of adversarial examples when
    deciding whether to use machine learning.

-   We did not need access to the model parameters or the training set.
    This makes the task even easier for an attacker with bad intentions.

-   To actually assess the robustness of its model, a researcher must
    measure his model’s error rate on fast gradient sign method
    adversarial examples and report it.

Technical Note
==============

Tools
-----

We used the following tools for our project:

-   PyLint was used to make static inspections of our codebase.

-   Travis was used to trigger tests at each push to our codebase.

-   All the models were implemented using .

Training Logs
-------------

### MNIST 

\[ frame=lines, framesep=2mm, baselinestretch=1.2, fontsize=, linenos \]
[python]{}

Test set: Average loss: 0.2002, Accuracy: 9436/10000 (94Test set:
Average loss: 0.1252, Accuracy: 9615/10000 (96Test set: Average loss:
0.0997, Accuracy: 9701/10000 (97Test set: Average loss: 0.0820,
Accuracy: 9733/10000 (97Test set: Average loss: 0.0780, Accuracy:
9758/10000 (98Test set: Average loss: 0.0655, Accuracy: 9786/10000
(98Test set: Average loss: 0.0718, Accuracy: 9761/10000 (98Test set:
Average loss: 0.0637, Accuracy: 9802/10000 (98Test set: Average loss:
0.0581, Accuracy: 9818/10000 (98Test set: Average loss: 0.0544,
Accuracy: 9823/10000 (98

### MNIST 

\[ frame=lines, framesep=2mm, baselinestretch=1.2, fontsize=, linenos \]
[python]{}

Test set: Average loss: 0.3054, Accuracy: 9144/10000 (91Test set:
Average loss: 0.2895, Accuracy: 9206/10000 (92Test set: Average loss:
0.2879, Accuracy: 9183/10000 (92Test set: Average loss: 0.2796,
Accuracy: 9180/10000 (92Test set: Average loss: 0.2756, Accuracy:
9209/10000 (92Test set: Average loss: 0.2742, Accuracy: 9205/10000
(92Test set: Average loss: 0.2761, Accuracy: 9224/10000 (92Test set:
Average loss: 0.2718, Accuracy: 9235/10000 (92Test set: Average loss:
0.2721, Accuracy: 9221/10000 (92Test set: Average loss: 0.2702,
Accuracy: 9244/10000 (92
