"""
Handles images
"""
import glob
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image


def load_images(input_dir, batch_shape):
    """
    Read png images from input directory in batches.
    Args:
        input_dir (): input directory
        batch_shape (): shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
        filenames: list file names without path of each image
          Length of this list could be less than batch_size, in this case only
          first few images of the result are elements of the minibatch.
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    true_labels = []
    target_classes = []
    idx = 0
    batch_size = batch_shape[0]

    script_dir = os.path.dirname(os.path.abspath(__file__)) + '/' + input_dir

    labels = pd.read_csv(script_dir + '/dev_dataset.csv',
                         index_col=['ImageId'],
                         usecols=['ImageId', 'TrueLabel', 'TargetClass'])

    for filename in glob.glob(script_dir + '/*.png'):
        im = Image.open(filename)
        image = np.array(im.convert('RGB')).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        image_id = filename.split('/')[-1][:-4]
        filenames.append(os.path.basename(image_id))
        true_labels.append(labels.loc[image_id]['TrueLabel'])
        target_classes.append(labels.loc[image_id]['TargetClass'])
        idx += 1
        if idx == batch_size:
            yield images.reshape(batch_size, -1), np.array(true_labels), np.array(target_classes)
            filenames = []
            true_labels = []
            target_classes = []
            images = np.zeros(batch_shape)
            idx = 0
    yield images.reshape(batch_size, -1), np.array(true_labels), np.array(target_classes)


def save_images(images, filenames, output_dir):
    """
    Saves images to the output directory.
    Args:
        images (): array with minibatch of images
        filenames (): list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir (): directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')
