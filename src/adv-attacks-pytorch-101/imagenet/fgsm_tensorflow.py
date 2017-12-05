# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    """
    Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Length of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


class InceptionModel(object):
    """Inception Model"""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(x_input,
                                                   num_classes=self.num_classes,
                                                   is_training=False,
                                                   reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # batch_shape_ = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    batch_shape_ = [1, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1000
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(shape=batch_shape_, dtype=tf.float32)
        model = InceptionModel(num_classes)
        probs = model(x_input=x_input)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(scaffold=tf.train.Scaffold(saver=saver),
                                                       checkpoint_filename_with_path=FLAGS.checkpoint_path,
                                                       master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(FLAGS.input_dir, batch_shape_):
                probs_ = sess.run(probs, feed_dict={x_input: images})
                logging.warning('probs_ = {}'.format(probs_))

    logging.warning('Terminated')
    logging.warning('Terminated')
    logging.warning('Terminated')
